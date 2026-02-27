"""
Symmetrized triangular elastic cloak with Rayleigh-damping absorbing layers.

Based on:
  - "Cloaking Rayleigh waves via symmetrized elastic tensors" (Chatzopoulos et al., 2023)
  - "A Simple Numerical Absorbing Layer Method in Elastodynamics" (Semblat et al., 2010)

Absorbing-layer strategy
========================
The frequency-domain elastodynamic equation with Rayleigh damping is:

    (K + iωC − ω²M) u = f,   where  C = a₀M + a₁K   (Rayleigh damping)

Expanding:
    (1 + iωa₁)K·u − ω²(1 − ia₀/ω)M·u = f

Since JAX-FEM operates on real-valued fields only, we split u = u_R + i·u_I
into real and imaginary parts.  This gives the coupled real system:

    K·u_R − ω a₁ K·u_I  −  ω²ρ u_R − ω a₀ ρ u_I  =  f_R
    K·u_I + ω a₁ K·u_R  −  ω²ρ u_I + ω a₀ ρ u_R  =  f_I

We encode this with vec=4:  DOFs = [Re(uₓ), Re(u_y), Im(uₓ), Im(u_y)].

Choosing a₀ = ξω and a₁ = ξ/ω (so both give damping ratio ξ at ω) simplifies
the coupling coefficient to just ξ(x) everywhere, with ξ increasing from 0 at
the PML/physical interface to ξ_max at the outer boundary (quadratic ramp).

JAX-FEM hooks used
==================
  • custom_init   – precompute C_eff, ρ_eff, ξ at every quadrature point
  • get_tensor_map – stiffness + stiffness-proportional damping coupling
  • get_mass_map   – inertia + mass-proportional damping coupling
  • get_surface_maps – Rayleigh-wave point source (real part only)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, rectangle_mesh

# ═══════════════════════════════════════════════════════════════════════
# 1.  Physical parameters  (identical to the original triangle.py)
# ═══════════════════════════════════════════════════════════════════════

rho0 = 1600.0                          # mass density  [kg/m³]
cs   = 300.0                           # shear wave speed  [m/s]
cp   = np.sqrt(3.0) * cs               # pressure wave speed

mu   = rho0 * cs**2
lam  = rho0 * cp**2 - 2 * mu
nu   = lam / (2 * (lam + mu))          # Poisson's ratio

cR   = cs * (0.826 + 1.14 * nu) / (1 + nu)   # Rayleigh wave speed

# Normalized frequency / wavelength
f_star      = 2.0
lambda_star = 1.0
omega       = 2 * np.pi * f_star * cR / lambda_star

# Physical domain size  (same as the paper)
H = 4.305 * lambda_star                # depth
W = 12.5  * lambda_star                # width

# Cloak geometry
a   = 0.0774 * H                       # inner triangle depth
b   = 3 * a                            # outer triangle depth
c   = 0.309 * H / 2.0                  # half-width at surface

# ═══════════════════════════════════════════════════════════════════════
# 2.  Absorbing-layer (PML-region) parameters
# ═══════════════════════════════════════════════════════════════════════

L_pml    = 1.0 * lambda_star           # thickness of each absorbing layer
xi_max   = 4.0                         # peak damping ratio at outer edge
pml_pow  = 2                           # ramp exponent  (quadratic = 2)

# Extended domain with absorbing layers on left, right, bottom
#   Physical domain occupies  x ∈ [L_pml, L_pml+W],  y ∈ [L_pml, L_pml+H]
#   Free surface at y = L_pml + H  (top edge — no PML)
W_total = 2 * L_pml + W
H_total = L_pml + H

# Offsets so that physical coordinates map back to original system
x_off = L_pml
y_off = L_pml

# Source location in the EXTENDED mesh coordinate system
x_src_phys = 0.05 * W                  # in original coords
x_src      = x_off + x_src_phys        # in extended-mesh coords
y_top      = H_total                   # free-surface y-coordinate

# Cloak centre in extended coordinates
x_c = x_off + W / 2.0

# ═══════════════════════════════════════════════════════════════════════
# 3.  Mesh
# ═══════════════════════════════════════════════════════════════════════

n_pml_x = 32                           # elements across each lateral PML
n_pml_y = 32                           # elements across the bottom PML
nx_phys = 400                          # elements across physical width
ny_phys = 131                          # elements across physical depth

nx_total = n_pml_x + nx_phys + n_pml_x
ny_total = n_pml_y + ny_phys

meshio_mesh = rectangle_mesh(nx_total, ny_total, W_total, H_total)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'],
            ele_type='QUAD4')

# ═══════════════════════════════════════════════════════════════════════
# 4.  Damping profile  ξ(x)
# ═══════════════════════════════════════════════════════════════════════

def _xi_profile(x):
    """Compute the local damping ratio ξ(x).

    ξ = 0 inside the physical domain and ramps quadratically
    into each absorbing-layer region (left, right, bottom).
    In PML corner regions both directions contribute additively.
    """
    # --- lateral (x-direction) attenuation ---
    d_left  = jnp.maximum(x_off - x[0], 0.0)          # distance into left PML
    d_right = jnp.maximum(x[0] - (x_off + W), 0.0)    # distance into right PML
    xi_x    = xi_max * (jnp.maximum(d_left, d_right) / L_pml) ** pml_pow

    # --- vertical (y-direction) attenuation – bottom only ---
    d_bot   = jnp.maximum(y_off - x[1], 0.0)           # distance into bottom PML
    xi_y    = xi_max * (d_bot / L_pml) ** pml_pow

    # additive combination (works well for corner regions)
    return xi_x + xi_y

# ═══════════════════════════════════════════════════════════════════════
# 5.  Triangular-cloak coordinate transformation
# ═══════════════════════════════════════════════════════════════════════

def _in_cloak(x):
    """True inside the cloak annulus (uses EXTENDED mesh coords)."""
    depth = y_top - x[1]                # depth from free surface
    r     = jnp.abs(x[0] - x_c) / c
    d2    = b * (1.0 - r)
    d1    = a * (1.0 - r)
    return (r <= 1.0) & (depth >= d1) & (depth <= d2)

def _in_defect(x):
    """True inside the hidden void (uses EXTENDED mesh coords)."""
    depth = y_top - x[1]
    r     = jnp.abs(x[0] - x_c) / c
    d1    = a * (1.0 - r)
    return (r <= 1.0) & (depth >= 0.0) & (depth <= d1)

def F_tensor(x):
    """Deformation gradient of the triangular transformation."""
    sign = jnp.sign(x[0] - x_c)
    F21  = sign * a / c
    F22  = (b - a) / b
    F_cloak = jnp.array([[1.0, 0.0],
                          [F21, F22]])
    return jnp.where(_in_cloak(x), F_cloak, jnp.eye(2))

# ═══════════════════════════════════════════════════════════════════════
# 6.  Effective material tensors
# ═══════════════════════════════════════════════════════════════════════

def C_iso():
    """4th-order isotropic stiffness tensor  C₀[i,j,k,l]."""
    C = jnp.zeros((2, 2, 2, 2))
    delta = jnp.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    C = C.at[i, j, k, l].set(
                        lam * delta[i, j] * delta[k, l]
                        + mu * (delta[i, k] * delta[j, l]
                                + delta[i, l] * delta[j, k])
                    )
    return C

C0 = C_iso()

_eps_void = 1e-4          # near-zero stiffness / density for the defect

def C_eff(x):
    """Position-dependent effective stiffness tensor."""
    F   = F_tensor(x)
    J   = jnp.linalg.det(F)
    # Cosserat transformed tensor
    Cnew = jnp.einsum('iI,kK,IjKl->ijkl', F, F, C0) / J
    # Full Cauchy symmetrisation (arithmetic-mean of 4 minor-index permutations)
    Csym = (Cnew
            + jnp.transpose(Cnew, (1, 0, 2, 3))
            + jnp.transpose(Cnew, (0, 1, 3, 2))
            + jnp.transpose(Cnew, (1, 0, 3, 2))) / 4.0
    C_void = _eps_void * C0
    return jnp.where(_in_defect(x), C_void,
                     jnp.where(_in_cloak(x), Csym, C0))

def rho_eff(x):
    """Position-dependent effective density."""
    F   = F_tensor(x)
    J   = jnp.linalg.det(F)
    rho_cloak = rho0 / J
    rho_void  = _eps_void * rho0
    return jnp.where(_in_defect(x), rho_void,
                     jnp.where(_in_cloak(x), rho_cloak, rho0))

# ═══════════════════════════════════════════════════════════════════════
# 7.  Source
# ═══════════════════════════════════════════════════════════════════════

sigma_src = 3.0 * (W / nx_phys)        # Gaussian half-width  (~3 elements)
F0        = 1.0                         # force amplitude (Pa)

def top_surface(point):
    """Selects the free (top) surface  y = H_total."""
    return jnp.isclose(point[1], H_total)

# ═══════════════════════════════════════════════════════════════════════
# 8.  FEM Problem  (vec = 4 :  Re(uₓ), Re(u_y), Im(uₓ), Im(u_y))
# ═══════════════════════════════════════════════════════════════════════

class RayleighCloakProblem(Problem):
    """Frequency-domain elastodynamics with Rayleigh-damping absorbing layers.

    DOF ordering per node:  [Re(uₓ), Re(u_y), Im(uₓ), Im(u_y)]
    """

    def custom_init(self):
        # Precompute material data at every quadrature point
        # physical_quad_points shape: (num_cells, num_quads, dim)
        self.internal_vars = [
            jax.vmap(jax.vmap(C_eff))(self.physical_quad_points),    # (nc, nq, 2,2,2,2)
            jax.vmap(jax.vmap(rho_eff))(self.physical_quad_points),  # (nc, nq)
            jax.vmap(jax.vmap(_xi_profile))(self.physical_quad_points),  # (nc, nq)
        ]

    # -----------------------------------------------------------------
    def get_tensor_map(self):
        """σ(∇u) — stiffness + stiffness-proportional damping coupling.

        u_grad shape: (4, 2)   →  rows 0-1 = ∇u_R,  rows 2-3 = ∇u_I
        Returns  stress  shape (4, 2).
        """
        def stress(u_grad, C_q, _rho_q, xi_q):
            grad_R = u_grad[:2, :]       # (2, 2)  – ∂Re(u)/∂x
            grad_I = u_grad[2:, :]       # (2, 2)  – ∂Im(u)/∂x

            sig_R_undamped = jnp.einsum('ijkl,kl->ij', C_q, grad_R)
            sig_I_undamped = jnp.einsum('ijkl,kl->ij', C_q, grad_I)

            # With stiffness-proportional damping  (coupling coeff = ξ):
            #   σ_R = C:ε_R − ξ C:ε_I
            #   σ_I = C:ε_I + ξ C:ε_R
            sig_R = sig_R_undamped - xi_q * sig_I_undamped
            sig_I = sig_I_undamped + xi_q * sig_R_undamped

            return jnp.concatenate([sig_R, sig_I], axis=0)   # (4, 2)

        return stress

    # -----------------------------------------------------------------
    def get_mass_map(self):
        """Inertia + mass-proportional damping coupling.

        u shape: (4,)  →  u[:2] = u_R,  u[2:] = u_I
        Returns  (4,).
        """
        def inertia(u, _x, _C_q, rho_q, xi_q):
            u_R = u[:2]
            u_I = u[2:]

            # −ω²ρ u_R − ξ ω² ρ u_I   (real part)
            # −ω²ρ u_I + ξ ω² ρ u_R   (imaginary part)
            m_R = -omega**2 * rho_q * (u_R + xi_q * u_I)
            m_I = -omega**2 * rho_q * (u_I - xi_q * u_R)

            return jnp.concatenate([m_R, m_I])

        return inertia

    # -----------------------------------------------------------------
    def get_surface_maps(self):
        """Surface traction — real vertical point load, no imaginary component."""
        def traction(_u, x):
            g = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
            # Only the real-part vertical component  Re(u_y)
            return jnp.array([0.0, g, 0.0, 0.0])

        return [traction]

# ═══════════════════════════════════════════════════════════════════════
# 9.  Boundary conditions
# ═══════════════════════════════════════════════════════════════════════
#
# Zero displacement (real & imaginary) on the OUTER edges of the PML:
#   • bottom  (y = 0)
#   • left    (x = 0)
#   • right   (x = W_total)
# The top surface (y = H_total) remains FREE (stress-free for Rayleigh waves).
# ═══════════════════════════════════════════════════════════════════════

def bc_bottom(point):
    return jnp.isclose(point[1], 0.0)

def bc_left(point):
    return jnp.isclose(point[0], 0.0)

def bc_right(point):
    return jnp.isclose(point[0], W_total)

def zero(point):
    return 0.0

# All 4 DOFs fixed on bottom, left, and right outer PML boundaries
dirichlet_bc_info = [
    # location_fns  (one per constrained DOF)
    [bc_bottom, bc_bottom, bc_bottom, bc_bottom,
     bc_left,   bc_left,   bc_left,   bc_left,
     bc_right,  bc_right,  bc_right,  bc_right],
    # DOF indices
    [0, 1, 2, 3,
     0, 1, 2, 3,
     0, 1, 2, 3],
    # prescribed values
    [zero, zero, zero, zero,
     zero, zero, zero, zero,
     zero, zero, zero, zero],
]

# ═══════════════════════════════════════════════════════════════════════
# 10.  Assemble and solve
# ═══════════════════════════════════════════════════════════════════════

problem = RayleighCloakProblem(
    mesh          = mesh,
    vec           = 4,
    dim           = 2,
    ele_type      = 'QUAD4',
    dirichlet_bc_info = dirichlet_bc_info,
    location_fns  = [top_surface],
)

print("Solving frequency-domain system with absorbing layers …")
sol_list = solver(problem, solver_options={'umfpack_solver': {}})
u = sol_list[0]            # shape (num_nodes, 4)

# ═══════════════════════════════════════════════════════════════════════
# 11.  Save results for plotting
# ═══════════════════════════════════════════════════════════════════════

import os
os.makedirs("output", exist_ok=True)

np.savez("output/results.npz",
         u=np.asarray(u),
         pts_x=np.asarray(mesh.points[:, 0]),
         pts_y=np.asarray(mesh.points[:, 1]),
         x_src=x_src, y_top=y_top,
         x_off=x_off, y_off=y_off,
         W=W, H=H,
         x_src_phys=x_src_phys,
         f_star=f_star)
print("Results saved → output/results.npz")
print("Run `python plot_results.py` to generate plots.")

# ═══════════════════════════════════════════════════════════════════════
# 12.  Autograd-compatible loss function  (example)
# ═══════════════════════════════════════════════════════════════════════

def cloaking_loss(problem_instance, u_sol):
    """Example loss: mean |u|² on the surface downstream of the cloak.

    This function is differentiable w.r.t. any JAX parameter that
    flows into the Problem (e.g. cloak geometry a, b, c, or material
    properties) because the entire forward solve is built on JAX ops.
    """
    # Surface nodes downstream of cloak  (right half of physical domain)
    surface_mask = (
        jnp.isclose(mesh.points[:, 1], y_top) &
        (mesh.points[:, 0] > x_c + c)
    )
    # Sum of squared displacement magnitudes on those nodes
    u_R_sq = u_sol[:, 0]**2 + u_sol[:, 1]**2
    u_I_sq = u_sol[:, 2]**2 + u_sol[:, 3]**2
    energy  = jnp.where(surface_mask, u_R_sq + u_I_sq, 0.0)
    return jnp.sum(energy) / jnp.maximum(jnp.sum(surface_mask.astype(float)), 1.0)

print("\n✓  Script finished.  Absorbing layers active on left / right / bottom.")
print(f"   Domain:  {W_total:.2f} × {H_total:.2f}   (physical {W:.2f} × {H:.2f})")
print(f"   PML thickness = {L_pml:.3f},  ξ_max = {xi_max},  ramp power = {pml_pow}")
print(f"   Mesh:  {nx_total} × {ny_total} = {nx_total*ny_total} quads")