"""
Implementation of the symmetrized triangular cloak from:
 "Symmetrized triangular elastic cloak with broadband Rayleigh wave cloaking" (2024)
  by Y. Li, S. Guenneau, and J. B. Pendry
The transformation is defined by a piecewise linear function F(x) that maps the
physical coordinates to virtual coordinates. The effective material properties are
computed using the standard transformation elastodynamics formulas, and then
used in a frequency-domain FEM problem to solve for the displacement field u.

frequency-domain elastodynamics equation with source f_ext:
K·u − ω²M·u − f_ext = 0

JAX-FEM's weak-form residual is:
R(u; v) = ∫_Ω σ(u):ε(v) dΩ  +  ∫_Ω mass_map(u)·v dΩ  +  ∫_Γ surface_map(u,x)·v dΓ  = 0
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, rectangle_mesh

# ---------------------------
# 1. Physical parameters
# ---------------------------

rho0 = 1600.0
cs = 300.0
cp = np.sqrt(3.0) * cs

mu = rho0 * cs**2
lam = rho0 * cp**2 - 2*mu
nu = lam / (2*(lam+mu))

cR = cs * (0.826 + 1.14*nu)/(1+nu)

# Normalized frequency f*
f_star = 1.0
lambda_star = 1.0
omega = 2*np.pi * f_star * cR / lambda_star

# Domain
H = 4.305 * lambda_star
W = 12.5 * lambda_star

a = 0.0774 * H
b = 3*a
c = 0.309 * H / 2.0
x_c = W / 2.0              # horizontal centre of the cloak

# ---------------------------
# 2. Mesh
# ---------------------------

nx, ny = 400, 131
meshio_mesh = rectangle_mesh(nx, ny, W, H)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'], ele_type='QUAD4')

# ---------------------------
# 3. Triangular transform F
# ---------------------------

def _in_cloak(x):
    """True inside the cloak annulus between the inner (z1) and outer (z2) V-boundaries.

    Both boundaries meet the free surface at |x1 - x_c| = c and narrow with depth.
    Outer depth: d2(x) = b*(1 - |x1-xc|/c)
    Inner depth: d1(x) = a*(1 - |x1-xc|/c)
    Cloak region: d1 <= depth <= d2  (widest at surface, apex pointing down).
    """
    depth = H - x[1]
    r = jnp.abs(x[0] - x_c) / c        # normalised lateral distance, 0..1 at surface
    d2 = b * (1.0 - r)                  # outer boundary depth
    d1 = a * (1.0 - r)                  # inner boundary depth
    return (r <= 1.0) & (depth >= d1) & (depth <= d2)

def _in_defect(x):
    """True inside the inner void region that the cloak conceals (0 <= depth <= d1)."""
    depth = H - x[1]
    r = jnp.abs(x[0] - x_c) / c
    d1 = a * (1.0 - r)
    return (r <= 1.0) & (depth >= 0.0) & (depth <= d1)

def F_tensor(x):
    x1 = x[0]
    sign = jnp.sign(x1 - x_c)          # sign relative to cloak centre
    F21 = sign * a / c
    F22 = (b - a) / b

    F_cloak = jnp.array([[1.0, 0.0],
                          [F21, F22]])
    # return identity (original material) outside the cloak region
    return jnp.where(_in_cloak(x), F_cloak, jnp.eye(2))

# ---------------------------
# 4. Effective tensor
# ---------------------------

def C_iso():
    C = jnp.zeros((2,2,2,2))
    delta = jnp.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    C = C.at[i,j,k,l].set(
                        lam*delta[i,j]*delta[k,l]
                        + mu*(delta[i,k]*delta[j,l]
                              + delta[i,l]*delta[j,k])
                    )
    return C

C0 = C_iso()

_eps_void = 1e-4   # near-zero stiffness/density scaling for the defect cavity

def C_eff(x):
    F = F_tensor(x)
    J = jnp.linalg.det(F)

    # Cosserat transformed tensor: C'[i,j,k,l] = (1/J) * F[i,I] * F[k,K] * C0[I,j,K,l]
    Cnew = jnp.einsum('iI,kK,IjKl->ijkl', F, F, C0) / J

    # Full Cauchy symmetrization: average over all 4 minor-index permutations.
    # Enforces both minor symmetries and preserves the major symmetry of Cnew.
    Csym = (Cnew
            + jnp.transpose(Cnew, (1, 0, 2, 3))   # swap i↔j
            + jnp.transpose(Cnew, (0, 1, 3, 2))   # swap k↔l
            + jnp.transpose(Cnew, (1, 0, 3, 2))   # swap both
           ) / 4.0

    C_void = _eps_void * C0
    return jnp.where(_in_defect(x), C_void, jnp.where(_in_cloak(x), Csym, C0))

def rho_eff(x):
    F = F_tensor(x)
    J = jnp.linalg.det(F)
    rho_cloak = rho0 / J
    rho_void = _eps_void * rho0
    return jnp.where(_in_defect(x), rho_void, jnp.where(_in_cloak(x), rho_cloak, rho0))

# --------------------------
# Incident wave (Rayleigh wave surface point source)
# --------------------------

# Point source located at 0.35*W from the left edge of the model,
# on the free (top) surface at y = H.
x_src = 0.35 * W

# Gaussian half-width for spatial smoothing of the point source
# (~3 mesh-element widths keeps the source above the Nyquist limit).
sigma_src = 3.0 * (W / nx)

# Normalized vertical force amplitude
F0 = 1.0

def top_surface(point):
    """Selects all faces on the free (top) surface at y = H."""
    return jnp.isclose(point[1], H)


# ---------------------------
# 5. FEM Problem
# ---------------------------

class RayleighCloakProblem(Problem):
    def custom_init(self):
        # Precompute position-dependent material at all quadrature points.
        # physical_quad_points shape: (num_cells, num_quads, dim)
        # Stored as internal_vars: each entry (num_cells, num_quads, ...)
        self.internal_vars = [
            jax.vmap(jax.vmap(C_eff))(self.physical_quad_points),   # (num_cells, num_quads, 2, 2, 2, 2)
            jax.vmap(jax.vmap(rho_eff))(self.physical_quad_points),  # (num_cells, num_quads)
        ]

    def get_tensor_map(self):
        # tensor_map(u_grad, *internal_vars_at_quad) -> stress
        # Called via vmap over quads; u_grad shape: (vec, dim) = (2, 2)
        def stress(u_grad, C_at_quad, _rho_at_quad):
            # strain = 0.5 * (u_grad + u_grad.T)
            strain = u_grad
            return jnp.einsum('ijkl,kl->ij', C_at_quad, strain)
        return stress

    def get_mass_map(self):
        # mass_map(u, x, *internal_vars_at_quad) -> body-like term
        # Called via vmap over quads; u shape: (vec,) = (2,)
        # Contributes -omega^2 * rho * u to the residual (frequency-domain inertia)
        def inertia(u, _x, _C_at_quad, rho_at_quad):
            return -omega**2 * rho_at_quad * u
        return inertia

    def get_surface_maps(self):
        # Surface traction for the Rayleigh wave point source.
        # JAX-FEM sign convention: surface_map returns -t_ext, so that the
        # contribution ∫ surface_map·v dΓ is added to the residual
        # (R = K*u - ω²M*u + ∫ surface_map·v dΓ = 0).
        # For a downward unit force t_ext = [0, -F0*g]:  surface_map = [0, +F0*g].
        def traction(_u, x):
            # Gaussian approximation of a vertical point load at x_src.
            g = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
            return jnp.array([0.0, g])
        return [traction]

# ---------------------------
# 6. Boundary conditions
# ---------------------------

def bottom(point):
    return jnp.isclose(point[1], 0.0)

def zero(point):
    return 0.0

dirichlet_bc_info = [[bottom, bottom], [0, 1], [zero, zero]]

problem = RayleighCloakProblem(
    mesh=mesh,
    vec=2,
    dim=2,
    ele_type='QUAD4',
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=[top_surface]
)

# ---------------------------
# 7. Solve
# ---------------------------

sol_list = solver(problem, solver_options={'umfpack_solver': {}})
u = sol_list[0]

# ---------------------------
# 8. Postprocess
# ---------------------------

import matplotlib.pyplot as plt
from datetime import datetime

ux = u[:,0]
uy = u[:,1]
mag = np.sqrt(ux**2 + uy**2)

plt.figure(figsize=(12,4))
print(mag.shape, mag.min(), mag.max())
plt.tricontourf(mesh.points[:,0],
                mesh.points[:,1],
                mag,
                levels=100)
# mark source point with a red star
plt.plot(x_src, H, 'r*', markersize=20)
plt.colorbar()
plt.title(f"Symmetrized Triangular Cloak (f*={f_star})")
plt.savefig(f"output/cloak_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
print("Plot saved to", f"output/cloak_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")