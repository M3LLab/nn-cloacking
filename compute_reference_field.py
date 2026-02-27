"""
Reference field: same domain & absorbing layers as boundaries.py but with
uniform isotropic material everywhere (no cloak, no defect).

The result is saved to output/reference.npz in the same format as boundaries.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, rectangle_mesh

# ═══════════════════════════════════════════════════════════════════════
# 1.  Physical parameters  (identical to boundaries.py)
# ═══════════════════════════════════════════════════════════════════════

rho0 = 1600.0
cs   = 300.0
cp   = np.sqrt(3.0) * cs

mu   = rho0 * cs**2
lam  = rho0 * cp**2 - 2 * mu
nu   = lam / (2 * (lam + mu))

cR   = cs * (0.826 + 1.14 * nu) / (1 + nu)

f_star      = 2.0
lambda_star = 1.0
omega       = 2 * np.pi * f_star * cR / lambda_star

H = 4.305 * lambda_star
W = 12.5  * lambda_star

# ═══════════════════════════════════════════════════════════════════════
# 2.  Absorbing-layer parameters  (identical to boundaries.py)
# ═══════════════════════════════════════════════════════════════════════

L_pml    = 1.0 * lambda_star
xi_max   = 4.0
pml_pow  = 2

W_total = 2 * L_pml + W
H_total = L_pml + H

x_off = L_pml
y_off = L_pml

x_src_phys = 0.05 * W
x_src      = x_off + x_src_phys
y_top      = H_total

# ═══════════════════════════════════════════════════════════════════════
# 3.  Mesh  (identical to boundaries.py)
# ═══════════════════════════════════════════════════════════════════════

n_pml_x = 32
n_pml_y = 32
nx_phys = 400
ny_phys = 131

nx_total = n_pml_x + nx_phys + n_pml_x
ny_total = n_pml_y + ny_phys

meshio_mesh = rectangle_mesh(nx_total, ny_total, W_total, H_total)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'],
            ele_type='QUAD4')

# ═══════════════════════════════════════════════════════════════════════
# 4.  Damping profile  ξ(x)  (identical to boundaries.py)
# ═══════════════════════════════════════════════════════════════════════

def _xi_profile(x):
    d_left  = jnp.maximum(x_off - x[0], 0.0)
    d_right = jnp.maximum(x[0] - (x_off + W), 0.0)
    xi_x    = xi_max * (jnp.maximum(d_left, d_right) / L_pml) ** pml_pow

    d_bot   = jnp.maximum(y_off - x[1], 0.0)
    xi_y    = xi_max * (d_bot / L_pml) ** pml_pow

    return xi_x + xi_y

# ═══════════════════════════════════════════════════════════════════════
# 5.  Uniform isotropic stiffness  (C_iso everywhere, no cloak)
# ═══════════════════════════════════════════════════════════════════════

def C_iso():
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

# ═══════════════════════════════════════════════════════════════════════
# 6.  Source
# ═══════════════════════════════════════════════════════════════════════

sigma_src = 3.0 * (W / nx_phys)
F0        = 1.0

def top_surface(point):
    return jnp.isclose(point[1], H_total)

# ═══════════════════════════════════════════════════════════════════════
# 7.  FEM Problem  (vec = 4, uniform material)
# ═══════════════════════════════════════════════════════════════════════

class ReferenceProblem(Problem):
    def custom_init(self):
        num_cells = self.physical_quad_points.shape[0]
        num_quads = self.physical_quad_points.shape[1]
        # Uniform C0 everywhere
        C_all = jnp.broadcast_to(C0, (num_cells, num_quads, 2, 2, 2, 2))
        # Uniform rho0 everywhere
        rho_all = jnp.full((num_cells, num_quads), rho0)
        # Damping profile (varies spatially)
        xi_all = jax.vmap(jax.vmap(_xi_profile))(self.physical_quad_points)
        self.internal_vars = [C_all, rho_all, xi_all]

    def get_tensor_map(self):
        def stress(u_grad, C_q, _rho_q, xi_q):
            grad_R = u_grad[:2, :]
            grad_I = u_grad[2:, :]

            sig_R_undamped = jnp.einsum('ijkl,kl->ij', C_q, grad_R)
            sig_I_undamped = jnp.einsum('ijkl,kl->ij', C_q, grad_I)

            sig_R = sig_R_undamped - xi_q * sig_I_undamped
            sig_I = sig_I_undamped + xi_q * sig_R_undamped

            return jnp.concatenate([sig_R, sig_I], axis=0)

        return stress

    def get_mass_map(self):
        def inertia(u, _x, _C_q, rho_q, xi_q):
            u_R = u[:2]
            u_I = u[2:]

            m_R = -omega**2 * rho_q * (u_R + xi_q * u_I)
            m_I = -omega**2 * rho_q * (u_I - xi_q * u_R)

            return jnp.concatenate([m_R, m_I])

        return inertia

    def get_surface_maps(self):
        def traction(_u, x):
            g = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
            return jnp.array([0.0, g, 0.0, 0.0])

        return [traction]

# ═══════════════════════════════════════════════════════════════════════
# 8.  Boundary conditions  (identical to boundaries.py)
# ═══════════════════════════════════════════════════════════════════════

def bc_bottom(point):
    return jnp.isclose(point[1], 0.0)

def bc_left(point):
    return jnp.isclose(point[0], 0.0)

def bc_right(point):
    return jnp.isclose(point[0], W_total)

def zero(point):
    return 0.0

dirichlet_bc_info = [
    [bc_bottom, bc_bottom, bc_bottom, bc_bottom,
     bc_left,   bc_left,   bc_left,   bc_left,
     bc_right,  bc_right,  bc_right,  bc_right],
    [0, 1, 2, 3,
     0, 1, 2, 3,
     0, 1, 2, 3],
    [zero, zero, zero, zero,
     zero, zero, zero, zero,
     zero, zero, zero, zero],
]

# ═══════════════════════════════════════════════════════════════════════
# 9.  Assemble and solve
# ═══════════════════════════════════════════════════════════════════════

problem = ReferenceProblem(
    mesh          = mesh,
    vec           = 4,
    dim           = 2,
    ele_type      = 'QUAD4',
    dirichlet_bc_info = dirichlet_bc_info,
    location_fns  = [top_surface],
)

print("Solving reference system (no cloak, uniform material) …")
sol_list = solver(problem, solver_options={'umfpack_solver': {}})
u = sol_list[0]

# ═══════════════════════════════════════════════════════════════════════
# 10.  Save results
# ═══════════════════════════════════════════════════════════════════════

import os
os.makedirs("output", exist_ok=True)

np.savez("output/reference.npz",
         u=np.asarray(u),
         pts_x=np.asarray(mesh.points[:, 0]),
         pts_y=np.asarray(mesh.points[:, 1]),
         x_src=x_src, y_top=y_top,
         x_off=x_off, y_off=y_off,
         W=W, H=H,
         x_src_phys=x_src_phys,
         f_star=f_star)
print("Reference results saved → output/reference.npz")
