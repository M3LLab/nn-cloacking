"""Compute effective 4x4 stiffness tensor of a 2D unit cell via FEM homogenization
with periodic boundary conditions using JAX-FEM.

The unit cell is defined by an NxN binary pixel image where
1 = solid (cement) and 0 = void.

Output: 4x4 stiffness matrix in augmented Voigt notation
    [sigma_11, sigma_22, sigma_12, sigma_21] = C_eff @ [e_11, e_22, e_12, e_21]
where e_ij = du_i / dx_j (full displacement gradient, not symmetric strain).

Method
------
Fluctuation-based periodic homogenization on a Cauchy microscale:

    u(x) = H_macro . x + u_tilde(x)

where H_macro is the imposed macroscopic displacement gradient (2x2) and
u_tilde is Y-periodic.  For each of 4 unit basis gradients H^(k), solve the
cell problem

    int_Y  C(y) : (grad u_tilde + H^(k)) : grad v  dY = 0   for all periodic v

then compute the k-th column of C_eff via volume averaging:

    C_eff[:, k] = <sigma^(k)>_Y = (1/|Y|) int_Y C(y) : (grad u_tilde^(k) + H^(k)) dY

Note: with isotropic Cauchy microscale material, C_eff has Cauchy symmetry
(sigma_12 = sigma_21 for every load case).  To obtain a genuinely asymmetric
C_eff (chirality in the Cosserat sense), one would need micropolar DOFs or
higher-order polynomial loading — see the Cosserat homogenization report for
details.

Usage:
    python -m rayleigh_cloak.dataset.stiffness.chiral_homogenization [input.npy] [-o output.npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as onp
import jax
import jax.numpy as jnp
import scipy.sparse

from jax_fem.problem import Problem
from jax_fem.solver import solver as jax_fem_solver
from jax_fem.generate_mesh import Mesh

# ---------------------------------------------------------------------------
# Material parameters (cement, plane strain)
# ---------------------------------------------------------------------------
E_CEMENT = 30e9       # Pa
NU = 0.2
RHO_CEMENT = 2300.0   # kg/m^3
E_VOID_RATIO = 1e-6   # void stiffness = E_CEMENT * ratio (ersatz material)


# ---------------------------------------------------------------------------
# Structured triangular mesh on [0, 1]^2
# ---------------------------------------------------------------------------
def make_structured_tri_mesh(N: int):
    """Create (N+1)x(N+1) node grid with 2*N*N TRI3 elements.

    Node ordering: row-major, node(ix, iy) = iy * (N+1) + ix.
    Each pixel is split into two counter-clockwise triangles.
    """
    xs = onp.linspace(0, 1, N + 1)
    ys = onp.linspace(0, 1, N + 1)
    xx, yy = onp.meshgrid(xs, ys, indexing='xy')
    points = onp.stack([xx.ravel(), yy.ravel()], axis=1)  # ((N+1)^2, 2)

    cells = []
    for iy in range(N):
        for ix in range(N):
            n0 = iy * (N + 1) + ix        # bottom-left
            n1 = n0 + 1                    # bottom-right
            n2 = (iy + 1) * (N + 1) + ix  # top-left
            n3 = n2 + 1                    # top-right
            cells.append([n0, n1, n3])     # lower-right triangle
            cells.append([n0, n3, n2])     # upper-left triangle
    cells = onp.array(cells, dtype=onp.int32)
    return points, cells


# ---------------------------------------------------------------------------
# Periodic boundary condition P-matrix
# ---------------------------------------------------------------------------
def build_periodic_pmat(N: int, vec: int):
    """Build sparse DOF projection matrix tying periodic node pairs.

    Independent nodes: ix in [0, N-1], iy in [0, N-1]  (N*N nodes).
    Right edge (ix=N) -> ix=0, top edge (iy=N) -> iy=0, corners included.
    """
    num_nodes = (N + 1) ** 2
    num_dofs = num_nodes * vec

    # Map every node to its independent counterpart
    dep_to_ind = {}
    for iy in range(N + 1):
        for ix in range(N + 1):
            ix_ind = ix % N
            iy_ind = iy % N
            node = iy * (N + 1) + ix
            ind_node = iy_ind * (N + 1) + ix_ind
            if node != ind_node:
                dep_to_ind[node] = ind_node

    ind_nodes = sorted(set(range(num_nodes)) - set(dep_to_ind.keys()))
    ind_node_to_col = {n: i for i, n in enumerate(ind_nodes)}
    M = len(ind_nodes) * vec

    I, J, V = [], [], []
    for node in range(num_nodes):
        target = dep_to_ind.get(node, node)
        col_base = ind_node_to_col[target]
        for v in range(vec):
            I.append(node * vec + v)
            J.append(col_base * vec + v)
            V.append(1.0)

    P_mat = scipy.sparse.csr_array(
        (onp.array(V), (onp.array(I, dtype=onp.int32),
                        onp.array(J, dtype=onp.int32))),
        shape=(num_dofs, M),
    )
    return P_mat


# ---------------------------------------------------------------------------
# Material assignment at element centroids
# ---------------------------------------------------------------------------
def assign_material(pixel_image, points, cells, num_quads: int):
    """Return Young's modulus array (num_cells, num_quads) from pixel image.

    Image convention: row 0 = top of image = high y.
    """
    N = pixel_image.shape[0]
    centroids = points[cells].mean(axis=1)  # (num_cells, 2)
    ix = onp.clip((centroids[:, 0] * N).astype(int), 0, N - 1)
    iy = onp.clip((centroids[:, 1] * N).astype(int), 0, N - 1)
    is_material = pixel_image[N - 1 - iy, ix].astype(bool)
    E_per_cell = onp.where(is_material, E_CEMENT, E_CEMENT * E_VOID_RATIO)
    return onp.repeat(E_per_cell[:, None], num_quads, axis=1)


# ---------------------------------------------------------------------------
# FEM Problem for periodic homogenization
# ---------------------------------------------------------------------------
class HomogenizationProblem(Problem):
    """Fluctuation-based periodic unit-cell problem.

    Finds periodic u_tilde such that:
        int C : (grad(u_tilde) + H_macro) : grad(v) dOmega = 0
    for all periodic test functions v.

    Class attributes (set before __init__):
        _E_field    : (num_cells, num_quads) Young's modulus
        _H_macro    : (2, 2) macroscopic displacement gradient
        _nu         : Poisson's ratio
    """
    _E_field = None
    _H_macro = None
    _nu = NU

    def custom_init(self):
        H_macro_field = onp.broadcast_to(
            self._H_macro[None, None, :, :],
            (self.num_cells, self.fes[0].num_quads, 2, 2),
        ).copy()
        self.internal_vars = [self._E_field, H_macro_field]

    def get_tensor_map(self):
        nu = self._nu

        def tensor_map(u_grad, E, H_macro):
            # u_grad: (2,2) fluctuation gradient
            # E: ()  Young's modulus at this quad point
            # H_macro: (2,2) imposed macroscopic gradient
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            I = jnp.eye(2)
            C = (lam * jnp.einsum('ij,kl->ijkl', I, I)
                 + mu * (jnp.einsum('ik,jl->ijkl', I, I)
                         + jnp.einsum('il,jk->ijkl', I, I)))
            total_grad = u_grad + H_macro
            stress = jnp.einsum('ijkl,kl->ij', C, total_grad)
            return stress

        return tensor_map


# ---------------------------------------------------------------------------
# Post-processing: volume-averaged stress
# ---------------------------------------------------------------------------
def compute_average_stress(problem, sol, H_macro, E_field):
    """Compute <sigma> = (1/|Y|) int_Y sigma dY using full gradient."""
    fe = problem.fes[0]
    nu = problem._nu

    cell_sol = sol[fe.cells]  # (num_cells, nodes_per_cell, vec)
    # u_grad[c,q,i,j] = sum_n cell_sol[c,n,i] * shape_grads[c,q,n,j]
    u_grads = jnp.einsum('cni,cqnj->cqij', cell_sol, fe.shape_grads)
    total_grads = u_grads + H_macro[None, None, :, :]

    E = jnp.array(E_field)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    I = jnp.eye(2)
    C_lam = jnp.einsum('ij,kl->ijkl', I, I)
    C_mu = jnp.einsum('ik,jl->ijkl', I, I) + jnp.einsum('il,jk->ijkl', I, I)
    stress = (lam[:, :, None, None, None, None] * C_lam[None, None]
              + mu[:, :, None, None, None, None] * C_mu[None, None])
    stress = jnp.einsum('cqijkl,cqkl->cqij', stress, total_grads)

    JxW = fe.JxW  # (num_cells, num_quads)
    total_area = jnp.sum(JxW)
    avg_stress = jnp.sum(stress * JxW[:, :, None, None], axis=(0, 1)) / total_area
    return avg_stress


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
VOIGT_LABELS = ['e11', 'e22', 'e12', 'e21']
SIGMA_LABELS = ['sigma_11', 'sigma_22', 'sigma_12', 'sigma_21']

# 4 load cases: unit macroscopic displacement gradients
# H_macro[i,j] = du_i/dx_j
LOAD_CASES = [
    onp.array([[1., 0.], [0., 0.]]),   # e11: du1/dx1 = 1
    onp.array([[0., 0.], [0., 1.]]),   # e22: du2/dx2 = 1
    onp.array([[0., 1.], [0., 0.]]),   # e12: du1/dx2 = 1
    onp.array([[0., 0.], [1., 0.]]),   # e21: du2/dx1 = 1
]


def compute_effective_stiffness(pixel_image_path, output_path=None, verbose=True):
    """Compute 4x4 effective stiffness in augmented Voigt notation.

    Parameters
    ----------
    pixel_image_path : str or Path
        Path to NxN binary pixel image (.npy).
    output_path : str or Path, optional
        If given, save C_eff and rho_eff to this .npz file.
    verbose : bool
        Print progress and results.

    Returns
    -------
    C_eff : (4, 4) ndarray
        Effective stiffness in augmented Voigt notation (Pa).
    rho_eff : float
        Effective density (kg/m^3), rule-of-mixtures estimate.
    """
    pixel_image = onp.load(pixel_image_path)
    N = pixel_image.shape[0]
    assert pixel_image.shape == (N, N), f"Expected square image, got {pixel_image.shape}"
    vf = pixel_image.astype(float).mean()
    if verbose:
        print(f"Loaded {pixel_image_path}: {N}x{N}, volume fraction = {vf:.3f}")

    # --- Mesh ---
    points, cells = make_structured_tri_mesh(N)
    mesh = Mesh(points, cells, ele_type='TRI3')

    # --- Material (TRI3 with 1 gauss point) ---
    E_field = assign_material(pixel_image, points, cells, num_quads=1)

    # --- Periodic BCs via projection matrix ---
    P_mat = build_periodic_pmat(N, vec=2)

    # --- Pin corner node to remove rigid body translation ---
    def corner(point):
        return jnp.isclose(point[0], 0., atol=1e-5) & jnp.isclose(point[1], 0., atol=1e-5)

    dirichlet_bc_info = [[corner, corner], [0, 1],
                         [lambda p: 0., lambda p: 0.]]

    # --- Solve 4 load cases ---
    C_eff = onp.zeros((4, 4))

    for col, H_macro in enumerate(LOAD_CASES):
        if verbose:
            print(f"\nLoad case {col + 1}/4: {VOIGT_LABELS[col]} = 1")

        HomogenizationProblem._H_macro = H_macro
        HomogenizationProblem._E_field = E_field

        problem = HomogenizationProblem(
            mesh=mesh,
            vec=2,
            dim=2,
            ele_type='TRI3',
            dirichlet_bc_info=dirichlet_bc_info,
        )
        problem.P_mat = P_mat

        sol_list = jax_fem_solver(problem, solver_options={'umfpack_solver': {}})
        sol = sol_list[0]  # (num_nodes, 2)

        avg_stress = compute_average_stress(problem, sol, H_macro, E_field)

        # Augmented Voigt: [sigma_11, sigma_22, sigma_12, sigma_21]
        C_eff[0, col] = float(avg_stress[0, 0])
        C_eff[1, col] = float(avg_stress[1, 1])
        C_eff[2, col] = float(avg_stress[0, 1])
        C_eff[3, col] = float(avg_stress[1, 0])

        if verbose:
            print(f"  sigma_11={avg_stress[0,0]:.6e}  sigma_22={avg_stress[1,1]:.6e}  "
                  f"sigma_12={avg_stress[0,1]:.6e}  sigma_21={avg_stress[1,0]:.6e}")

    # --- Output ---
    rho_eff = vf * RHO_CEMENT

    if verbose:
        print("\n" + "=" * 60)
        print("Effective stiffness C_eff (4x4, augmented Voigt, Pa):")
        print("  Rows: [sigma_11, sigma_22, sigma_12, sigma_21]")
        print("  Cols: [e_11,     e_22,     e_12,     e_21    ]")
        print("-" * 60)
        for i in range(4):
            print("  " + "  ".join(f"{C_eff[i, j]:12.4e}" for j in range(4)))
        print("=" * 60)
        print(f"\nEffective density: {rho_eff:.1f} kg/m^3  (vf = {vf:.3f})")

    if output_path:
        onp.savez(output_path, C_eff=C_eff, rho_eff=rho_eff,
                  volume_fraction=vf, pixel_image_path=str(pixel_image_path))
        if verbose:
            print(f"Saved to {output_path}")

    return C_eff, rho_eff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute effective stiffness of a 2D unit cell via FEM homogenization")
    parser.add_argument('input', nargs='?',
                        default='output/ca_chiral/dataset/seed_0.npy',
                        help='Path to NxN binary pixel image (.npy)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output path (.npz)')
    args = parser.parse_args()

    compute_effective_stiffness(args.input, args.output)
