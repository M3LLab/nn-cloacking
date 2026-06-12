"""High-fidelity periodic homogenization (TRI6 / factor-once / configurable N).

Drop-in alternative to :func:`dataset.stiffness.calc_fem.compute_effective_stiffness`
for the bulk pipeline. It returns the SAME 4x4 augmented-Voigt ``C_eff``
(rows/cols ordered ``[sigma_11, sigma_22, sigma_12, sigma_21]`` vs the full
displacement gradient ``[G_11, G_22, G_12, G_21]``), so downstream code
(``_ortho_params_from_C`` etc.) is unchanged.

Why this exists
---------------
The legacy path uses linear **TRI3 at exactly 1 element per pixel**. A mesh
convergence study (``results/mesh_artifact/SUMMARY.md``) showed that this is
**systematically over-stiff** — by ~12% for bulky cells up to **>130% for
thin / low-volume-fraction cells** — because 1-pixel-wide ligaments are a single
linear element wide and cannot bend. The error is an *element-order* artifact,
not a pixel-resolution one.

Three improvements here
-----------------------
1. **TRI6 (quadratic) elements** — ~2x more accurate per linear resolution
   (TRI6 @ N ≈ TRI3 @ 2N). Set ``ele_type="TRI3"`` to fall back to linear.
2. **Configurable mesh resolution ``N``** (default 100 = 2 elements/pixel for a
   50x50 image), decoupled from the pixel image. A loud warning fires when ``N``
   implies fewer than ~2 elements per pixel (and a hard error-level warning below
   1, which aliases the microstructure).
3. **Factor-once solve** — the 4 macro-strain load cases share one stiffness
   matrix, so we assemble + LU-factor ONCE and back-substitute 4 times, instead
   of four independent solves.

The assembly uses the same full-displacement-gradient (non-symmetrised) elastic
formulation as ``calc_fem.compute_average_stress`` so results are directly
comparable; only the element order, mesh resolution, and linear-algebra path
differ.
"""
from __future__ import annotations

import contextlib
import functools
import io

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

with contextlib.redirect_stdout(io.StringIO()):
    from jax_fem.fe import FiniteElement
    from jax_fem.generate_mesh import Mesh

    from .calc_fem import (
        E_CEMENT,
        E_VOID_RATIO,
        NU,
        RHO_CEMENT,
        build_periodic_pmat,
        make_structured_tri_mesh,
    )

# Augmented-Voigt unit load cases, gradient order [G11, G22, G12, G21].
_LOAD_CASES = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),  # e11
    np.array([[0.0, 0.0], [0.0, 1.0]]),  # e22
    np.array([[0.0, 1.0], [0.0, 0.0]]),  # e12
    np.array([[0.0, 0.0], [1.0, 0.0]]),  # e21
]
# eps as a 4-vector [G11, G22, G12, G21] per load case.
_EPS_VECS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)


# ---------------------------------------------------------------------------
# Structured quadratic-triangle mesh (meshio triangle6 ordering)
# ---------------------------------------------------------------------------

def make_structured_tri6_mesh(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Structured 6-node-triangle mesh of [0,1]^2: same triangulation as
    ``make_structured_tri_mesh`` plus edge-midpoint nodes on a (2N+1)x(2N+1)
    grid. Cells are emitted in meshio ``triangle6`` order
    ``[c0, c1, c2, m01, m12, m20]`` (jax_fem reorders basix to this). Because the
    node grid is (2N+1)^2, the periodic projection is ``build_periodic_pmat(2N)``.
    """
    M = 2 * N + 1
    xs = np.linspace(0.0, 1.0, M)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # node(ix,iy)=iy*M+ix

    def nid(ix, iy):
        return iy * M + ix

    cells = []
    for py in range(N):
        for px in range(N):
            bx, by = 2 * px, 2 * py
            BL, BR = nid(bx, by), nid(bx + 2, by)
            TL, TR = nid(bx, by + 2), nid(bx + 2, by + 2)
            cells.append([BL, BR, TR,
                          nid(bx + 1, by), nid(bx + 2, by + 1), nid(bx + 1, by + 1)])
            cells.append([BL, TR, TL,
                          nid(bx + 1, by + 1), nid(bx + 1, by + 2), nid(bx, by + 1)])
    return points, np.array(cells, dtype=np.int32)


# ---------------------------------------------------------------------------
# Mesh kinematics (cached): shape gradients, JxW, periodic matrix, B operator
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _get_kinematics(N: int, ele_type: str):
    """Build mesh + FiniteElement once per (N, ele_type) and precompute the
    element strain-displacement operator ``B`` and global DOF map.

    Returns a dict with everything the assembler/solver needs. Cached so a bulk
    run pays this once, not per cell.
    """
    if ele_type == "TRI6":
        points, cells = make_structured_tri6_mesh(N)
        p_grid = 2 * N
    elif ele_type == "TRI3":
        points, cells = make_structured_tri_mesh(N)
        p_grid = N
    else:
        raise ValueError(f"ele_type must be TRI3 or TRI6, got {ele_type!r}")

    mesh = Mesh(points, cells, ele_type=ele_type)
    with contextlib.redirect_stdout(io.StringIO()):
        fe = FiniteElement(mesh=mesh, vec=2, dim=2, ele_type=ele_type,
                           gauss_order=None, dirichlet_bc_info=None)

    cells_np = np.asarray(fe.cells)                       # (n_elem, nn)
    sg = np.asarray(fe.shape_grads)                       # (n_elem, n_quad, nn, dim)
    JxW = np.asarray(fe.JxW)                              # (n_elem, n_quad)
    n_elem, n_quad, nn, _ = sg.shape
    n_dofs = 2 * fe.num_total_nodes

    # B[e,q,a,2nn]: maps element nodal dofs -> gradient [G11,G22,G12,G21].
    B = np.zeros((n_elem, n_quad, 4, 2 * nn))
    B[:, :, 0, 0::2] = sg[:, :, :, 0]   # G11 = du1/dx1
    B[:, :, 1, 1::2] = sg[:, :, :, 1]   # G22 = du2/dx2
    B[:, :, 2, 0::2] = sg[:, :, :, 1]   # G12 = du1/dx2
    B[:, :, 3, 1::2] = sg[:, :, :, 0]   # G21 = du2/dx1

    # Global dof index per element-local dof: edof[e, 2*n+c] = 2*cells[e,n]+c.
    edof = (2 * cells_np[:, :, None] + np.arange(2)[None, None, :]).reshape(n_elem, 2 * nn)

    P_mat = build_periodic_pmat(p_grid, vec=2).tocsr()    # (n_dofs, M)

    return {
        "points": points, "cells": cells_np, "B": B, "JxW": JxW, "edof": edof,
        "n_elem": n_elem, "n_quad": n_quad, "nn": nn, "n_dofs": n_dofs,
        "P_mat": P_mat, "total_area": float(JxW.sum()),
    }


def _material_D(E: np.ndarray, nu: float) -> np.ndarray:
    """Per-element 4x4 elastic matrix in gradient order [G11,G22,G12,G21].

    sigma_ij = lam*tr(G)*delta_ij + mu*(G_ij + G_ji)  (full-gradient form, matches
    calc_fem.compute_average_stress). Returns (n_elem, 4, 4).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    n = E.shape[0]
    D = np.zeros((n, 4, 4))
    D[:, 0, 0] = lam + 2 * mu; D[:, 0, 1] = lam
    D[:, 1, 0] = lam;          D[:, 1, 1] = lam + 2 * mu
    D[:, 2, 2] = mu;           D[:, 2, 3] = mu
    D[:, 3, 2] = mu;           D[:, 3, 3] = mu
    return D


# ---------------------------------------------------------------------------
# Resolution estimate / warning
# ---------------------------------------------------------------------------

def estimate_and_warn(N: int, img_res: int, ele_type: str) -> str:
    """Return a human-readable mesh-size estimate; loudly warn if under-resolved.

    Prints element / node / DOF counts and elements-per-pixel. A coarse mesh
    (< ~2 elem/pixel) is the regime where homogenised stiffness is not converged;
    below 1 elem/pixel the microstructure is *aliased* and results are invalid.
    """
    elem_per_pixel = N / img_res
    n_elem = 2 * N * N
    n_nodes = (2 * N + 1) ** 2 if ele_type == "TRI6" else (N + 1) ** 2
    n_dofs = 2 * n_nodes
    lines = [
        f"  homogenization mesh: {ele_type}  N={N}  ({elem_per_pixel:.2f} elements/pixel "
        f"for a {img_res}x{img_res} image)",
        f"    ~{n_elem:,} triangles, ~{n_nodes:,} nodes, ~{n_dofs:,} DOFs",
    ]
    if elem_per_pixel < 1.0:
        lines += [
            "  " + "!" * 70,
            "  !!  SEVERE: < 1 element per pixel — the microstructure is ALIASED.",
            "  !!  Each element samples one pixel and ignores the rest; the effective",
            "  !!  stiffness is mesh-dependent and physically meaningless. Raise --mesh-N",
            f"  !!  to at least {img_res} (1/px), ideally {2*img_res}+ ({2}+/px).",
            "  " + "!" * 70,
        ]
    elif elem_per_pixel < 2.0:
        lines += [
            "  " + "*" * 70,
            f"  **  WARNING: only {elem_per_pixel:.2f} elements/pixel. Below ~2/px the",
            "  **  homogenised stiffness is not converged (thin/low-vf cells are",
            "  **  over-stiff by tens of %). Consider --mesh-N >= {}.".format(2 * img_res),
            "  " + "*" * 70,
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core: factor-once periodic homogenization
# ---------------------------------------------------------------------------

def compute_stiffness_hifi(
    pixel_image: np.ndarray,
    N: int = 100,
    ele_type: str = "TRI6",
    rho_solid: float = RHO_CEMENT,
    nu: float = NU,
    e_solid: float = E_CEMENT,
    e_void_ratio: float = E_VOID_RATIO,
) -> tuple[np.ndarray, float, float]:
    """Periodic FEM homogenization of one (R,R) binary pixel image.

    The mesh is ``N x N`` (decoupled from the image resolution R); material is
    sampled pointwise at element centroids. The 4 load cases share one LU
    factorization. Returns ``(C_eff 4x4 augmented Voigt, rho_eff, vf)`` — same
    convention as :func:`calc_fem.compute_effective_stiffness`.
    """
    R = pixel_image.shape[0]
    assert pixel_image.shape == (R, R), f"expected square image, got {pixel_image.shape}"
    vf = float(np.asarray(pixel_image, dtype=np.float64).mean())

    k = _get_kinematics(N, ele_type)
    B, JxW, edof = k["B"], k["JxW"], k["edof"]
    n_elem, n_dofs = k["n_elem"], k["n_dofs"]
    P = k["P_mat"]

    # Material per element: sample the image at each element centroid (pointwise,
    # exactly like calc_fem.assign_material; image row 0 = top = high y). Indexes
    # the image by ITS own resolution R, so the N-mesh and the R-pixel image are
    # decoupled.
    img = np.asarray(pixel_image)
    centroids = k["points"][k["cells"]].mean(axis=1)      # (n_elem, 2)
    ix = np.clip((centroids[:, 0] * R).astype(int), 0, R - 1)
    iy = np.clip((centroids[:, 1] * R).astype(int), 0, R - 1)
    is_solid = img[R - 1 - iy, ix].astype(bool)
    E = np.where(is_solid, e_solid, e_solid * e_void_ratio).astype(np.float64)  # (n_elem,)
    D = _material_D(E, nu)                                 # (n_elem, 4, 4)

    # Element stiffness K_e = sum_q B^T D B JxW   -> (n_elem, 2nn, 2nn)
    DB = np.einsum("eab,eqbc->eqac", D, B)                 # (n_elem, n_quad, 4, 2nn)
    K_elem = np.einsum("eqac,eqad,eq->ecd", B, DB, JxW)    # (n_elem, 2nn, 2nn)

    # Assemble global K (sparse) via vectorized COO scatter.
    nn2 = edof.shape[1]
    rows = np.repeat(edof[:, :, None], nn2, axis=2)
    cols = np.repeat(edof[:, None, :], nn2, axis=1)
    K = scipy.sparse.coo_matrix(
        (K_elem.ravel(), (rows.ravel(), cols.ravel())), shape=(n_dofs, n_dofs)
    ).tocsr()

    # Element RHS for each load case: f_e = -sum_q B^T D eps JxW  -> (n_elem, 2nn, 4)
    # D @ eps_vec for unit eps in component j is just column j of D.
    f_elem = -np.einsum("eqac,eaj,eq->ecj", B, D, JxW)     # (n_elem, 2nn, 4)
    F = np.zeros((n_dofs, 4))
    np.add.at(F, edof.reshape(-1), f_elem.reshape(-1, 4))

    # Periodic reduction + corner pin (node 0 -> reduced dofs 0,1).
    K_red = (P.T @ K @ P).tolil()
    F_red = P.T @ F
    for d in (0, 1):
        K_red.rows[d] = [d]
        K_red.data[d] = [1.0]
        F_red[d, :] = 0.0
    K_red = K_red.tocsc()

    # Factor ONCE, back-substitute for all 4 load cases.
    lu = scipy.sparse.linalg.splu(K_red)
    u_red = lu.solve(np.asarray(F_red))                    # (M, 4)
    u = P @ u_red                                          # (n_dofs, 4)

    # Average stress per load case: <sigma> = (1/A) sum D (B u + eps) JxW.
    u_elem = u[edof]                                       # (n_elem, 2nn, 4)
    G = np.einsum("eqac,ecj->eqaj", B, u_elem) + _EPS_VECS.T[None, None, :, :]
    sigma = np.einsum("eab,eqbj->eqaj", D, G)              # (n_elem, n_quad, 4, 4)
    C_eff = np.einsum("eqaj,eq->aj", sigma, JxW) / k["total_area"]  # (4 stress, 4 case)

    rho_eff = vf * rho_solid
    return C_eff, rho_eff, vf


# ---------------------------------------------------------------------------
# Standalone validation: hifi-TRI3 must match legacy calc_fem on a real cell
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check the hifi homogenizer.")
    ap.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5")
    ap.add_argument("--n-cells", type=int, default=3)
    args = ap.parse_args()

    import h5py

    with h5py.File(args.dataset, "r") as fh:
        vf = fh["vf"][:]
        idxs = np.argsort(vf)[np.linspace(0, len(vf) - 1, args.n_cells).astype(int)]
        for i in idxs:
            img = fh["cells"][int(i)].astype(np.uint8)
            ds = (float(fh["C11"][i]), float(fh["C22"][i]),
                  float(fh["C12"][i]), float(fh["C66"][i]))
            # hifi TRI3 at N=50 must reproduce the legacy dataset value (same mesh).
            C3, _, _ = compute_stiffness_hifi(img, N=50, ele_type="TRI3")
            h3 = (C3[0, 0], C3[1, 1], 0.5 * (C3[0, 1] + C3[1, 0]), 0.5 * (C3[2, 2] + C3[3, 3]))
            C6, _, _ = compute_stiffness_hifi(img, N=100, ele_type="TRI6")
            h6 = (C6[0, 0], C6[1, 1], 0.5 * (C6[0, 1] + C6[1, 0]), 0.5 * (C6[2, 2] + C6[3, 3]))
            err3 = max(abs(a - b) / abs(b) for a, b in zip(h3, ds))
            print(f"idx={int(i):>7} vf={float(vf[i]):.3f}")
            print(f"  dataset (legacy TRI3@50): C11={ds[0]:.3e} C22={ds[1]:.3e} C12={ds[2]:.3e} C66={ds[3]:.3e}")
            print(f"  hifi    TRI3@50         : C11={h3[0]:.3e} C22={h3[1]:.3e} C12={h3[2]:.3e} C66={h3[3]:.3e}  (max rel diff vs legacy {err3:.2%})")
            print(f"  hifi    TRI6@100        : C11={h6[0]:.3e} C22={h6[1]:.3e} C12={h6[2]:.3e} C66={h6[3]:.3e}")
