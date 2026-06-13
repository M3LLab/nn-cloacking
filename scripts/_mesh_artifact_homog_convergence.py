"""Isolate the validated-path mesh artifact at the cheapest possible level.

The validated frequency sweep assigns material POINTWISE at each quadrature
point of an unstructured gmsh mesh that is NOT aligned to the 50x50 pixel
grid. At the production refinement (factor 5..25) an element spans several
pixels, so its single quadrature point samples one pixel and ignores the
rest -- aliasing. This script reproduces that mechanism with the production
periodic-homogenisation code, decoupling the FEM mesh resolution N_mesh from
the fixed 50x50 pixel image:

  * N_mesh  < 50  -> element bigger than a pixel: pointwise sampling drops
                    pixels (the validated regime, <1 element/pixel).
  * N_mesh == 50  -> exactly 1 element/pixel, pixel-aligned (the discretisation
                    the dataset C was computed on).
  * N_mesh  > 50  -> several elements/pixel: geometry faithfully resolved,
                    FEM error -> 0; the effective C converges.

If the effective stiffness only stabilises for N_mesh >= 50 and is erratic
below it, then the validated wave solve (which runs at <1 element/pixel) is
reading a different, mesh-dependent effective medium every time -- exactly the
non-convergent u_ratio seen in mesh_convergence_validated_f2.00.csv.

Usage
-----
    PYTHONPATH=. /home/m3l/miniconda3/envs/jax-fem-env/bin/python \
        scripts/_mesh_artifact_homog_convergence.py \
        --dataset output/ca_bulk_squared/stiffness.h5 \
        --n-cells 4 --mesh-res 12,18,25,37,50,75,100,150
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import h5py
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import jax.numpy as jnp
    from jax_fem.generate_mesh import Mesh
    from jax_fem.solver import solver as jax_fem_solver

    from dataset.stiffness.calc_fem import (
        HomogenizationProblem,
        assign_material,
        build_periodic_pmat,
        compute_average_stress,
        make_structured_tri_mesh,
    )

_LOAD_CASES = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),  # e11
    np.array([[0.0, 0.0], [0.0, 1.0]]),  # e22
    np.array([[0.0, 1.0], [0.0, 0.0]]),  # e12
    np.array([[0.0, 0.0], [1.0, 0.0]]),  # e21
]

# Quad points per element by type (TRI3: 1, TRI6: 3 — matches jax_fem basis.py).
_NUM_QUADS = {"TRI3": 1, "TRI6": 3}


def make_structured_tri6_mesh(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Structured quadratic (6-node) triangle mesh of [0,1]^2, same triangulation
    as ``make_structured_tri_mesh`` but with edge-midpoint nodes.

    Nodes live on a (2N+1)x(2N+1) grid; corner nodes at even indices. Cells are
    emitted in meshio ``triangle6`` order [c0, c1, c2, m01, m12, m20] (jax_fem
    reorders basix shape functions to this convention). The (2N+1)^2 node grid
    means the periodic projection matrix is ``build_periodic_pmat(2N)``.
    """
    M = 2 * N + 1
    xs = np.linspace(0.0, 1.0, M)
    ys = np.linspace(0.0, 1.0, M)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # node(ix,iy)=iy*M+ix

    def nid(ix, iy):
        return iy * M + ix

    cells = []
    for py in range(N):
        for px in range(N):
            bx, by = 2 * px, 2 * py  # fine-grid coords of the pixel's BL corner
            BL, BR = nid(bx, by), nid(bx + 2, by)
            TL, TR = nid(bx, by + 2), nid(bx + 2, by + 2)
            # lower triangle (BL, BR, TR): midpoints of BL-BR, BR-TR, TR-BL
            cells.append([BL, BR, TR,
                          nid(bx + 1, by), nid(bx + 2, by + 1), nid(bx + 1, by + 1)])
            # upper triangle (BL, TR, TL): midpoints of BL-TR, TR-TL, TL-BL
            cells.append([BL, TR, TL,
                          nid(bx + 1, by + 1), nid(bx + 1, by + 2), nid(bx, by + 1)])
    return points, np.array(cells, dtype=np.int32)


def compute_C_flat4(image: np.ndarray, n_mesh: int,
                    ele_type: str = "TRI3") -> tuple[float, float, float, float]:
    """Periodic homogenisation of `image` (50x50) on an n_mesh x n_mesh grid.

    Material is sampled POINTWISE at element centroids (assign_material indexes
    the image by its own resolution), so n_mesh < image-res reproduces the
    validated aliasing. ``ele_type`` selects linear (TRI3) or quadratic (TRI6)
    triangles on the SAME triangulation, isolating element order from geometry.
    Returns (C11, C22, C12, C66).
    """
    num_quads = _NUM_QUADS[ele_type]
    if ele_type == "TRI6":
        points, cells = make_structured_tri6_mesh(n_mesh)
        P_mat = build_periodic_pmat(2 * n_mesh, vec=2)
    else:
        points, cells = make_structured_tri_mesh(n_mesh)
        P_mat = build_periodic_pmat(n_mesh, vec=2)
    mesh = Mesh(points, cells, ele_type=ele_type)
    # assign_material indexes the image by its own resolution at element
    # centroids; mean over 6 nodes is still the centroid for TRI6.
    E_field = assign_material(image, points, cells, num_quads=num_quads)

    def corner(point):
        return jnp.isclose(point[0], 0.0, atol=1e-5) & jnp.isclose(point[1], 0.0, atol=1e-5)

    dirichlet_bc_info = [[corner, corner], [0, 1], [lambda p: 0.0, lambda p: 0.0]]

    C = np.zeros((4, 4))
    for col, eps_macro in enumerate(_LOAD_CASES):
        HomogenizationProblem._eps_macro = eps_macro
        HomogenizationProblem._E_field = E_field
        problem = HomogenizationProblem(
            mesh=mesh, vec=2, dim=2, ele_type=ele_type,
            dirichlet_bc_info=dirichlet_bc_info,
        )
        problem.P_mat = P_mat
        with contextlib.redirect_stdout(io.StringIO()):
            sol = jax_fem_solver(problem, solver_options={"umfpack_solver": {}})[0]
        avg = compute_average_stress(problem, sol, eps_macro, E_field)
        C[0, col], C[1, col] = float(avg[0, 0]), float(avg[1, 1])
        C[2, col], C[3, col] = float(avg[0, 1]), float(avg[1, 0])

    C11, C22 = C[0, 0], C[1, 1]
    C12 = 0.5 * (C[0, 1] + C[1, 0])
    C66 = 0.5 * (C[2, 2] + C[3, 3])
    return C11, C22, float(C12), float(C66)


def self_test(dataset: str = "output/ca_bulk_squared/stiffness.h5") -> None:
    """Validate the TRI6 mesh/ordering by cross-checking against TRI3 in the
    converged limit on a real (void-containing) cell.

    A *solid* cell is degenerate for this periodic BC (single corner pin makes
    the homogeneous system near-singular and Newton thrashes), so we use a real
    microstructure. A node-ordering bug in TRI6 would make it converge to a
    different value than TRI3; agreement at high resolution confirms correctness.
    """
    with h5py.File(dataset, "r") as f:
        # mid-vf cell — well-conditioned, non-trivial anisotropy
        vf = f["vf"][:]
        idx = int(np.argsort(vf)[len(vf) // 2])
        img = f["cells"][idx].astype(np.uint8)
    n_hi = 120
    t3 = np.array(compute_C_flat4(img, n_hi, ele_type="TRI3"))
    t6 = np.array(compute_C_flat4(img, n_hi, ele_type="TRI6"))
    rel = np.abs(t3 - t6) / (np.abs(t6) + 1e-30)
    print(f"self-test (cell idx={idx}, vf={float(vf[idx]):.3f}, N={n_hi}):")
    print(f"  TRI3 C=[{t3[0]:.3e} {t3[1]:.3e} {t3[2]:.3e} {t3[3]:.3e}]")
    print(f"  TRI6 C=[{t6[0]:.3e} {t6[1]:.3e} {t6[2]:.3e} {t6[3]:.3e}]")
    print(f"  TRI3-vs-TRI6 max rel diff = {rel.max():.2%}  "
          f"{'OK (consistent)' if rel.max() < 0.05 else 'CHECK ordering'}")


def pick_cells(dataset: Path, n_cells: int) -> list[dict]:
    """Pick n_cells dataset entries spread across volume fraction."""
    with h5py.File(dataset, "r") as f:
        vf = f["vf"][:]
        order = np.argsort(vf)
        picks = order[np.linspace(0, len(order) - 1, n_cells).astype(int)]
        out = []
        for idx in picks:
            out.append({
                "idx": int(idx),
                "image": f["cells"][idx].astype(np.uint8),
                "vf": float(f["vf"][idx]),
                "C11": float(f["C11"][idx]),
                "C22": float(f["C22"][idx]),
                "C12": float(f["C12"][idx]),
                "C66": float(f["C66"][idx]),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5")
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--mesh-res", default="25,50,75,100")
    ap.add_argument("--ele-types", default="TRI3,TRI6",
                    help="Comma-separated element types to compare (TRI3,TRI6).")
    ap.add_argument("--ref-res", type=int, default=150,
                    help="N_mesh of the TRI6 gold reference for relative error.")
    ap.add_argument("--img-res", type=int, default=50,
                    help="Pixel resolution of the dataset images (for elem/pixel).")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the solid-cell analytic check and exit.")
    ap.add_argument("-o", "--output-dir", default="results/mesh_artifact")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    mesh_res = [int(r) for r in args.mesh_res.split(",") if r.strip()]
    ele_types = [e.strip() for e in args.ele_types.split(",") if e.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = pick_cells(Path(args.dataset), args.n_cells)
    print(f"\nDataset: {args.dataset}   image res: {args.img_res}x{args.img_res}")
    print(f"Picked {len(cells)} cells (idx, vf): "
          + ", ".join(f"{c['idx']}(vf={c['vf']:.2f})" for c in cells))
    print(f"Element types: {ele_types}   mesh resolutions: {mesh_res}   "
          f"gold ref: TRI6 @ N={args.ref_res}  (elem/pixel = N/{args.img_res})\n")

    rows = []
    for c in cells:
        print(f"=== cell idx={c['idx']}  vf={c['vf']:.3f}  "
              f"dataset C11={c['C11']:.3e} C22={c['C22']:.3e} "
              f"C12={c['C12']:.3e} C66={c['C66']:.3e} ===")
        # Gold reference: TRI6 at the finest resolution.
        t0 = time.time()
        ref = np.array(compute_C_flat4(c["image"], args.ref_res, ele_type="TRI6"))
        print(f"  gold ref  TRI6 @ N={args.ref_res} ({args.ref_res/args.img_res:.1f} el/pix): "
              f"C11={ref[0]:.3e} C22={ref[1]:.3e} C12={ref[2]:.3e} C66={ref[3]:.3e}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        hdr = (f"  {'ele':>5} {'N_mesh':>7} {'el/pix':>7} {'C11':>11} {'C22':>11} "
               f"{'C12':>11} {'C66':>11} {'maxRelErr':>10} {'time_s':>7}")
        print(hdr)
        for et in ele_types:
            for nm in mesh_res:
                t0 = time.time()
                v = np.array(compute_C_flat4(c["image"], nm, ele_type=et))
                dt = time.time() - t0
                rel = np.abs(v - ref) / (np.abs(ref) + 1e-30)
                print(f"  {et:>5} {nm:>7d} {nm/args.img_res:>7.2f} {v[0]:>11.3e} {v[1]:>11.3e} "
                      f"{v[2]:>11.3e} {v[3]:>11.3e} {rel.max():>10.1%} {dt:>7.1f}", flush=True)
                rows.append({
                    "idx": c["idx"], "vf": c["vf"], "ele_type": et, "N_mesh": nm,
                    "elem_per_pixel": nm / args.img_res,
                    "C11": v[0], "C22": v[1], "C12": v[2], "C66": v[3],
                    "max_rel_err_vs_ref": float(rel.max()),
                })
        # Dataset value (TRI3 @ N=img_res) vs gold ref
        ds = np.array([c["C11"], c["C22"], c["C12"], c["C66"]])
        rel_ds = np.abs(ds - ref) / (np.abs(ref) + 1e-30)
        print(f"  dataset (TRI3 N={args.img_res}) vs gold ref: max|rel|={rel_ds.max():.1%}\n")

    csv_path = out_dir / "homog_mesh_convergence_tri36.csv"
    with open(csv_path, "w") as fh:
        fh.write("idx,vf,ele_type,N_mesh,elem_per_pixel,C11,C22,C12,C66,max_rel_err_vs_ref\n")
        for r in rows:
            fh.write(f"{r['idx']},{r['vf']:.4f},{r['ele_type']},{r['N_mesh']},"
                     f"{r['elem_per_pixel']:.3f},{r['C11']:.6e},{r['C22']:.6e},"
                     f"{r['C12']:.6e},{r['C66']:.6e},{r['max_rel_err_vs_ref']:.6f}\n")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
