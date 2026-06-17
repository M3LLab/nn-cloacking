"""
Generate a 2D microstructure dataset: geometry + homogenized C tensor.

Pipeline:
  1. Generate diverse 2D binary unit-cell masks using the cellular-automaton
     generator (dataset/generate_chiral.py).
  2. For each mask, compute the 4×4 effective stiffness via FEM periodic
     homogenization (dataset/stiffness/calc_fem.py).
  3. Save everything to a single .npz file.

Output NPZ keys:
  masks      (N, res, res)  float32  binary occupancy {0, 1}
  C_eff      (N, 4, 4)      float64  stiffness in augmented Voigt [e11,e22,e12,e21]
  rho_eff    (N,)           float64  effective density (kg/m³)
  vf         (N,)           float64  volume fraction

Usage:
  python scripts/generate_2d_dataset.py -N 200 --out dataset/2d_homogenized.npz
  python scripts/generate_2d_dataset.py -N 500 --res 64 --k 8 --seed 42
  python scripts/generate_2d_dataset.py --masks-dir dataset/out_chiral_ca/masks  # skip generation
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "jax-fem"))

from dataset.generate_chiral import GeneratorConfig, generate_dataset
from dataset.stiffness.calc_fem import (
    make_structured_tri_mesh,
    build_periodic_pmat,
    assign_material,
    compute_average_stress,
    HomogenizationProblem,
    E_CEMENT,
    RHO_CEMENT,
)
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver as jax_fem_solver
import numpy as onp


# ---------------------------------------------------------------------------
# Homogenization on a numpy array (no file I/O)
# ---------------------------------------------------------------------------

def homogenize(pixel_image: np.ndarray):
    """Compute 4×4 C_eff for a 2D binary unit cell given as a numpy array.

    Returns (C_eff, rho_eff) with C_eff in augmented Voigt notation
    [sigma_11, sigma_22, sigma_12, sigma_21] / [e_11, e_22, e_12, e_21].
    """
    N = pixel_image.shape[0]
    assert pixel_image.shape == (N, N)

    points, cells = make_structured_tri_mesh(N)
    mesh = Mesh(points, cells, ele_type='TRI3')
    E_field = assign_material(pixel_image, points, cells, num_quads=1)
    P_mat = build_periodic_pmat(N, vec=2)

    def corner(point):
        import jax.numpy as jnp
        return jnp.isclose(point[0], 0., atol=1e-5) & jnp.isclose(point[1], 0., atol=1e-5)

    dirichlet_bc_info = [[corner, corner], [0, 1],
                         [lambda p: 0., lambda p: 0.]]

    load_cases = [
        onp.array([[1., 0.], [0., 0.]]),
        onp.array([[0., 0.], [0., 1.]]),
        onp.array([[0., 1.], [0., 0.]]),
        onp.array([[0., 0.], [1., 0.]]),
    ]

    C_eff = onp.zeros((4, 4))
    for col, eps_macro in enumerate(load_cases):
        HomogenizationProblem._eps_macro = eps_macro
        HomogenizationProblem._E_field = E_field
        problem = HomogenizationProblem(
            mesh=mesh, vec=2, dim=2, ele_type='TRI3',
            dirichlet_bc_info=dirichlet_bc_info,
        )
        problem.P_mat = P_mat
        sol_list = jax_fem_solver(problem, solver_options={'umfpack_solver': {}})
        avg_stress = compute_average_stress(problem, sol_list[0], eps_macro, E_field)
        C_eff[0, col] = float(avg_stress[0, 0])
        C_eff[1, col] = float(avg_stress[1, 1])
        C_eff[2, col] = float(avg_stress[0, 1])
        C_eff[3, col] = float(avg_stress[1, 0])

    vf = float(pixel_image.astype(float).mean())
    rho_eff = vf * RHO_CEMENT
    return C_eff, rho_eff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-N", type=int, default=200,
                   help="Number of base geometries to generate (mirrors double this)")
    p.add_argument("--res", type=int, default=64,
                   help="Image resolution for FEM (resizes generated 128px masks)")
    p.add_argument("--k", type=int, default=8, choices=[1, 2, 4, 8, 16],
                   help="CA coarse-graining factor (higher = coarser blobs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mirror", action="store_true",
                   help="Disable mirror partners (halves dataset size)")
    p.add_argument("--out", type=str, default="dataset/2d_homogenized.npz",
                   help="Output .npz path")
    p.add_argument("--masks-dir", type=str, default=None,
                   help="Skip generation, use existing masks directory")
    p.add_argument("--skip-existing", action="store_true",
                   help="Load partial results from --out and skip already-computed samples")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: collect mask paths ----------------------------------------
    if args.masks_dir:
        masks_dir = Path(args.masks_dir)
        mask_paths = sorted(masks_dir.glob("*.npy"))
        print(f"Using existing masks in {masks_dir}: {len(mask_paths)} files")
    else:
        tmpdir = tempfile.mkdtemp(prefix="nn_cloak_masks_")
        print(f"Generating masks → {tmpdir}")
        cfg = GeneratorConfig(
            N=args.N,
            k=args.k,
            seed=args.seed,
            output_dir=tmpdir,
            include_mirror_pairs=not args.no_mirror,
        )
        generate_dataset(cfg)
        masks_dir = Path(tmpdir) / "masks"
        mask_paths = sorted(masks_dir.glob("*.npy"))
        print(f"Generated {len(mask_paths)} masks")

    if not mask_paths:
        print("No masks found — exiting.")
        sys.exit(1)

    # ---- Step 2: load partial results if resuming --------------------------
    done_indices = set()
    masks_done, C_done, rho_done, vf_done = [], [], [], []
    if args.skip_existing and out_path.exists():
        data = np.load(out_path)
        n_done = data["masks"].shape[0]
        masks_done = list(data["masks"])
        C_done = list(data["C_eff"])
        rho_done = list(data["rho_eff"])
        vf_done = list(data["vf"])
        done_indices = set(range(n_done))
        print(f"Resuming: {n_done} samples already computed")

    # ---- Step 3: homogenize ------------------------------------------------
    all_masks, all_C, all_rho, all_vf = (
        list(masks_done), list(C_done), list(rho_done), list(vf_done)
    )

    for i, mp in enumerate(mask_paths):
        if i in done_indices:
            continue

        mask = np.load(mp).astype(np.float32)  # original resolution (128×128)

        # Resize to FEM resolution if needed
        if args.res != mask.shape[0]:
            from skimage.transform import resize
            mask = (resize(mask.astype(float), (args.res, args.res),
                           anti_aliasing=True) > 0.5).astype(np.float32)

        vf = float(mask.mean())
        t0 = time.time()
        print(f"[{i+1}/{len(mask_paths)}]  {mp.name}  vf={vf:.3f}  res={mask.shape[0]}", end="  ", flush=True)

        try:
            C_eff, rho_eff = homogenize(mask)
        except Exception as e:
            print(f"FAILED ({e}) — skipping")
            continue

        dt = time.time() - t0
        print(f"C11={C_eff[0,0]:.3g}  C12={C_eff[0,1]:.3g}  C66={C_eff[2,2]:.3g}  ({dt:.1f}s)")

        all_masks.append(mask)
        all_C.append(C_eff)
        all_rho.append(rho_eff)
        all_vf.append(vf)

        # Save incrementally every 10 samples
        if len(all_masks) % 10 == 0:
            np.savez(out_path,
                     masks=np.stack(all_masks),
                     C_eff=np.stack(all_C),
                     rho_eff=np.array(all_rho),
                     vf=np.array(all_vf))
            print(f"  → checkpoint saved ({len(all_masks)} samples)")

    # ---- Final save --------------------------------------------------------
    np.savez(out_path,
             masks=np.stack(all_masks),
             C_eff=np.stack(all_C),
             rho_eff=np.array(all_rho),
             vf=np.array(all_vf))
    print(f"\nDone. {len(all_masks)} samples saved to {out_path}")


if __name__ == "__main__":
    main()
