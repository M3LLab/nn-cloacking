"""Frequency sweep for the symmetrized-ideal continuous cloak.

Runs the exact forward path of ``run.py configs/A_symmetrized.yaml`` (continuous
C_eff, symmetrized; no cells, no optimization) across a range of dimensionless
frequencies ``f_star`` and records the surface transmission ratio at each one.

The transmission ratio is the same headline metric ``run.py`` prints/saves:
the Chatzopoulos Fig 2(k) measure ``<|u_cloak|> / <|u_ref|>`` on the free
surface beyond the cloak footprint (→1 is a perfect cloak).

Geometry, mesh, PML, and the (frequency-independent) transformation material are
built once; only ``omega`` changes with ``f_star``, so the shared mesh and the
surface-evaluation node indices are reused across all frequencies.

Usage::

    python scripts/sweep_A_symmetrized_frequency.py
    python scripts/sweep_A_symmetrized_frequency.py --f-min 0 --f-max 2.5 --f-step 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rayleigh_cloak import load_config, solve, solve_reference
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import transmitted_displacement_ratio
from rayleigh_cloak.mesh import extract_submesh
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.solver import _create_geometry, _full_mesh

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/A_symmetrized.yaml")
    ap.add_argument("--f-min", type=float, default=0.0)
    ap.add_argument("--f-max", type=float, default=2.5)
    ap.add_argument("--f-step", type=float, default=0.1)
    ap.add_argument("--csv", default=None,
                    help="Output CSV path (default: <output_dir>/transmission_vs_frequency.csv)")
    args = ap.parse_args()

    config = load_config(args.config)

    # f_star grid (inclusive of f_max, robust to floating-point drift).
    n = int(round((args.f_max - args.f_min) / args.f_step)) + 1
    f_stars = [round(args.f_min + i * args.f_step, 10) for i in range(n)]

    csv_path = Path(args.csv) if args.csv else \
        Path(config.output_dir) / "transmission_vs_frequency.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # --- f_star-independent geometry + meshes (built once) ---
    params0 = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params0)
    full_mesh = _full_mesh(config, params0, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

    # Surface eval nodes beyond the cloak footprint (geometry-only -> fixed).
    case_surf = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, params0.y_top,
        params0.x_off, params0.x_off + params0.W)
    ref_surf = kept_nodes[case_surf]
    print(f"Sweeping f_star over {f_stars[0]}..{f_stars[-1]} "
          f"(step {args.f_step}, {len(f_stars)} points); "
          f"{len(case_surf)} surface eval nodes")

    # Stream results to CSV so partial sweeps survive interruption.
    with open(csv_path, "w") as fcsv:
        fcsv.write("f_star,transmission_ratio\n")
        for fs in f_stars:
            config.domain.f_star = fs  # only omega changes; mesh is reused
            cloak_result = solve(config, mesh=cloak_mesh)
            ref_result = solve_reference(config, mesh=full_mesh)
            tr = transmitted_displacement_ratio(
                cloak_result.u, ref_result.u, case_surf, ref_surf)
            fcsv.write(f"{fs:.1f},{float(tr):.6f}\n")
            fcsv.flush()
            print(f"  f_star = {fs:.1f}  ->  transmission_ratio = {float(tr):.4f}")

    print(f"\nDone. CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
