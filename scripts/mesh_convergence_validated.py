"""Mesh-convergence study at a single frequency for the pixel-level validation.

Sweeps cloak-region ``refinement_factor`` and records validated u_ratio,
mesh size, peak RSS, and wall time. Results are printed as a table and
written to a CSV next to the run's optimised params.

Usage
-----

    PYTHONPATH=/home/m3l/workspace/nn-cloaking \\
    python scripts/mesh_convergence_validated.py \\
        configs/multifreq_small.yaml \\
        output/multifreq_small/optimized_params.npz \\
        --f-star 2.0 \\
        --refinements 3,5,8,12,18,25

The frequency is held fixed; only the cloak refinement factor varies. Each
solve reuses the same matched canvas, so the only thing changing is the
mesh fineness with which the optimised microstructure is resolved.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import resource
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Load frequency_sweep_validated.py as a module so we can reuse build_canvas,
# build_pixel_problem, _make_config_at_fstar, _surface_indices_at_f, etc.
# (scripts/ isn't a Python package, so we go through importlib.util.)
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "frequency_sweep_validated",
    _HERE / "frequency_sweep_validated.py",
)
fsv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsv)

import jax_fem.solver  # noqa: E402  (after fsv import which imports jax_fem)
from rayleigh_cloak import load_config  # noqa: E402
from rayleigh_cloak.config import DerivedParams  # noqa: E402
from rayleigh_cloak.loss import transmitted_displacement_ratio  # noqa: E402
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full  # noqa: E402
from rayleigh_cloak.solver import _create_geometry, solve_reference  # noqa: E402


def _peak_rss_gb() -> float:
    """Peak RSS of this process in GiB (Linux: ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _format_table(rows: list[dict]) -> str:
    headers = ["refinement", "nodes", "cells", "ratio", "wall_s", "peak_rss_gb"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    fmt_row = lambda r: "  ".join(f"{str(r[h]):>{widths[h]}}" for h in headers)
    sep = "  ".join("-" * widths[h] for h in headers)
    lines = [
        "  ".join(f"{h:>{widths[h]}}" for h in headers),
        sep,
    ]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config", help="Path to YAML config file")
    p.add_argument("params", help="Path to optimized_params.npz")
    p.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5",
                   help="Stiffness HDF5 used for matching.")
    p.add_argument("--f-star", type=float, default=2.0,
                   help="Single normalised frequency at which to evaluate.")
    p.add_argument("--refinements", default="3,5,8,12,18,25",
                   help="Comma-separated list of refinement_factor values.")
    p.add_argument("--void-ratio", type=float, default=1e-6)
    p.add_argument("--rho-weight", type=float, default=1.0)
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output dir for the CSV (default: <params dir>).")
    p.add_argument("--bail-on-oom", action="store_true",
                   help="If a refinement run dies (OOM, MUMPS fail, …), abort the "
                        "remaining sweep instead of carrying on with the next one.")
    args = p.parse_args()

    refinements = [int(r.strip()) for r in args.refinements.split(",") if r.strip()]
    base_config = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / f"mesh_convergence_validated_f{args.f_star:.2f}.csv"

    print(f"=== Matching cloak cells & assembling canvas ===")
    # build_canvas also returns matched_cell_C_flat / matched_cell_rho for the
    # cell-level (homogenised) path; the pixel-level convergence study ignores them.
    canvas, (n_x, n_y), (H_pix, W_pix), cloak_bbox, _matched_C, _matched_rho, diag = fsv.build_canvas(
        Path(args.params), Path(args.dataset), Path(args.config),
        rho_weight=args.rho_weight,
    )
    print(
        f"canvas: {canvas.shape}  cloak cells: {diag['n_cloak']}/{diag['n_cells']}  "
        f"unique dataset entries: {diag['n_unique_dataset_entries']}\n"
        f"match-distance (std-L2): median={diag['match_d_median']:.3f}, "
        f"max={diag['match_d_max']:.3f}\n"
    )

    solver_opts = {
        "petsc_solver": {
            "ksp_type": base_config.solver.ksp_type,
            "pc_type": base_config.solver.pc_type,
        }
    }

    rows: list[dict] = []
    for r in refinements:
        config = fsv._make_config_at_fstar(base_config, args.f_star, r)
        dp = DerivedParams.from_config(config)
        geometry = _create_geometry(config, dp)

        t0 = time.time()
        try:
            full_mesh = generate_mesh_full(config, dp, geometry)
            cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
            n_nodes = len(cloak_mesh.points)
            n_cells = int(cloak_mesh.cells.shape[0])
            print(
                f"refinement={r:3d}  nodes={n_nodes:>8d}  cells={n_cells:>9d}  "
                f"...solving",
                end="",
                flush=True,
            )

            ref_result = solve_reference(config, mesh=full_mesh)
            problem = fsv.build_pixel_problem(
                cloak_mesh, config, dp, geometry,
                canvas=canvas, cloak_bbox=cloak_bbox, void_ratio=args.void_ratio,
            )
            sol_list = jax_fem.solver.solver(problem, solver_options=solver_opts)
            u_val = np.asarray(sol_list[0])
            cs_idx, rs_idx = fsv._surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes)
            ratio = float(transmitted_displacement_ratio(u_val, ref_result.u, cs_idx, rs_idx))
            wall = time.time() - t0
            rss_gb = _peak_rss_gb()
            print(f"  ratio={ratio:.4f}  wall={wall:.1f}s  rss={rss_gb:.2f} GB")
            rows.append({
                "refinement": r,
                "nodes": n_nodes,
                "cells": n_cells,
                "ratio": f"{ratio:.6f}",
                "wall_s": f"{wall:.1f}",
                "peak_rss_gb": f"{rss_gb:.2f}",
            })
        except Exception as exc:                                  # noqa: BLE001
            wall = time.time() - t0
            rss_gb = _peak_rss_gb()
            print(f"\n  FAILED at refinement={r}: {type(exc).__name__}: {exc}  "
                  f"(wall={wall:.1f}s, rss={rss_gb:.2f} GB)")
            rows.append({
                "refinement": r,
                "nodes": "-",
                "cells": "-",
                "ratio": f"FAIL:{type(exc).__name__}",
                "wall_s": f"{wall:.1f}",
                "peak_rss_gb": f"{rss_gb:.2f}",
            })
            if args.bail_on_oom:
                print("--bail-on-oom set; stopping sweep.")
                break

    # ── write CSV ─────────────────────────────────────────────────
    with open(csv_path, "w") as fh:
        fh.write("refinement,nodes,cells,ratio,wall_s,peak_rss_gb\n")
        for r in rows:
            fh.write(f"{r['refinement']},{r['nodes']},{r['cells']},"
                     f"{r['ratio']},{r['wall_s']},{r['peak_rss_gb']}\n")
    print(f"\nCSV → {csv_path}")

    print(f"\n{Path(args.params).parent.name}  f*={args.f_star:.2f}")
    print(_format_table(rows))


if __name__ == "__main__":
    main()
