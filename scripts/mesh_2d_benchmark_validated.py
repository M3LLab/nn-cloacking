"""2-D mesh-resolution benchmark of the pixel-level validation.

Sweeps independent ``refinement_factor_cloak`` × ``refinement_factor_outside``
factors at a single frequency. Every cell of the resulting matrix is one
pixel-level FEM solve on a different mesh:

    inside the cloak  : finer (driven by --cloak list)
    outside the cloak : coarser (driven by --outside list, values ≤ 1.0)
    free surface      : kept tied to the cloak refinement (DistMax ~ λ*),
                        which matches the legacy behaviour for that band.

Each cell records (cells, nodes, ratio, wall_s, peak_rss_gb). Results print
as a table per metric and dump to a CSV.

Usage
-----

    PYTHONPATH=/home/m3l/workspace/nn-cloaking \\
    python scripts/mesh_2d_benchmark_validated.py \\
        configs/multifreq_small.yaml \\
        output/multifreq_small/optimized_params.npz \\
        --f-star 2.0 \\
        --cloak 5,10,15,25 \\
        --outside 1.0,0.5,0.25
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

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "frequency_sweep_validated", _HERE / "frequency_sweep_validated.py"
)
fsv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsv)

import jax_fem.solver  # noqa: E402
from rayleigh_cloak import load_config  # noqa: E402
from rayleigh_cloak.config import DerivedParams  # noqa: E402
from rayleigh_cloak.loss import transmitted_displacement_ratio  # noqa: E402
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full  # noqa: E402
from rayleigh_cloak.solver import _create_geometry, solve_reference  # noqa: E402


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _make_config(base_config, f_star: float, rf_cloak: float, rf_outside: float):
    """Override f_star and the two new mesh refinement knobs."""
    return base_config.model_copy(update={
        "domain": base_config.domain.model_copy(update={"f_star": float(f_star)}),
        "mesh": base_config.mesh.model_copy(update={
            "refinement_factor_cloak":   float(rf_cloak),
            "refinement_factor_outside": float(rf_outside),
            # leave refinement_factor_surface unset → tracks rf_cloak
        }),
    })


def _format_grid(metric_name: str, cloaks, outsides, grid: dict[tuple, str]) -> str:
    col_w = max(8, max(len(str(g)) for g in grid.values()) + 1)
    head = f"{metric_name:>16}  " + "  ".join(f"out={o:>5}".rjust(col_w) for o in outsides)
    sep  = "-" * len(head)
    lines = [head, sep]
    for c in cloaks:
        row = f"clk={c:<5}".rjust(16) + "  " + "  ".join(
            f"{grid[(c, o)]:>{col_w}}" for o in outsides
        )
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config")
    p.add_argument("params")
    p.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5")
    p.add_argument("--f-star", type=float, default=2.0)
    p.add_argument("--cloak", default="5,10,15,25",
                   help="Comma-separated refinement_factor_cloak values (≥1).")
    p.add_argument("--outside", default="1.0,0.5,0.25",
                   help="Comma-separated refinement_factor_outside values "
                        "(≤1 → coarser than h_elem).")
    p.add_argument("--void-ratio", type=float, default=1e-6)
    p.add_argument("--rho-weight", type=float, default=1.0)
    p.add_argument("-o", "--output-dir", default=None)
    args = p.parse_args()

    cloaks   = [float(x.strip()) for x in args.cloak.split(",")    if x.strip()]
    outsides = [float(x.strip()) for x in args.outside.split(",") if x.strip()]
    base_config = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / f"mesh_2d_benchmark_validated_f{args.f_star:.2f}.csv"

    print("=== Matching cloak cells & assembling canvas ===")
    canvas, (n_x, n_y), (H_pix, W_pix), cloak_bbox, _matched_C, _matched_rho, diag = fsv.build_canvas(
        Path(args.params), Path(args.dataset), Path(args.config),
        rho_weight=args.rho_weight,
    )
    # Solid phase of the tiled microstructure (cement, from the dataset's own
    # provenance attrs) — NOT the soil background of the macro simulation.
    solid = fsv.read_solid_phase(Path(args.dataset))
    print(f"solid phase (microstructure): {solid}")
    print(
        f"canvas {canvas.shape}  cloak cells {diag['n_cloak']}/{diag['n_cells']}  "
        f"unique entries {diag['n_unique_dataset_entries']}\n"
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
    print(f"sweeping {len(cloaks)}×{len(outsides)} = {len(cloaks)*len(outsides)} cells")
    for c in cloaks:
        for o in outsides:
            cfg = _make_config(base_config, args.f_star, c, o)
            dp = DerivedParams.from_config(cfg)
            geo = _create_geometry(cfg, dp)

            t0 = time.time()
            try:
                full_mesh = generate_mesh_full(cfg, dp, geo)
                cloak_mesh, kept_nodes = extract_submesh(full_mesh, geo)
                n_nodes = len(cloak_mesh.points)
                n_cells = int(cloak_mesh.cells.shape[0])
                print(
                    f"  rf_cloak={c:>5}  rf_out={o:>5}  "
                    f"nodes={n_nodes:>7}  cells={n_cells:>8}  ...",
                    end="", flush=True,
                )

                ref_result = solve_reference(cfg, mesh=full_mesh)
                problem = fsv.build_pixel_problem(
                    cloak_mesh, cfg, dp, geo,
                    canvas=canvas, cloak_bbox=cloak_bbox, void_ratio=args.void_ratio,
                    solid=solid,
                )
                sol_list = jax_fem.solver.solver(problem, solver_options=solver_opts)
                u_val = np.asarray(sol_list[0])
                cs_idx, rs_idx = fsv._surface_indices_at_f(cloak_mesh, geo, dp, kept_nodes)
                ratio = float(transmitted_displacement_ratio(u_val, ref_result.u, cs_idx, rs_idx))
                wall = time.time() - t0
                rss = _peak_rss_gb()
                print(f"ratio={ratio:.4f}  wall={wall:.1f}s  rss={rss:.2f} GB")
                rows.append({
                    "rf_cloak": c, "rf_outside": o,
                    "nodes": n_nodes, "cells": n_cells,
                    "ratio": ratio, "wall_s": wall, "peak_rss_gb": rss,
                    "status": "ok",
                })
            except Exception as exc:                                # noqa: BLE001
                wall = time.time() - t0
                rss = _peak_rss_gb()
                print(f"\n  FAILED: {type(exc).__name__}: {exc}")
                rows.append({
                    "rf_cloak": c, "rf_outside": o,
                    "nodes": -1, "cells": -1,
                    "ratio": float("nan"), "wall_s": wall, "peak_rss_gb": rss,
                    "status": f"fail:{type(exc).__name__}",
                })

    # ── CSV ─────────────────────────────────────────────────────────
    with open(csv_path, "w") as fh:
        fh.write("rf_cloak,rf_outside,nodes,cells,ratio,wall_s,peak_rss_gb,status\n")
        for r in rows:
            fh.write(
                f"{r['rf_cloak']},{r['rf_outside']},{r['nodes']},{r['cells']},"
                f"{r['ratio']:.6f},{r['wall_s']:.1f},{r['peak_rss_gb']:.2f},"
                f"{r['status']}\n"
            )
    print(f"\nCSV → {csv_path}")

    # ── tables ──────────────────────────────────────────────────────
    by_pair = {(r["rf_cloak"], r["rf_outside"]): r for r in rows}
    print(f"\n=== {Path(args.params).parent.name}  f*={args.f_star:.2f} ===")
    print(_format_grid(
        "u_ratio",
        cloaks, outsides,
        {k: f"{v['ratio']:.4f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "cells",
        cloaks, outsides,
        {k: f"{v['cells']:>7}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "wall_s",
        cloaks, outsides,
        {k: f"{v['wall_s']:.1f}" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "rss_gb",
        cloaks, outsides,
        {k: f"{v['peak_rss_gb']:.2f}" for k, v in by_pair.items()},
    ))


if __name__ == "__main__":
    main()
