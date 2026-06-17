"""CLI entry point for running a forward cloaking simulation.

Runs FEM, saves displacement plots (Re(ux), Re(uy)), and computes cloaking
loss measured two ways:
  1. Right physical boundary only
  2. All physical-domain nodes outside the cloak region

Supports both continuous C_eff (transformational) and cell-based (piecewise-
constant) material modes.

Usage::

    python run.py                           # continuous cloak (default)
    python run.py configs/continuous.yaml   # explicit continuous config
    python run.py configs/cell_based.yaml   # cell-based forward solve
    python run.py configs/reference.yaml    # reference (no cloak)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

from rayleigh_cloak import load_config, solve, solve_reference
from rayleigh_cloak.io import save_npz
from rayleigh_cloak.mesh import extract_submesh
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import compute_cloaking_loss, transmitted_displacement_ratio
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.solver import (
    SolutionResult, solve_cell_based, _create_geometry, _full_mesh,
)


# ── Plotting ──────────────────────────────────────────────────────────

def _plot_re_displacement(result: SolutionResult, output_dir: str,
                          title: str = "|u|") -> None:
    """Save |u|, Re(u_x), Re(u_y), |Re(u)| panels to output_dir.

    Delegates to the shared, void-aware ``plot.plot_field_panels`` (the same
    renderer the optimisation pipeline uses): passing the mesh connectivity
    preserves the cut-out cloak void as a hole instead of letting matplotlib
    Delaunay-fill the defect region, which has no mesh.
    """
    from rayleigh_cloak.plot import plot_field_panels

    os.makedirs(output_dir, exist_ok=True)
    pts = np.asarray(result.mesh.points)
    cells = np.asarray(result.mesh.cells)
    plot_field_panels(
        result.u, pts[:, 0], pts[:, 1], result.params,
        save_path=os.path.join(output_dir, "field.png"),
        title=title, cells=cells,
    )
    print(f"  Displacement plots saved to {output_dir}/ (field*.png)")


# ── Main ──────────────────────────────────────────────────────────────

def main(config_path: str = "configs/continuous.yaml") -> None:
    config = load_config(config_path)
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save a copy of the config file to output directory
    shutil.copy2(config_path, Path(output_dir) / "config.yaml")

    # --- Reference (no cloak) ---
    if config.is_reference:
        print("=== Reference simulation (no cloak) ===")
        result = solve(config)
        save_npz(result)
        _plot_re_displacement(result, output_dir)
        _print_summary(result)
        return

    # --- Cloaked simulation ---
    cell_based = config.cells.enabled

    if cell_based:
        print("=== Cell-based forward solve ===")
        cloak_result = solve_cell_based(config)
        full_mesh = cloak_result.full_mesh
        kept_nodes = cloak_result.kept_nodes
    else:
        print("=== Continuous C_eff forward solve ===")
        # Generate full mesh (shared with reference), then extract submesh.
        # _full_mesh dispatches on config.mesh.builder (legacy / uniform_tri6),
        # so this path honours the TRI6 uniform builder like the cell pipeline.
        full_mesh = _full_mesh(config, params, geometry)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
        cloak_result = solve(config, mesh=cloak_mesh)
        cloak_result.full_mesh = full_mesh
        cloak_result.kept_nodes = kept_nodes

    # --- Reference on same full mesh ---
    print("=== Solving reference on shared mesh ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    # --- Displacement plots ---
    _plot_re_displacement(cloak_result, output_dir)

    # --- Cloaking loss ---
    loss = compute_cloaking_loss(cloak_result, ref_result, geometry)

    # --- Surface transmission ratio (Chatzopoulos Fig 2k; →1 is perfect) ---
    # Same headline metric as the A_single_frequency cell runs, so a continuous
    # baseline is directly comparable. Surface nodes beyond the cloak footprint
    # on the cloak submesh map back to the shared full mesh via kept_nodes.
    case_surf = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, params.y_top, params.x_off,
        params.x_off + params.W)
    ref_surf = kept_nodes[case_surf]
    transmission_ratio = transmitted_displacement_ratio(
        cloak_result.u, ref_result.u, case_surf, ref_surf)

    # --- Report ---
    mode = "cell-based" if cell_based else "continuous"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Cloaking distortion (right boundary):  {loss.dist_right:.2f}%"
          f"  ({loss.n_right} nodes)")
    print(f"  Cloaking distortion (outside cloak):   {loss.dist_outside:.2f}%"
          f"  ({loss.n_outside} nodes)")
    print(f"  Surface transmission ratio:            {transmission_ratio:.4f}"
          f"  ({len(case_surf)} surface nodes, →1 ideal)")
    print(f"{'='*60}")

    # Persist the scalar metrics next to the field plots / npz.
    summary_path = Path(output_dir) / "summary.txt"
    summary_path.write_text(
        f"mode: {mode}\n"
        f"symmetrize_cloak: {config.symmetrize_cloak}\n"
        f"dist_right_pct: {loss.dist_right:.4f}\n"
        f"dist_outside_pct: {loss.dist_outside:.4f}\n"
        f"transmission_ratio: {transmission_ratio:.6f}\n"
        f"n_surface_nodes: {len(case_surf)}\n"
    )
    print(f"  Metrics summary saved to {summary_path}")

    _print_summary(cloak_result)

    # Save results
    save_npz(cloak_result)


def _print_summary(result: SolutionResult) -> None:
    p = result.params
    print(f"\nDone.  Domain: {p.W_total:.2f} x {p.H_total:.2f} "
          f"(physical {p.W:.2f} x {p.H:.2f})")
    print(f"  PML thickness = {p.L_pml:.3f},  xi_max = {p.xi_max},  "
          f"ramp power = {p.pml_pow}")
    print(f"  Mesh: {result.mesh.cells.shape[0]} triangles")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/continuous.yaml"
    main(path)
