"""Frequency sweep: evaluate cloaking performance vs f_star.

Implements the metric from Chatzopoulos et al. (2023), Fig 2(k):
  <|u_cloak|> / <|u_ref|>
i.e. the ratio of the average total displacement magnitude on the
free surface beyond the cloaked region.  A perfect cloak -> 1.0.

Three cases are computed:
  - Obstacle: defect cutout with homogeneous material (no cloak)
  - Ideal:    defect cutout with analytical transformation-based materials
  - Optimized: defect cutout with optimized cell-based materials

The domain geometry, mesh, and PML are independent of f_star -- only omega
changes -- so they are built once and reused across all frequencies.

By default, plots from existing CSVs. Use -f to force re-running solves.

Usage::

    python scripts/frequency_sweep.py \\
        configs/triangular_optimize_neural_flat2.yaml \\
        output/triangular_optimize_neural_flat2/optimized_params.npz

    # Force re-run all:
    python scripts/frequency_sweep.py -f \\
        configs/triangular_optimize_neural_flat2.yaml \\
        output/triangular_optimize_neural_flat2/optimized_params.npz

    # Skip specific cases:
    python scripts/frequency_sweep.py --no-optimized \\
        configs/triangular_optimize_neural_flat2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import (
    find_embedded_eval_node_indices,
    make_fixed_surface_eval_points,
    transmitted_displacement_ratio,
)
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _create_geometry, solve_reference

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── CSV I/O ───────────────────────────────────────────────────────────


def _save_csv(csv_path: Path, f_stars: list[float], ratios: list[float]) -> None:
    with open(csv_path, "w") as f:
        f.write("f_star,u_ratio\n")
        for fs, r in zip(f_stars, ratios):
            f.write(f"{fs:.1f},{r:.6f}\n")
    print(f"  CSV saved to {csv_path}")


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data["f_star"], data["u_ratio"]


# ── plotting ──────────────────────────────────────────────────────────


CASE_STYLES = {
    "obstacle": {"color": "black", "ls": "--", "marker": "s", "label": "Obstacle"},
    "ideal":    {"color": "C3",    "ls": "-",  "marker": "o", "label": "Ideal Cloak"},
    "optimized": {"color": "C0",   "ls": "-",  "marker": "D", "label": "Optimized"},
}


def plot_results(case_csvs: dict[str, Path], out_dir: Path) -> None:
    """Plot all available cases on one figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    f_max = 0.0
    y_max = 0.0
    for case_name, csv_path in case_csvs.items():
        if not csv_path.exists():
            continue
        f_vals, ratio_vals = _load_csv(csv_path)
        style = CASE_STYLES[case_name]
        ax.plot(f_vals, ratio_vals,
                color=style["color"], ls=style["ls"], marker=style["marker"],
                lw=1.5, markersize=4, label=style["label"] if style["label"] != "Optimized" else "Optimized (surface boundary)")
        f_max = max(f_max, f_vals.max())
        y_max = max(y_max, ratio_vals.max())

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title("Cloaking performance vs frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_max + 0.1)
    ax.set_ylim(0, max(y_max * 1.1, 1.15))

    plot_path = out_dir / "frequency_sweep.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ── sweep runners ─────────────────────────────────────────────────────


def _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes, loss_cfg=None):
    """Surface evaluation indices for the f-specific domain height.

    Domain height (and thus ``y_top``) depends on ``f_star`` via wavelength,
    so the evaluation surface must be rebuilt per frequency — matching what
    ``resolve_loss_target`` does during training.

    When ``loss_cfg`` is supplied with ``n_eval_points > 0`` the function
    returns the indices of the *embedded* fixed-x evaluation nodes (mirroring
    the optimizer's ``top_surface`` loss); otherwise it falls back to the
    legacy node set returned by ``get_top_surface_beyond_cloak_indices``.
    """
    if loss_cfg is not None and int(loss_cfg.n_eval_points) > 0:
        eval_xs = make_fixed_surface_eval_points(
            geometry, dp, int(loss_cfg.n_eval_points),
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed),
        )
        cs_idx = find_embedded_eval_node_indices(
            cloak_mesh.points, eval_xs, dp.y_top,
        )
        return cs_idx, kept_nodes[cs_idx]

    x_left = dp.x_off
    x_right = dp.x_off + dp.W
    cs_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top, x_left, x_right,
    )
    return cs_idx, kept_nodes[cs_idx]


def _make_config_at_fstar(base_config, f_star: float):
    """Return a config copy with only f_star changed."""
    return base_config.model_copy(
        update={"domain": base_config.domain.model_copy(
            update={"f_star": float(f_star)}
        )}
    )


def run_obstacle_sweep(
    base_config, f_stars, full_mesh, cloak_mesh, kept_nodes,
    geometry, dp_base, cloak_surface_idx, ref_surface_idx,
    solver_opts, csv_path,
) -> None:
    """Sweep obstacle case: defect cutout, homogeneous material."""
    print("\n>>> Obstacle sweep")
    ratios = []
    for f_star in f_stars:
        print(f"  f* = {f_star:.1f}", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star)
        dp = DerivedParams.from_config(config)

        ref_result = solve_reference(config, mesh=full_mesh)

        # Obstacle = is_reference on submesh (homogeneous, with hole)
        obs_config = config.model_copy(update={"is_reference": True})
        problem = build_problem(cloak_mesh, obs_config, dp, geometry)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_obs = np.asarray(sol_list[0])

        cs_idx, rs_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp, kept_nodes, loss_cfg=base_config.loss,
        )
        ratio = transmitted_displacement_ratio(
            u_obs, ref_result.u, cs_idx, rs_idx,
        )
        print(f"  ratio = {ratio:.4f}")
        ratios.append(ratio)

    _save_csv(csv_path, f_stars.tolist(), ratios)


def run_ideal_sweep(
    base_config, f_stars, full_mesh, cloak_mesh, kept_nodes,
    geometry, dp_base, cloak_surface_idx, ref_surface_idx,
    solver_opts, csv_path,
) -> None:
    """Sweep ideal cloak: defect cutout, transformation-based C_eff/rho_eff."""
    print("\n>>> Ideal cloak sweep")
    ratios = []
    for f_star in f_stars:
        print(f"  f* = {f_star:.1f}", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star)
        dp = DerivedParams.from_config(config)

        ref_result = solve_reference(config, mesh=full_mesh)

        # Ideal = is_reference=False on submesh, no cell_decomp
        # -> uses analytical C_eff/rho_eff from coordinate transformation
        ideal_config = config.model_copy(update={"is_reference": False})
        problem = build_problem(cloak_mesh, ideal_config, dp, geometry)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_ideal = np.asarray(sol_list[0])

        cs_idx, rs_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp, kept_nodes, loss_cfg=base_config.loss,
        )
        ratio = transmitted_displacement_ratio(
            u_ideal, ref_result.u, cs_idx, rs_idx,
        )
        print(f"  ratio = {ratio:.4f}")
        ratios.append(ratio)

    _save_csv(csv_path, f_stars.tolist(), ratios)


def run_optimized_sweep(
    base_config, f_stars, full_mesh, cloak_mesh, kept_nodes,
    geometry, dp_base, cloak_surface_idx, ref_surface_idx,
    solver_opts, csv_path, params_path,
) -> None:
    """Sweep optimized cloak: defect cutout, optimized cell-based materials."""
    print("\n>>> Optimized cloak sweep")
    data = np.load(params_path)
    cell_C_flat = jnp.array(data["cell_C_flat"])
    cell_rho = jnp.array(data["cell_rho"])
    opt_params = (cell_C_flat, cell_rho)
    print(f"  Loaded params: cell_C_flat {cell_C_flat.shape}, cell_rho {cell_rho.shape}")

    # Cell decomposition (geometry-dependent, built once)
    cell_decomp = CellDecomposition(geometry, base_config.cells.n_x, base_config.cells.n_y)
    C0 = C_iso(dp_base.lam, dp_base.mu)
    CellMaterial(
        geometry, C0, dp_base.rho0, cell_decomp,
        n_C_params=base_config.cells.n_C_params,
    )

    ratios = []
    for f_star in f_stars:
        print(f"  f* = {f_star:.1f}", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star)
        dp = DerivedParams.from_config(config)

        ref_result = solve_reference(config, mesh=full_mesh)

        problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
        problem.set_params(opt_params)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_opt = np.asarray(sol_list[0])

        cs_idx, rs_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp, kept_nodes, loss_cfg=base_config.loss,
        )
        ratio = transmitted_displacement_ratio(
            u_opt, ref_result.u, cs_idx, rs_idx,
        )
        print(f"  ratio = {ratio:.4f}")
        ratios.append(ratio)

    _save_csv(csv_path, f_stars.tolist(), ratios)


# ── main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Frequency sweep for cloaking performance")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("params", nargs="?", default=None,
                        help="Path to optimized_params.npz (required unless --no-optimized)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force re-run solves (overwrite existing CSVs)")
    parser.add_argument("--no-obstacle", action="store_true",
                        help="Skip obstacle case")
    parser.add_argument("--no-ideal", action="store_true",
                        help="Skip ideal cloak case")
    parser.add_argument("--no-optimized", action="store_true",
                        help="Skip optimized cloak case")
    parser.add_argument("--fmin", type=float, default=0.7,
                        help="Lower end of f_star sweep (default: 0.7).")
    parser.add_argument("--fmax", type=float, default=3.3,
                        help="Upper end of f_star sweep, inclusive (default: 3.3).")
    parser.add_argument("--fstep", type=float, default=0.1,
                        help="Step size for f_star sweep (default: 0.1).")
    args = parser.parse_args()

    if not args.no_optimized and args.params is None:
        parser.error("params path is required unless --no-optimized is set")

    base_config = load_config(args.config)
    out_dir = Path(base_config.output_dir)
    out_dir.mkdir(exist_ok=True)

    csv_paths = {
        "obstacle":  out_dir / "frequency_sweep_obstacle.csv",
        "ideal":     out_dir / "frequency_sweep_ideal.csv",
        "optimized": out_dir / "frequency_sweep_optimized.csv",
    }

    # Determine which cases need solving
    cases_to_run = {}
    if not args.no_obstacle:
        cases_to_run["obstacle"] = args.force or not csv_paths["obstacle"].exists()
    if not args.no_ideal:
        cases_to_run["ideal"] = args.force or not csv_paths["ideal"].exists()
    if not args.no_optimized:
        cases_to_run["optimized"] = args.force or not csv_paths["optimized"].exists()

    need_solves = any(cases_to_run.values())

    if need_solves:
        # Build mesh/geometry once, shared across all cases
        solver_opts = {
            "petsc_solver": {
                "ksp_type": base_config.solver.ksp_type,
                "pc_type": base_config.solver.pc_type,
            }
        }
        dp_base = DerivedParams.from_config(base_config)
        geometry = _create_geometry(base_config, dp_base)

        print("=== Generating mesh (reused across all frequencies and cases) ===")
        full_mesh = generate_mesh_full(base_config, dp_base, geometry)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

        cloak_surface_idx, ref_surface_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp_base, kept_nodes,
            loss_cfg=base_config.loss,
        )
        if int(base_config.loss.n_eval_points) > 0:
            print(f"Surface evaluation nodes (fixed eval points): "
                  f"{len(cloak_surface_idx)}/{base_config.loss.n_eval_points}")
        else:
            print(f"Surface evaluation nodes beyond cloak: "
                  f"{len(cloak_surface_idx)}")

        f_stars = np.arange(args.fmin, args.fmax + 0.5 * args.fstep, args.fstep)

        shared = dict(
            base_config=base_config, f_stars=f_stars,
            full_mesh=full_mesh, cloak_mesh=cloak_mesh, kept_nodes=kept_nodes,
            geometry=geometry, dp_base=dp_base,
            cloak_surface_idx=cloak_surface_idx, ref_surface_idx=ref_surface_idx,
            solver_opts=solver_opts,
        )

        if cases_to_run.get("obstacle"):
            run_obstacle_sweep(**shared, csv_path=csv_paths["obstacle"])
        if cases_to_run.get("ideal"):
            run_ideal_sweep(**shared, csv_path=csv_paths["ideal"])
        if cases_to_run.get("optimized"):
            run_optimized_sweep(**shared, csv_path=csv_paths["optimized"],
                                params_path=args.params)
    else:
        print("All requested cases already have cached CSVs. Use -f to re-run.")

    # Plot all enabled cases (from CSVs)
    plot_csvs = {k: v for k, v in csv_paths.items() if k in cases_to_run}
    plot_results(plot_csvs, out_dir)


if __name__ == "__main__":
    main()
