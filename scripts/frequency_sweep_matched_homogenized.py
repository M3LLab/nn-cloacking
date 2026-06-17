"""Frequency sweep with **matched but still homogenised** cloak materials.

Companion to ``frequency_sweep.py`` (homogenised optimum) and
``frequency_sweep_validated.py`` (pixel-level dataset microstructures).

The validated sweep typically loses a chunk of performance versus the
homogenised optimum.  Two effects can be blamed:

1. **Matching error.**  Each optimised macro cell's continuous (λ, μ, ρ)
   triple is snapped to the nearest entry in the dataset; the matched
   (λ, μ, ρ) differs from the optimum.
2. **Homogenisation error.**  Even with a perfect (λ, μ, ρ) match, the
   pixel-level microstructure does not behave exactly like a homogeneous
   medium with those moduli.

This script isolates effect (1) by keeping the homogenised FEM (i.e. each
cloak cell carries a single (C, ρ) pair), but replacing the optimum with
the *matched* dataset (λ, μ, ρ).  Comparing this curve against
``frequency_sweep_optimized.csv`` shows how much performance is lost to
matching alone; comparing against ``frequency_sweep_validated.csv`` shows
how much extra is lost to going from homogenised to pixel-level.

Usage
-----

    python scripts/frequency_sweep_matched_homogenized.py \\
        configs/triangular_optimize_neural_flat2.yaml \\
        output/cell20_cement_init/optimized_params.npz \\
        --fmin 0.7 --fmax 3.3 --fstep 0.1
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
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


# ── grid + cloak-mask helpers (mirror frequency_sweep_validated.py) ──


def _resolve_grid(config_path: Path, n_cells: int) -> tuple[int, int]:
    import yaml
    if config_path.exists():
        cfg = yaml.safe_load(open(config_path)) or {}
        cells_cfg = cfg.get("cells", {}) or {}
        if "n_x" in cells_cfg and "n_y" in cells_cfg:
            nx, ny = int(cells_cfg["n_x"]), int(cells_cfg["n_y"])
            if nx * ny != n_cells:
                raise ValueError(f"n_x*n_y={nx*ny} != n_cells={n_cells}")
            return nx, ny
    for nx in range(int(np.sqrt(n_cells)), 0, -1):
        if n_cells % nx == 0:
            return nx, n_cells // nx
    raise ValueError(f"can't factor n_cells={n_cells}")


def _build_cloak_mask(config_path: Path, n_x: int, n_y: int):
    cfg = load_config(config_path)
    dp = DerivedParams.from_config(cfg)

    if cfg.geometry_type == "triangular":
        x_c, y_top = dp.x_c, dp.y_top
        a, b, c = dp.a, dp.b, dp.c
        x_min, x_max = x_c - c, x_c + c
        y_min, y_max = y_top - b, y_top
    elif cfg.geometry_type == "circular":
        x_c, y_c = dp.x_c, dp.y_c
        ri, rc = dp.ri, dp.rc
        x_min, x_max = x_c - rc, x_c + rc
        y_min, y_max = y_c - rc, y_c + rc
    else:
        raise ValueError(f"unsupported geometry_type={cfg.geometry_type!r}")

    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y
    cx = x_min + (np.arange(n_x) + 0.5) * cell_dx
    cy = y_min + (np.arange(n_y) + 0.5) * cell_dy
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    if cfg.geometry_type == "triangular":
        depth = y_top - centers[:, 1]
        r = np.abs(centers[:, 0] - x_c) / c
        d1 = a * (1.0 - r)
        d2 = b * (1.0 - r)
        cloak_mask = (r <= 1.0) & (depth >= d1) & (depth <= d2)
    else:
        rad = np.sqrt((centers[:, 0] - x_c) ** 2 + (centers[:, 1] - y_c) ** 2)
        cloak_mask = (rad >= ri) & (rad <= rc)
    return cloak_mask


# ── matching ────────────────────────────────────────────────────────


def build_matched_params(
    optimized_params_npz: Path,
    dataset_h5: Path,
    config_path: Path,
    rho_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match each cloak macro cell to the nearest dataset entry by
    standardised (λ, μ, ρ) and return *homogenised* per-cell parameters.

    Returns
    -------
    cell_C_flat_matched : (n_cells, 2) float — [λ, μ] per cell.  Cloak cells
        carry the matched dataset values; non-cloak cells keep their original
        optimised values (they are inert in the homogenised FEM, see
        ``CellDecomposition.expand_to_quadpoints``).
    cell_rho_matched    : (n_cells,) float — ρ per cell.  Same convention.
    diag                : misc stats for logging.
    """
    npz = np.load(optimized_params_npz)
    cell_C_flat = np.asarray(npz["cell_C_flat"])
    cell_rho = np.asarray(npz["cell_rho"])
    n_cells, n_C = cell_C_flat.shape
    if n_C != 2:
        raise SystemExit(f"this script handles n_C_params=2; got {n_C}")
    lam_q = cell_C_flat[:, 0]
    mu_q = cell_C_flat[:, 1]

    n_x, n_y = _resolve_grid(config_path, n_cells)
    cloak_mask = _build_cloak_mask(config_path, n_x, n_y)
    cloak_indices = np.where(cloak_mask)[0]

    # Standardisation must match the optimisation-time GMM standardisation:
    # use the dataset's own mean/std (no rho-weighting in the GMM, but kept
    # configurable here to mirror ``frequency_sweep_validated.py``).
    with h5py.File(dataset_h5, "r") as f:
        lam_ds = f["lambda_"][:]
        mu_ds = f["mu"][:]
        rho_ds = f["rho"][:]

    X_ds = np.column_stack([lam_ds, mu_ds, rho_ds]).astype(np.float64)
    mean = X_ds.mean(axis=0)
    std = X_ds.std(axis=0)
    Xs_ds = (X_ds - mean) / std
    Xs_ds[:, 2] *= rho_weight

    Xq_raw = np.column_stack([lam_q, mu_q, cell_rho])[cloak_indices]
    Xs_q = (Xq_raw - mean) / std
    Xs_q[:, 2] *= rho_weight

    # Brute-force NN.
    a2 = np.sum(Xs_q ** 2, axis=1, keepdims=True)
    b2 = np.sum(Xs_ds ** 2, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (Xs_q @ Xs_ds.T)
    cloak_match_idx = np.argmin(d2, axis=1)
    cloak_match_d = np.sqrt(np.maximum(
        d2[np.arange(cloak_indices.size), cloak_match_idx], 0.0,
    ))

    matched_lam = lam_ds[cloak_match_idx]
    matched_mu = mu_ds[cloak_match_idx]
    matched_rho = rho_ds[cloak_match_idx]

    # Replace cloak entries; leave non-cloak entries as-is (they don't enter
    # the FEM via the cell decomposition — non-cloak quadpoints map to the
    # background C0/ρ0 sentinel).
    cell_C_flat_matched = cell_C_flat.copy()
    cell_rho_matched = cell_rho.copy()
    cell_C_flat_matched[cloak_indices, 0] = matched_lam
    cell_C_flat_matched[cloak_indices, 1] = matched_mu
    cell_rho_matched[cloak_indices] = matched_rho

    # Per-cell parameter shifts in physical units (cloak cells only).
    dlam = matched_lam - lam_q[cloak_indices]
    dmu = matched_mu - mu_q[cloak_indices]
    drho = matched_rho - cell_rho[cloak_indices]

    diag = {
        "n_cloak": int(cloak_indices.size),
        "n_cells": n_cells,
        "match_d_median": float(np.median(cloak_match_d)),
        "match_d_mean": float(cloak_match_d.mean()),
        "match_d_max": float(cloak_match_d.max()),
        "n_unique_dataset_entries": int(np.unique(cloak_match_idx).size),
        "abs_dlam_mean_rel": float(np.mean(np.abs(dlam) / np.maximum(np.abs(lam_q[cloak_indices]), 1e-12))),
        "abs_dmu_mean_rel":  float(np.mean(np.abs(dmu)  / np.maximum(np.abs(mu_q[cloak_indices]),  1e-12))),
        "abs_drho_mean_rel": float(np.mean(np.abs(drho) / np.maximum(np.abs(cell_rho[cloak_indices]), 1e-12))),
    }
    return cell_C_flat_matched, cell_rho_matched, diag


# ── frequency sweep helpers (mirror frequency_sweep.py) ─────────────


def _save_csv(csv_path: Path, f_stars, ratios) -> None:
    with open(csv_path, "w") as f:
        f.write("f_star,u_ratio\n")
        for fs, r in zip(f_stars, ratios):
            f.write(f"{fs:.4f},{r:.6f}\n")
    print(f"  CSV → {csv_path}")


def _load_csv(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data["f_star"], data["u_ratio"]


def _make_config_at_fstar(base_config, f_star: float, refinement_factor: int | None = None):
    updates = {"domain": base_config.domain.model_copy(update={"f_star": float(f_star)})}
    if refinement_factor is not None:
        updates["mesh"] = base_config.mesh.model_copy(
            update={"refinement_factor": int(refinement_factor)}
        )
    return base_config.model_copy(update=updates)


def _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes, loss_cfg=None):
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


def run_matched_homogenized_sweep(
    base_config,
    f_stars,
    cell_C_flat_matched: np.ndarray,
    cell_rho_matched: np.ndarray,
    csv_path: Path,
    solver_opts: dict,
    refinement_factor: int | None = None,
) -> None:
    """Standard homogenised FEM with cloak cells set to matched dataset (λ, μ, ρ)."""
    rf_str = "" if refinement_factor is None else f", refinement={refinement_factor}"
    print(f"\n>>> Matched-homogenised sweep ({len(f_stars)} freqs{rf_str})")
    opt_params = (jnp.asarray(cell_C_flat_matched), jnp.asarray(cell_rho_matched))

    ratios = []
    for f_star in f_stars:
        t0 = time.time()
        print(f"  f* = {f_star:.2f} ", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star, refinement_factor)
        dp = DerivedParams.from_config(config)
        geometry = _create_geometry(config, dp)

        # Mesh per-frequency: domain dimensions depend on f_star (wavelength).
        full_mesh = generate_mesh_full(config, dp, geometry)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
        print(f"[mesh nodes={len(cloak_mesh.points)}, cells={cloak_mesh.cells.shape[0]}] ", end="", flush=True)

        cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
        C0 = C_iso(dp.lam, dp.mu)
        CellMaterial(
            geometry, C0, dp.rho0, cell_decomp,
            n_C_params=config.cells.n_C_params,
        )

        ref_result = solve_reference(config, mesh=full_mesh)

        problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
        problem.set_params(opt_params)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u = np.asarray(sol_list[0])

        cs_idx, rs_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp, kept_nodes, loss_cfg=base_config.loss,
        )
        ratio = transmitted_displacement_ratio(u, ref_result.u, cs_idx, rs_idx)
        print(f"ratio={ratio:.4f}  ({time.time()-t0:.1f}s)")
        ratios.append(ratio)

    _save_csv(csv_path, f_stars.tolist(), ratios)


# ── plotting ────────────────────────────────────────────────────────


_CASE_STYLES = {
    "obstacle":           {"color": "black", "ls": "--", "marker": "s", "label": "Obstacle"},
    "ideal":              {"color": "C3",    "ls": "-",  "marker": "o", "label": "Ideal Cloak"},
    "optimized":          {"color": "C0",    "ls": "-",  "marker": "D", "label": "Optimised (homogenised)"},
    "matched_homogenized":{"color": "C1",    "ls": "-",  "marker": "v", "label": "Matched (homogenised)"},
    "validated":          {"color": "C2",    "ls": "-",  "marker": "^", "label": "Validated (pixel-level)"},
}


def plot_results(case_csvs: dict[str, Path], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    f_max = 0.0
    y_max = 0.0
    any_plotted = False
    for case, csv_path in case_csvs.items():
        if not csv_path.exists():
            continue
        any_plotted = True
        f_vals, ratios = _load_csv(csv_path)
        s = _CASE_STYLES[case]
        ax.plot(f_vals, ratios, color=s["color"], ls=s["ls"], marker=s["marker"],
                lw=1.5, markersize=4, label=s["label"])
        f_max = max(f_max, f_vals.max())
        y_max = max(y_max, ratios.max())

    if not any_plotted:
        plt.close(fig)
        return

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title("Cloaking performance vs frequency — matching-only error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_max + 0.1)
    ax.set_ylim(0, max(y_max * 1.1, 1.15))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out_path}")


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to YAML config file")
    p.add_argument("params", help="Path to optimized_params.npz")
    p.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5",
                   help="Stiffness HDF5 (cells + lambda_/mu/rho).")
    p.add_argument("--fmin", type=float, default=0.7)
    p.add_argument("--fmax", type=float, default=3.3)
    p.add_argument("--fstep", type=float, default=0.1)
    p.add_argument("--refinement-factor", type=int, default=None,
                   help="Override mesh.refinement_factor for this sweep "
                        "(default: use the value in the config).")
    p.add_argument("--rho-weight", type=float, default=1.0,
                   help="Weight on standardised ρ in the matching distance "
                        "(must mirror frequency_sweep_validated.py for an "
                        "apples-to-apples comparison).")
    p.add_argument("-f", "--force", action="store_true",
                   help="Re-run solves even if frequency_sweep_matched_homogenized.csv exists.")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: <params dir>).")
    args = p.parse_args()

    base_config = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    csv_paths = {
        "obstacle":            out_dir / "frequency_sweep_obstacle.csv",
        "ideal":               out_dir / "frequency_sweep_ideal.csv",
        "optimized":           out_dir / "frequency_sweep_optimized.csv",
        "matched_homogenized": out_dir / "frequency_sweep_matched_homogenized.csv",
        "validated":           out_dir / "frequency_sweep_validated.csv",
    }

    # ── matching ────────────────────────────────────────────────────
    print("=== Matching cloak cells (homogenised) ===")
    cell_C_flat_m, cell_rho_m, diag = build_matched_params(
        Path(args.params), Path(args.dataset), Path(args.config),
        rho_weight=args.rho_weight,
    )
    print(
        f"cloak cells: {diag['n_cloak']}/{diag['n_cells']}  "
        f"unique dataset entries used: {diag['n_unique_dataset_entries']}\n"
        f"match-distance (std-L2): median={diag['match_d_median']:.3f}, "
        f"mean={diag['match_d_mean']:.3f}, max={diag['match_d_max']:.3f}\n"
        f"mean relative shift: |Δλ|/λ={diag['abs_dlam_mean_rel']:.3f}, "
        f"|Δμ|/μ={diag['abs_dmu_mean_rel']:.3f}, "
        f"|Δρ|/ρ={diag['abs_drho_mean_rel']:.3f}"
    )

    # ── frequency sweep ─────────────────────────────────────────────
    if csv_paths["matched_homogenized"].exists() and not args.force:
        print(f"matched-homogenised CSV exists at {csv_paths['matched_homogenized']}; "
              f"skipping solves (use -f to overwrite).")
    else:
        solver_opts = {
            "petsc_solver": {
                "ksp_type": base_config.solver.ksp_type,
                "pc_type": base_config.solver.pc_type,
            }
        }
        f_stars = np.arange(args.fmin, args.fmax + 0.5 * args.fstep, args.fstep)

        run_matched_homogenized_sweep(
            base_config=base_config,
            f_stars=f_stars,
            cell_C_flat_matched=cell_C_flat_m,
            cell_rho_matched=cell_rho_m,
            csv_path=csv_paths["matched_homogenized"],
            solver_opts=solver_opts,
            refinement_factor=args.refinement_factor,
        )

    plot_results(csv_paths, out_dir / "frequency_sweep_matched_homogenized.png")


if __name__ == "__main__":
    main()
