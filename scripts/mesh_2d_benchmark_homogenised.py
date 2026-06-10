"""2-D mesh-convergence benchmark of the *homogenised* FEM result.

Same as ``mesh_2d_benchmark_validated.py`` but the cloak material is the
per-cell (λ, μ, ρ) coming straight out of ``optimized_params.npz`` — the
exact model the optimiser saw. No dataset matching, no pixel canvas. Only
the mesh refinement varies, so any mesh-driven non-convergence is a property
of the FEM discretisation itself, not of any post-hoc snapping.

Usage
-----

    PYTHONPATH=/home/m3l/workspace/nn-cloaking \\
    python scripts/mesh_2d_benchmark_homogenised.py \\
        configs/multifreq_small.yaml \\
        output/multifreq_small/optimized_params.npz \\
        --f-star 2.0 \\
        --cloak 5,10,15,25,35,50 \\
        --outside 1.0,0.5,0.25
"""
from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import jax.numpy as jnp                              # noqa: E402
import jax_fem.solver                                # noqa: E402

from rayleigh_cloak import load_config               # noqa: E402
from rayleigh_cloak.cells import CellDecomposition   # noqa: E402
from rayleigh_cloak.config import DerivedParams      # noqa: E402
from rayleigh_cloak.loss import (                                # noqa: E402
    displacement_magnitude,
    get_magnitude_band_indices,
    make_band_grid_eval_points,
    make_fixed_surface_eval_points,
    normalized_l2_mag_error_fixed,
    profile_error_surface_fixed,
    transmission_loss,
    transmitted_band_metrics_fixed,
    transmitted_displacement_ratio,
    transmitted_displacement_ratio_fixed,
)
from rayleigh_cloak.materials import C_iso, CellMaterial  # noqa: E402
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full  # noqa: E402
from rayleigh_cloak.neural_reparam import load_theta, make_neural_reparam  # noqa: E402
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices  # noqa: E402
from rayleigh_cloak.problem import build_problem     # noqa: E402
from rayleigh_cloak.solver import _create_geometry, solve_reference  # noqa: E402

import logging                                        # noqa: E402
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _make_config(
    base_config,
    f_star: float,
    rf_cloak: float,
    rf_outside: float,
    n_eval_points: int | None = None,
    eval_noise_sigma: float | None = None,
    eval_noise_seed: int | None = None,
    embed_macro_grid: bool | None = None,
):
    loss_updates = {}
    if n_eval_points is not None:
        loss_updates["n_eval_points"] = int(n_eval_points)
    if eval_noise_sigma is not None:
        loss_updates["eval_noise_sigma"] = float(eval_noise_sigma)
    if eval_noise_seed is not None:
        loss_updates["eval_noise_seed"] = int(eval_noise_seed)
    mesh_updates = {
        "refinement_factor_cloak":   float(rf_cloak),
        "refinement_factor_outside": float(rf_outside),
    }
    if embed_macro_grid is not None:
        mesh_updates["embed_macro_grid"] = bool(embed_macro_grid)
    update = {
        "domain": base_config.domain.model_copy(update={"f_star": float(f_star)}),
        "mesh":   base_config.mesh.model_copy(update=mesh_updates),
    }
    if loss_updates:
        update["loss"] = base_config.loss.model_copy(update=loss_updates)
    return base_config.model_copy(update=update)


def _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes):
    cs_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top, dp.x_off, dp.x_off + dp.W,
    )
    return cs_idx, kept_nodes[cs_idx]


def _depth_band_metrics(u_opt, u_ref, cloak_mesh, geometry, dp, kept_nodes,
                        depth, band_x_filter):
    """Metrics over the depth band the optimiser's ``magnitude_band_integral``
    loss saw — all cloak-mesh nodes in ``[y_top - depth, y_top]`` (downstream of
    the cloak, excluding the cloak/defect interior).

    Returns ``(ratio_depth, loss)`` where

    * ``ratio_depth = <|u_opt|> / <|u_ref|>`` over the band (the depth-band
      analogue of the surface transmitted-displacement ratio), and
    * ``loss = mean((|u_opt|/|u_ref| - 1)^2)`` — the exact training objective
      (:func:`transmission_loss`) on the same nodes.

    With ``depth == 0`` the band collapses to the downstream free surface, so
    these reduce to the surface metric. Returns ``(nan, nan)`` if the band
    selects no nodes.
    """
    band_idx = get_magnitude_band_indices(
        cloak_mesh.points, geometry, dp.y_top,
        dp.x_off, dp.x_off + dp.W,
        depth=float(depth),
        mode=str(band_x_filter),
    )
    if len(band_idx) == 0:
        return float("nan"), float("nan")
    u_ref_band = u_ref[kept_nodes[band_idx]]
    mag_case = displacement_magnitude(u_opt[band_idx])
    mag_ref = displacement_magnitude(u_ref_band)
    ratio_depth = float(np.mean(mag_case)) / (float(np.mean(mag_ref)) + 1e-30)
    loss = float(transmission_loss(u_opt, u_ref_band, band_idx))
    return ratio_depth, loss


def _mesh_resolution_stats(mesh):
    """Per-element size stats from a TRI3 mesh: ``(h_min, h_mean, h_max)`` where
    each element's size is its longest edge (conservative for wave resolution).
    """
    pts = np.asarray(mesh.points)[:, :2]
    tri = pts[np.asarray(mesh.cells)]                 # (n_ele, 3, 2)
    e = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    h_e = e.max(axis=1)
    return float(h_e.min()), float(h_e.mean()), float(h_e.max())


def _format_grid(metric_name: str, cloaks, outsides, grid: dict[tuple, str]) -> str:
    col_w = max(8, max(len(g) for g in grid.values()) + 1)
    head = f"{metric_name:>16}  " + "  ".join(f"out={o:>5}".rjust(col_w) for o in outsides)
    sep = "-" * len(head)
    lines = [head, sep]
    for c in cloaks:
        row = f"clk={c:<5}".rjust(16) + "  " + "  ".join(
            f"{grid[(c, o)]:>{col_w}}" for o in outsides
        )
        lines.append(row)
    return "\n".join(lines)


def _read_csv_rows(csv_path) -> list[dict]:
    """Read a benchmark CSV back into a list of typed row dicts."""
    import csv

    rows: list[dict] = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "rf_cloak":    float(r["rf_cloak"]),
                "rf_outside":  float(r["rf_outside"]),
                "nodes":       int(r["nodes"]),
                "cells":       int(r["cells"]),
                "ratio":       float(r["ratio"]),
                # New columns; absent in CSVs written before they were added.
                "ratio_depth": float(r.get("ratio_depth", "nan")),
                "loss":        float(r.get("loss", "nan")),
                "ratio_area":  float(r.get("ratio_area", "nan")),
                "loss_area":   float(r.get("loss_area", "nan")),
                "gap_ratio":   float(r.get("gap_ratio", "nan")),
                "gap_loss":    float(r.get("gap_loss", "nan")),
                "profile_error_surface":  float(r.get("profile_error_surface", "nan")),
                "outside_band_mag_error": float(r.get("outside_band_mag_error", "nan")),
                "h_min":       float(r.get("h_min", "nan")),
                "h_mean":      float(r.get("h_mean", "nan")),
                "h_max":       float(r.get("h_max", "nan")),
                "lambda_min":  float(r.get("lambda_min", "nan")),
                "ppw":         float(r.get("ppw", "nan")),
                "wall_s":      float(r["wall_s"]),
                "peak_rss_gb": float(r["peak_rss_gb"]),
                "status":      r["status"],
            })
    return rows


def _fmt_count(v, _pos):
    """Compact tick label for cell/DOF counts: 110k, 1.2M, etc."""
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def _conv_panel(ax, rows, outsides, colors, ykey, ylabel, title,
                xkey="cells", xlabel="cells (log)", logx=True, logy=False):
    """Draw one convergence/cost panel: ``ykey`` vs ``xkey``, one line per
    outside refinement factor (sorted by cloak refinement factor)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for o in outsides:
        sub = sorted((r for r in rows if r["rf_outside"] == o),
                     key=lambda r: r["rf_cloak"])
        xs = [r[xkey] for r in sub]
        ys = [r[ykey] for r in sub]
        ax.plot(xs, ys, "o-", color=colors[o], label=f"outside={o:g}")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    # Declutter the cell-count axis: compact k/M labels, few ticks, rotated.
    if xkey == "cells":
        if logx:
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_count))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_results(csv_path, save_path=None, title: str | None = None):
    """Plot a mesh-convergence sweep CSV into TWO figures (one line per outside
    refinement factor), so panels stay large enough to read:

    ``<stem>.png`` — ratios & losses (2×3):
        ratio vs cloak-rf · ratio vs cells · ratio_depth [mesh-dep] ·
        ratio_area [mesh-indep] · loss [mesh-dep] · loss_area [mesh-indep]
    ``<stem>_diagnostics.png`` — gaps, Tier-2 errors, resolution & cost (2×4):
        gap_ratio · gap_loss · profile_error_surface · outside_band_mag_error ·
        elements_per_wavelength (ppw) · wall time · peak RSS

    Mesh-dependent metrics and their mesh-independent counterparts sit side by
    side; the gap_* panels show the sampling artifact shrinking, ppw shows wave
    resolution, and outside_band_mag_error flags overfitting beyond the trained
    band. Only ``status == "ok"`` rows are plotted. Returns ``[path1, path2]``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in _read_csv_rows(csv_path) if r["status"] == "ok"]
    if not rows:
        raise ValueError(f"no successful (status==ok) rows in {csv_path}")

    csv_path = Path(csv_path)
    if save_path is None:
        save_path = csv_path.with_suffix(".png")
    save_path = Path(save_path)
    diag_path = save_path.with_name(f"{save_path.stem}_diagnostics{save_path.suffix}")
    if title is None:
        title = csv_path.stem

    outsides = sorted({r["rf_outside"] for r in rows})
    cmap = plt.get_cmap("viridis")
    colors = {o: cmap(i / max(len(outsides) - 1, 1)) for i, o in enumerate(outsides)}

    # ── Figure 1: ratios & losses (2×3) ──
    fig1, ax1 = plt.subplots(2, 3, figsize=(18, 10))
    B = ax1.ravel()
    _conv_panel(B[0], rows, outsides, colors, "ratio",
                "transmitted-displacement ratio", "ratio (surface) vs cloak refinement",
                xkey="rf_cloak", xlabel="cloak refinement factor", logx=False)
    _conv_panel(B[1], rows, outsides, colors, "ratio",
                "ratio (surface)", "ratio (surface) vs cells")
    _conv_panel(B[2], rows, outsides, colors, "ratio_depth",
                "ratio (band, node-mean)", "ratio_depth vs cells  [mesh-dependent]")
    _conv_panel(B[3], rows, outsides, colors, "ratio_area",
                "ratio (band, area-weighted)", "ratio_area vs cells  [mesh-independent]")
    _conv_panel(B[4], rows, outsides, colors, "loss",
                "loss (band, node-mean)", "loss vs cells  [mesh-dependent]", logy=True)
    _conv_panel(B[5], rows, outsides, colors, "loss_area",
                "loss (band, area-weighted)", "loss_area vs cells  [mesh-independent]", logy=True)
    fig1.suptitle(f"{title} — ratios & losses", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    # ── Figure 2: gaps, Tier-2 errors, resolution & cost (2×4) ──
    fig2, ax2 = plt.subplots(2, 4, figsize=(22, 10))
    C = ax2.ravel()
    _conv_panel(C[0], rows, outsides, colors, "gap_ratio",
                "|ratio_depth - ratio_area|", "gap_ratio vs cells  [sampling artifact]", logy=True)
    _conv_panel(C[1], rows, outsides, colors, "gap_loss",
                "|loss - loss_area|", "gap_loss vs cells  [sampling artifact]", logy=True)
    _conv_panel(C[2], rows, outsides, colors, "profile_error_surface",
                "normalized L2 (surface profile)", "profile_error_surface vs cells", logy=True)
    _conv_panel(C[3], rows, outsides, colors, "outside_band_mag_error",
                "normalized L2 (out-of-band)", "outside_band_mag_error vs cells  [generalization]", logy=True)
    _conv_panel(C[4], rows, outsides, colors, "ppw",
                "elements per wavelength", "ppw (lambda_min / h_max) vs cells")
    _conv_panel(C[5], rows, outsides, colors, "wall_s",
                "wall time (s)", "cost: wall time", xlabel="cells", logx=False)
    _conv_panel(C[6], rows, outsides, colors, "peak_rss_gb",
                "peak RSS (GB)", "cost: peak memory", xlabel="cells", logx=False)
    C[7].set_visible(False)
    fig2.suptitle(f"{title} — diagnostics", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(diag_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    return [save_path, diag_path]


def _load_opt_params(params_path: str, base_config):
    """Load the optimised per-cell ``(cell_C_flat, cell_rho)`` from ``params_path``.

    Two on-disk formats are supported transparently:

    * ``optimized_params.npz`` — already-decoded arrays ``cell_C_flat`` and
      ``cell_rho`` (raw or topo optimisers, and the neural optimiser's decoded
      dump). Loaded directly.
    * ``best_weights.npz`` — neural-field MLP weights (``W_i``/``b_i`` +
      ``n_layers``) from a ``method: neural`` run (e.g. the *flat4* model). The
      network is rebuilt from ``base_config.optimization.neural`` and decoded at
      the cell centres, reproducing exactly what the optimiser saw.
    """
    npz = np.load(params_path)
    keys = set(npz.files)

    if {"cell_C_flat", "cell_rho"} <= keys:
        cell_C_flat = jnp.asarray(npz["cell_C_flat"])
        cell_rho = jnp.asarray(npz["cell_rho"])
        print(f"loaded decoded params: cell_C_flat {tuple(cell_C_flat.shape)}, "
              f"cell_rho {tuple(cell_rho.shape)}")
        return cell_C_flat, cell_rho

    if "n_layers" not in keys:
        raise ValueError(
            f"{params_path} has neither (cell_C_flat, cell_rho) nor neural-field "
            f"weights (n_layers); found keys: {sorted(keys)}"
        )

    # Neural-field weights → rebuild the reparam exactly as solve_optimization_neural
    # does and decode at the cell centres. The decode depends only on geometry,
    # the cell decomposition and the initial (pushforward) params — all
    # frequency-independent — so a single decode is valid across the whole sweep.
    dp = DerivedParams.from_config(base_config)
    geometry = _create_geometry(base_config, dp)
    cell_decomp = CellDecomposition(
        geometry, base_config.cells.n_x, base_config.cells.n_y,
    )
    cell_mat = CellMaterial(
        geometry, C_iso(dp.lam, dp.mu), dp.rho0, cell_decomp,
        n_C_params=base_config.cells.n_C_params,
        symmetrize_init=base_config.cells.symmetrize_init,
        init=base_config.cells.init,
        init_path=base_config.cells.init_path,
    )
    params_init = cell_mat.get_initial_params()

    ncfg = base_config.optimization.neural
    _, reparam = make_neural_reparam(
        cell_decomp, params_init,
        hidden_size=ncfg.hidden_size,
        n_layers=ncfg.n_layers,
        n_fourier=ncfg.n_fourier,
        seed=ncfg.seed,
        output_scale=ncfg.output_scale,
        constrained=ncfg.constrained,
        kappa=ncfg.kappa,
        cap_anisotropy=ncfg.cap_anisotropy,
        anisotropy_ratio=ncfg.anisotropy_ratio,
    )
    theta, _ = load_theta(params_path)
    cell_C_flat, cell_rho = reparam.decode(theta)
    print(f"loaded neural-field weights ({int(npz['n_layers'])} layers) and "
          f"decoded: cell_C_flat {tuple(cell_C_flat.shape)}, "
          f"cell_rho {tuple(cell_rho.shape)}")
    return cell_C_flat, cell_rho


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config")
    p.add_argument("params")
    p.add_argument("--f-star", type=float, default=2.0)
    p.add_argument("--cloak", default="5,10,15,25,35,50")
    p.add_argument("--outside", default="1.0,0.5,0.25")
    p.add_argument("-o", "--output-dir", default=None)
    p.add_argument("--n-eval-points", type=int, default=None,
                   help="Override loss.n_eval_points. 0 keeps the legacy "
                        "node-based metric; >0 evaluates |u| at this many "
                        "fixed x-positions (mesh-independent).")
    p.add_argument("--eval-noise-sigma", type=float, default=None,
                   help="Override loss.eval_noise_sigma (Gaussian jitter on "
                        "the fixed x-positions, in physical units).")
    p.add_argument("--eval-noise-seed", type=int, default=None,
                   help="Override loss.eval_noise_seed.")
    p.add_argument("--embed-macro-grid", action="store_true",
                   help="Embed the (n_x-1)+(n_y-1) interior macro-grid lines as "
                        "1-D constraints in the gmsh surface so no FEM element "
                        "straddles a macro-cell boundary.")
    p.add_argument("--band-nx", type=int, default=200,
                   help="Number of x-positions in the fixed grid used by the "
                        "mesh-independent area-weighted band metric "
                        "(ratio_area/loss_area).")
    p.add_argument("--band-ny", type=int, default=25,
                   help="Number of y-positions (over the depth band) in the "
                        "fixed grid for the area-weighted band metric. Ignored "
                        "when loss.depth == 0 (collapses to the surface line).")
    p.add_argument("--outside-band-depth", type=float, default=1.0,
                   help="Bottom of the out-of-band validation strip (physical "
                        "units below the free surface). The generalization "
                        "metric outside_band_mag_error is measured on "
                        "[y_top - this, y_top - loss.depth], i.e. just below "
                        "the trained band. Must be > loss.depth and < H.")
    args = p.parse_args()

    cloaks   = [float(x.strip()) for x in args.cloak.split(",")    if x.strip()]
    outsides = [float(x.strip()) for x in args.outside.split(",") if x.strip()]
    base_config = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / f"mesh_2d_benchmark_homogenised_f{args.f_star:.2f}.csv"

    # Load optimised params (kept identical across all mesh resolutions).
    # Accepts both decoded (cell_C_flat/cell_rho) and neural-field weight dumps.
    opt_params = _load_opt_params(args.params, base_config)

    solver_opts = {
        "petsc_solver": {
            "ksp_type": base_config.solver.ksp_type,
            "pc_type": base_config.solver.pc_type,
        }
    }

    # CellDecomposition is geometry-dependent; geometry is mesh-independent
    # (only depends on the *physical* cloak description), so we can build it
    # once outside the sweep. The dp/geometry inside the loop are rebuilt
    # because f_star may change them (in fact it doesn't here; both depend
    # only on dimensionless geometry factors).
    print(f"sweeping {len(cloaks)} × {len(outsides)} = {len(cloaks)*len(outsides)} cells")

    # Write the CSV header up front and append after each sweep point so an
    # OOM mid-sweep doesn't lose all previous results.
    with open(csv_path, "w") as fh:
        fh.write("rf_cloak,rf_outside,nodes,cells,ratio,ratio_depth,loss,"
                 "ratio_area,loss_area,gap_ratio,gap_loss,"
                 "profile_error_surface,outside_band_mag_error,"
                 "h_min,h_mean,h_max,lambda_min,ppw,"
                 "wall_s,peak_rss_gb,status\n")

    rows: list[dict] = []
    for c in cloaks:
        for o in outsides:
            cfg = _make_config(
                base_config, args.f_star, c, o,
                n_eval_points=args.n_eval_points,
                eval_noise_sigma=args.eval_noise_sigma,
                eval_noise_seed=args.eval_noise_seed,
                embed_macro_grid=(True if args.embed_macro_grid else None),
            )
            dp = DerivedParams.from_config(cfg)
            geometry = _create_geometry(cfg, dp)
            cell_decomp = CellDecomposition(
                geometry, base_config.cells.n_x, base_config.cells.n_y,
            )

            # Build the fixed evaluation x-positions ONCE per (c, o) pair
            # (they only depend on geometry/dp, which are constant across the
            # sweep at fixed f_star — but rebuilding is cheap and clearer).
            eval_xs = None
            if cfg.loss.n_eval_points > 0:
                eval_xs = make_fixed_surface_eval_points(
                    geometry, dp,
                    cfg.loss.n_eval_points,
                    cfg.loss.eval_noise_sigma,
                    cfg.loss.eval_noise_seed,
                )

            # Fixed (x, y) grid for the mesh-independent area-weighted band
            # metric. Depends only on geometry/dp/depth (constant across the
            # sweep), so the metric is a stable functional of the field.
            band_xs, band_ys = make_band_grid_eval_points(
                geometry, dp,
                depth=cfg.loss.depth,
                n_x=args.band_nx, n_y=args.band_ny,
                mode=cfg.loss.band_x_filter,
            )
            # Fixed downstream surface x-grid (depth 0) for profile_error_surface.
            surf_xs, _ = make_band_grid_eval_points(
                geometry, dp, depth=0.0,
                n_x=args.band_nx, n_y=1, mode=cfg.loss.band_x_filter,
            )
            # Out-of-band validation grid: the strip [y_top - obd, y_top - depth]
            # just below the trained band, for the generalization metric. None
            # (→ nan) when the requested depth is not a valid sub-band.
            obd = float(args.outside_band_depth)
            if obd > float(cfg.loss.depth) and obd < float(dp.H):
                outb_xs, outb_ys = make_band_grid_eval_points(
                    geometry, dp, depth=obd, depth_top=float(cfg.loss.depth),
                    n_x=args.band_nx, n_y=args.band_ny, mode=cfg.loss.band_x_filter,
                )
            else:
                outb_xs = outb_ys = None
            # Shortest relevant wavelength: Rayleigh wave at the swept f_star
            # (lambda_R = 2*pi*cR/omega = lambda_star/f_star). cR < cs < cp, so
            # the Rayleigh branch is the shortest and governs surface resolution.
            lambda_min = float(2.0 * np.pi * dp.cR / dp.omega)

            t0 = time.time()
            try:
                full_mesh = generate_mesh_full(cfg, dp, geometry)
                cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
                n_nodes = len(cloak_mesh.points)
                n_cells = int(cloak_mesh.cells.shape[0])
                print(
                    f"  rf_cloak={c:>5}  rf_out={o:>5}  "
                    f"nodes={n_nodes:>7}  cells={n_cells:>8}  ...",
                    end="", flush=True,
                )

                ref_result = solve_reference(cfg, mesh=full_mesh)
                problem = build_problem(cloak_mesh, cfg, dp, geometry, cell_decomp)
                problem.set_params(opt_params)
                sol_list = jax_fem.solver.solver(problem, solver_options=solver_opts)
                u_opt = np.asarray(sol_list[0])
                if eval_xs is not None:
                    ratio = transmitted_displacement_ratio_fixed(
                        u_opt, ref_result.u, cloak_mesh, full_mesh,
                        eval_xs, dp.y_top,
                    )
                else:
                    cs_idx, rs_idx = _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes)
                    ratio = float(transmitted_displacement_ratio(u_opt, ref_result.u, cs_idx, rs_idx))
                # Depth-band metrics matching the optimiser's
                # magnitude_band_integral loss (unweighted node-mean over the
                # band — mesh-density-dependent, i.e. "what the optimiser saw").
                ratio_depth, loss = _depth_band_metrics(
                    u_opt, ref_result.u, cloak_mesh, geometry, dp, kept_nodes,
                    depth=cfg.loss.depth, band_x_filter=cfg.loss.band_x_filter,
                )
                # Mesh-independent, area-weighted counterparts on a fixed grid
                # ("what the physics says" — converges to a single limit).
                ratio_area, loss_area = transmitted_band_metrics_fixed(
                    u_opt, ref_result.u, cloak_mesh, full_mesh, band_xs, band_ys,
                )
                # Tier-1: node-vs-area sampling-artifact gaps (→ 0 when the
                # mesh-sampled and mesh-independent readings agree).
                gap_ratio = abs(ratio_depth - ratio_area)
                gap_loss = abs(loss - loss_area)
                # Tier-1: wave-resolution diagnostics on the physical (cloak) mesh.
                h_min, h_mean, h_max = _mesh_resolution_stats(cloak_mesh)
                ppw = lambda_min / h_max if h_max > 0 else float("nan")
                # Tier-2: fixed-grid surface-profile shape error and out-of-band
                # generalization error (both mesh-independent).
                profile_err = profile_error_surface_fixed(
                    u_opt, ref_result.u, cloak_mesh, full_mesh, surf_xs, dp.y_top,
                )
                if outb_xs is not None:
                    outside_band_err = normalized_l2_mag_error_fixed(
                        u_opt, ref_result.u, cloak_mesh, full_mesh, outb_xs, outb_ys,
                    )
                else:
                    outside_band_err = float("nan")
                wall = time.time() - t0
                rss = _peak_rss_gb()
                print(
                    f"ratio={ratio:.4f}  ratio_depth={ratio_depth:.4f}  "
                    f"loss={loss:.4e}  ratio_area={ratio_area:.4f}  "
                    f"loss_area={loss_area:.4e}  ppw={ppw:.1f}  "
                    f"prof_err={profile_err:.4f}  out_err={outside_band_err:.4f}  "
                    f"wall={wall:.1f}s  rss={rss:.2f} GB"
                )
                row = {
                    "rf_cloak": c, "rf_outside": o,
                    "nodes": n_nodes, "cells": n_cells,
                    "ratio": ratio, "ratio_depth": ratio_depth, "loss": loss,
                    "ratio_area": ratio_area, "loss_area": loss_area,
                    "gap_ratio": gap_ratio, "gap_loss": gap_loss,
                    "profile_error_surface": profile_err,
                    "outside_band_mag_error": outside_band_err,
                    "h_min": h_min, "h_mean": h_mean, "h_max": h_max,
                    "lambda_min": lambda_min, "ppw": ppw,
                    "wall_s": wall, "peak_rss_gb": rss,
                    "status": "ok",
                }
            except Exception as exc:                                # noqa: BLE001
                wall = time.time() - t0
                rss = _peak_rss_gb()
                print(f"\n  FAILED: {type(exc).__name__}: {exc}")
                row = {
                    "rf_cloak": c, "rf_outside": o,
                    "nodes": -1, "cells": -1,
                    "ratio": float("nan"), "ratio_depth": float("nan"),
                    "loss": float("nan"), "ratio_area": float("nan"),
                    "loss_area": float("nan"),
                    "gap_ratio": float("nan"), "gap_loss": float("nan"),
                    "profile_error_surface": float("nan"),
                    "outside_band_mag_error": float("nan"),
                    "h_min": float("nan"), "h_mean": float("nan"),
                    "h_max": float("nan"), "lambda_min": float("nan"),
                    "ppw": float("nan"),
                    "wall_s": wall, "peak_rss_gb": rss,
                    "status": f"fail:{type(exc).__name__}",
                }
            rows.append(row)
            with open(csv_path, "a") as fh:
                fh.write(
                    f"{row['rf_cloak']},{row['rf_outside']},{row['nodes']},{row['cells']},"
                    f"{row['ratio']:.6f},{row['ratio_depth']:.6f},{row['loss']:.6e},"
                    f"{row['ratio_area']:.6f},{row['loss_area']:.6e},"
                    f"{row['gap_ratio']:.6f},{row['gap_loss']:.6e},"
                    f"{row['profile_error_surface']:.6f},{row['outside_band_mag_error']:.6f},"
                    f"{row['h_min']:.6f},{row['h_mean']:.6f},{row['h_max']:.6f},"
                    f"{row['lambda_min']:.6f},{row['ppw']:.4f},"
                    f"{row['wall_s']:.1f},{row['peak_rss_gb']:.2f},"
                    f"{row['status']}\n"
                )

    print(f"\nCSV → {csv_path}")

    by_pair = {(r["rf_cloak"], r["rf_outside"]): r for r in rows}
    print(f"\n=== {Path(args.params).parent.name}  f*={args.f_star:.2f} (homogenised) ===")
    print(_format_grid(
        "u_ratio", cloaks, outsides,
        {k: f"{v['ratio']:.4f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "ratio_depth", cloaks, outsides,
        {k: f"{v['ratio_depth']:.4f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "loss", cloaks, outsides,
        {k: f"{v['loss']:.3e}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "ratio_area", cloaks, outsides,
        {k: f"{v['ratio_area']:.4f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "loss_area", cloaks, outsides,
        {k: f"{v['loss_area']:.3e}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "gap_loss", cloaks, outsides,
        {k: f"{v['gap_loss']:.3e}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "out_band_err", cloaks, outsides,
        {k: f"{v['outside_band_mag_error']:.4f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "ppw", cloaks, outsides,
        {k: f"{v['ppw']:.1f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "cells", cloaks, outsides,
        {k: f"{v['cells']:>7}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "wall_s", cloaks, outsides,
        {k: f"{v['wall_s']:.1f}" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "rss_gb", cloaks, outsides,
        {k: f"{v['peak_rss_gb']:.2f}" for k, v in by_pair.items()},
    ))

    if any(r["status"] == "ok" for r in rows):
        plot_paths = plot_results(
            csv_path,
            title=f"{Path(args.params).parent.name}  f*={args.f_star:.2f} (homogenised)",
        )
        for pp in plot_paths:
            print(f"plot → {pp}")


if __name__ == "__main__":
    main()
