"""Compare stored loss with re-evaluated u_ratio for one source-frequency sweep.

For a chosen source frequency in ``surrogate_freq_sweep.h5``:
  1. Plot loss vs target f_star directly from the H5 (no solves).
  2. Re-solve each (C, rho, f_star) row and plot ``u_ratio`` vs target f_star,
     matching the metric from ``scripts/frequency_sweep.py``.

The u_ratio sweep is cached to CSV so re-runs skip the solve; use ``-f`` to
force.

Usage::

    PYTHONPATH=$(pwd) python scripts/surrogate_sweep_compare.py \\
        configs/surrogate_dataset.yaml 2.0

    # Different source frequency:
    PYTHONPATH=$(pwd) python scripts/surrogate_sweep_compare.py \\
        configs/surrogate_dataset.yaml 1.5

    # Skip the expensive u_ratio re-evaluation:
    PYTHONPATH=$(pwd) python scripts/surrogate_sweep_compare.py \\
        configs/surrogate_dataset.yaml 2.0 --no-reeval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import transmitted_displacement_ratio
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _create_geometry, solve_reference

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def _make_config_at_fstar(base_config, f_star: float):
    return base_config.model_copy(
        update={"domain": base_config.domain.model_copy(
            update={"f_star": float(f_star)}
        )}
    )


def load_sweep(h5_path: Path, source_freq: float):
    label = f"augment_from_f{source_freq:.3f}"
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        st = np.array([s.decode() if isinstance(s, bytes) else str(s)
                       for s in f["sample_type"][:]])
        mask = st == label
        if not mask.any():
            uniq = sorted({s for s in st if s.startswith("augment_from_f")})
            raise ValueError(
                f"No rows matching {label!r} in {h5_path}. "
                f"Available sources: {len(uniq)} (first: {uniq[:3]}, last: {uniq[-3:]})"
            )
        fs = f["f_star"][mask]
        loss = f["loss"][mask]
        C = f["cell_C_flat"][mask]
        rho = f["cell_rho"][mask]
    order = np.argsort(fs)
    return fs[order], loss[order], C[order], rho[order], attrs


def _check_config_matches_h5(base_config, attrs) -> None:
    n_cells_cfg = base_config.cells.n_x * base_config.cells.n_y
    if int(attrs["n_cells"]) != n_cells_cfg:
        raise ValueError(
            f"Config n_cells ({n_cells_cfg}) != H5 n_cells ({int(attrs['n_cells'])})"
        )
    if int(attrs["n_C_params"]) != base_config.cells.n_C_params:
        raise ValueError(
            f"Config n_C_params ({base_config.cells.n_C_params}) != "
            f"H5 n_C_params ({int(attrs['n_C_params'])})"
        )


def reevaluate_uratio(base_config, f_stars, C_rows, rho_rows) -> np.ndarray:
    solver_opts = {"petsc_solver": {
        "ksp_type": base_config.solver.ksp_type,
        "pc_type": base_config.solver.pc_type,
    }}
    dp_base = DerivedParams.from_config(base_config)
    geometry = _create_geometry(base_config, dp_base)

    print("=== Generating mesh (reused across all frequencies) ===")
    full_mesh = generate_mesh_full(base_config, dp_base, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

    x_left_phys = dp_base.x_off
    x_right_phys = dp_base.x_off + dp_base.W
    cloak_surface_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp_base.y_top, x_left_phys, x_right_phys,
    )
    ref_surface_idx = kept_nodes[cloak_surface_idx]
    print(f"Surface nodes beyond cloak: {len(cloak_surface_idx)}")

    cell_decomp = CellDecomposition(
        geometry, base_config.cells.n_x, base_config.cells.n_y,
    )
    C0 = C_iso(dp_base.lam, dp_base.mu)
    CellMaterial(
        geometry, C0, dp_base.rho0, cell_decomp,
        n_C_params=base_config.cells.n_C_params,
    )

    ratios = []
    for f_star, C_row, rho_row in zip(f_stars, C_rows, rho_rows):
        print(f"  f* = {f_star:.4f}", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star)
        dp = DerivedParams.from_config(config)

        ref_result = solve_reference(config, mesh=full_mesh)

        problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
        problem.set_params((jnp.array(C_row), jnp.array(rho_row)))
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u = np.asarray(sol_list[0])

        ratio = transmitted_displacement_ratio(
            u, ref_result.u, cloak_surface_idx, ref_surface_idx,
        )
        print(f"  u_ratio = {ratio:.4f}")
        ratios.append(float(ratio))
    return np.asarray(ratios)


def _save_csv(path: Path, f_stars, values, value_name: str) -> None:
    with open(path, "w") as f:
        f.write(f"f_star,{value_name}\n")
        for fs, v in zip(f_stars, values):
            f.write(f"{fs:.6f},{v:.6e}\n")


def _load_uratio_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data["f_star"]), np.atleast_1d(data["u_ratio"])


def _plot_loss(f_stars, loss, source_freq, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_stars, loss, "o-", color="C0", lw=1.5, markersize=4)
    ax.axvline(source_freq, color="gray", ls="--", lw=0.8,
               label=fr"source $f^*={source_freq:.3f}$")
    ax.set_xlabel(r"$f^*$ (target frequency)")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(fr"Dataset loss vs $f^*$ (source $f^*={source_freq:.3f}$)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot → {out_path}")


def _plot_uratio(f_stars, ratios, source_freq, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_stars, ratios, "D-", color="C0", lw=1.5, markersize=4,
            label=fr"Optimized (source $f^*={source_freq:.3f}$)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(source_freq, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$f^*$ (target frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title(
        fr"Cloaking performance vs $f^*$ (source $f^*={source_freq:.3f}$)"
    )
    ax.set_xlim(0, f_stars.max() + 0.1)
    ax.set_ylim(0, max(ratios.max() * 1.1, 1.15))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"u_ratio plot → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="Base YAML config (same as dataset generation)")
    ap.add_argument("source_freq", type=float,
                    help="Source frequency to extract, e.g. 2.0")
    ap.add_argument(
        "--h5",
        default="output/surrogate_dataset/surrogate_freq_sweep.h5",
        help="Path to cleaned sweep H5",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: <config.output_dir>/sweep_from_f<src>)")
    ap.add_argument("-f", "--force", action="store_true",
                    help="Force u_ratio re-evaluation (ignore cached CSV)")
    ap.add_argument("--no-reeval", action="store_true",
                    help="Skip u_ratio re-evaluation entirely")
    args = ap.parse_args()

    base_config = load_config(args.config)

    if args.out_dir is None:
        out_dir = Path(base_config.output_dir) / f"sweep_from_f{args.source_freq:.3f}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f_stars, loss, C_rows, rho_rows, attrs = load_sweep(
        Path(args.h5), args.source_freq,
    )
    _check_config_matches_h5(base_config, attrs)
    print(f"Loaded {len(f_stars)} rows for source f={args.source_freq:.3f}")
    print(f"  target range: [{f_stars.min():.4f}, {f_stars.max():.4f}]")

    # ── plot 1: loss vs f_star (from H5) ────────────────────────────────
    _save_csv(out_dir / "loss_vs_fstar.csv", f_stars, loss, "loss")
    _plot_loss(f_stars, loss, args.source_freq, out_dir / "loss_vs_fstar.png")

    if args.no_reeval:
        return

    # ── plot 2: u_ratio vs f_star (from re-solves, cached) ──────────────
    ratio_csv = out_dir / "uratio_vs_fstar.csv"
    if ratio_csv.exists() and not args.force:
        print(f"Reusing cached {ratio_csv} (pass -f to recompute)")
        fs_cached, ratios = _load_uratio_csv(ratio_csv)
        # sanity: cached f_stars should line up with H5's
        if len(fs_cached) != len(f_stars) or not np.allclose(fs_cached, f_stars):
            print("  ! cached CSV rows don't match H5 rows — re-running")
            ratios = reevaluate_uratio(base_config, f_stars, C_rows, rho_rows)
            _save_csv(ratio_csv, f_stars, ratios, "u_ratio")
    else:
        ratios = reevaluate_uratio(base_config, f_stars, C_rows, rho_rows)
        _save_csv(ratio_csv, f_stars, ratios, "u_ratio")

    _plot_uratio(f_stars, ratios, args.source_freq, out_dir / "uratio_vs_fstar.png")


if __name__ == "__main__":
    main()
