"""CLI entry point for 3D neural-reparam cloak optimisation.

Usage::

    python run_optimize_3d.py configs/rayleigh3d_conical_continuous.yaml
    python run_optimize_3d.py configs/rayleigh3d_conical_cells10.yaml
"""

from __future__ import annotations

import csv
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rayleigh_cloak_3d import load_config, solve_optimization_neural
from rayleigh_cloak_3d.neural import save_theta
from rayleigh_cloak_3d.vis.plot_material_slices import (
    extract_cell_params,
    plot_x_slices,
)

logging.getLogger("jax_fem").setLevel(logging.WARNING)


def _write_loss_csv(history, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for i, l in enumerate(history):
            w.writerow([i, f"{l:.10e}"])


def _plot_loss(history, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss  ($\\Vert u_\\mathrm{cloak} - u_\\mathrm{ref}\\Vert^2 / \\Vert u_\\mathrm{ref}\\Vert^2$)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python run_optimize_3d.py <config.yaml>", file=sys.stderr)
        sys.exit(2)

    cfg_path = Path(sys.argv[1])
    cfg = load_config(cfg_path)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / cfg_path.name)

    # Setup callbacks for visualization and checkpointing
    plot_every = cfg.optimization.plot_every
    best_loss = [float("inf")]
    mf_holder = [None]

    def _on_mf_ready(mf):
        mf_holder[0] = mf

    def _step_callback(step, loss_f, theta):
        mf = mf_holder[0]
        if loss_f < best_loss[0]:
            best_loss[0] = loss_f
            if mf is not None and cfg.cells.mode == "grid":
                cell_C_flat, cell_rho = extract_cell_params(mf, theta)
                np.savez(out / "best_params.npz",
                         cell_C_flat=cell_C_flat, cell_rho=cell_rho)
                print(f"    ✓ New best loss {loss_f:.4e} at step {step}, saved params")

        if plot_every > 0 and step % plot_every == 0 and mf is not None and cfg.cells.mode == "grid":
            slice_dir = out / "slices"
            slice_dir.mkdir(exist_ok=True)
            plot_x_slices(
                *extract_cell_params(mf, theta),
                mf, step, slice_dir,
            )
            print(f"    ✓ Plotted x-slices at step {step}")

    result = solve_optimization_neural(
        cfg,
        step_callback=_step_callback,
        on_material_field_ready=_on_mf_ready,
    )

    save_theta(result.theta, str(out / "theta_final.npz"))
    save_theta(result.best_theta, str(out / "theta_best.npz"))
    _write_loss_csv(result.loss_history, out / "loss_history.csv")
    _plot_loss(result.loss_history, out / "loss_curve.png")

    best = min(result.loss_history)
    print(f"\nDone. Best loss = {best:.4e}")
    print(f"  outputs → {out}")


if __name__ == "__main__":
    main()
