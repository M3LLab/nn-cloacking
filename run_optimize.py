"""CLI entry point for cell-based material optimisation.

Usage::

    python run_optimize.py                        # uses configs/cell_based.yaml
    python run_optimize.py configs/optimize.yaml  # custom config
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rayleigh_cloak import load_config
from rayleigh_cloak.neural_reparam import save_theta
from rayleigh_cloak.solver import (
    solve_optimization,
    solve_optimization_neural,
    solve_optimization_neural_topo,
)
import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


_FLAT_LABELS = {
    2: [r"$\lambda$", r"$\mu$"],
    4: [r"$C_{1111}$", r"$C_{2222}$", r"$C_{1212}$", r"$C_{1122}$"],
    6: [r"$C_{1111}$", r"$C_{2222}$", r"$C_{1122}$",
        r"$C_{1212}$", r"$C_{2121}$", r"$C_{1221}$"],
    10: [r"$M_{11,11}$", r"$M_{11,22}$", r"$M_{11,12}$", r"$M_{11,21}$",
         r"$M_{22,22}$", r"$M_{22,12}$", r"$M_{22,21}$",
         r"$M_{12,12}$", r"$M_{12,21}$", r"$M_{21,21}$"],
}


def plot_profiles(cell_C_flat, cell_rho, n_x, n_y, n_C_params, save_path, step=None):
    """Plot 2D heatmaps of material parameters (C components + rho)."""
    C_flat_grid = cell_C_flat.reshape(n_x, n_y, n_C_params)
    rho_grid = cell_rho.reshape(n_x, n_y)

    n_plots = n_C_params + 1
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).ravel()

    flat_labels = _FLAT_LABELS.get(n_C_params,
                                   [f"p{i}" for i in range(n_C_params)])

    for k in range(n_C_params):
        ax = axes[k]
        im = ax.pcolormesh((C_flat_grid[:, :, k] / 1e9).T, shading='auto')
        fig.colorbar(im, ax=ax, label="GPa")
        ax.set_title(flat_labels[k])
        ax.set_aspect('equal')

    ax = axes[n_C_params]
    im = ax.pcolormesh(rho_grid.T, shading='auto')
    fig.colorbar(im, ax=ax, label="kg/m³")
    ax.set_title(r"$\rho$")
    ax.set_aspect('equal')

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    title = "Optimized material fields"
    if step is not None:
        title += f" — step {step}"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_loss(result, save_path: str) -> None:
    """Plot total and component loss curves."""
    steps = np.arange(len(result.loss_history))
    has_neighbor = hasattr(result, "neighbor_history") and result.neighbor_history

    if has_neighbor:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1.semilogy(steps, result.loss_history, "k-", lw=1.5, label="total")
    ax1.semilogy(steps, result.cloak_history, "C0-", lw=1, label="cloak")
    ax1.semilogy(steps, result.l2_history, "C1-", lw=1, label="L2 reg")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if has_neighbor:
        ax2.semilogy(steps, result.l2_history, "C1-", lw=1, label="L2 reg")
        ax2.semilogy(steps, result.neighbor_history, "C2-", lw=1, label="neighbor")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Regularisation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "configs/cell_based.yaml") -> None:
    config = load_config(config_path)

    if not config.cells.enabled:
        print("Error: cells.enabled must be true for optimisation. "
              "Use run.py for forward solves.")
        sys.exit(1)

    out = Path(config.output_dir)
    out.mkdir(exist_ok=True)

    # Save a copy of the config file to output directory
    shutil.copy2(config_path, out / "config.yaml")

    # Open CSV for incremental writing
    loss_csv = out / "loss_history.csv"
    csv_file = open(loss_csv, "w")
    method = config.optimization.method
    n_x = config.cells.n_x
    n_y = config.cells.n_y
    n_C_params = config.cells.n_C_params
    # For a single-cell (1x1) run, also log the decoded C-tensor components +
    # rho each step so the optimisation trajectory / basins can be reconstructed
    # and visualised afterwards (appended after transmission_ratio so existing
    # column positions are unchanged).
    log_cell_C = method != "neural_topo" and n_x * n_y == 1
    _c_names = (["C11", "C22", "C66", "C12"] if n_C_params == 4
                else [f"C{i}" for i in range(n_C_params)])
    if method == "neural_topo":
        csv_file.write(
            "step,total,cloak,l2_reg,neighbor,"
            "n_solid,lam_rel_err,mu_rel_err,rho_rel_err,avg_rho_void,"
            "transmission_ratio\n"
        )
    else:
        extra = ("," + ",".join(_c_names) + ",rho") if log_cell_C else ""
        csv_file.write(
            "step,total,cloak,l2_reg,neighbor,transmission_ratio" + extra + "\n")
    csv_file.flush()

    best_loss = float("inf")
    plot_every = config.optimization.plot_every
    profile_dir = out / "profiles"
    if plot_every > 0:
        profile_dir.mkdir(exist_ok=True)

    def _log_step(step, total, cloak, l2, neighbor, params, mat_metrics=None,
                  theta=None, opt_state=None, transmission_ratio=None):
        nonlocal best_loss
        tr = "nan" if transmission_ratio is None else f"{transmission_ratio:.6f}"
        if mat_metrics is not None:
            csv_file.write(
                f"{step},{total:.8e},{cloak:.8e},{l2:.8e},{neighbor:.8e},"
                f"{mat_metrics['n_solid']},{mat_metrics['lam_rel_err']:.6f},"
                f"{mat_metrics['mu_rel_err']:.6f},"
                f"{mat_metrics['rho_rel_err']:.6f},"
                f"{mat_metrics['avg_rho_void']:.6f},{tr}\n"
            )
        else:
            row = (f"{step},{total:.8e},{cloak:.8e},{l2:.8e},"
                   f"{neighbor:.8e},{tr}")
            if log_cell_C:
                cC, cR = params
                cC = np.asarray(cC).reshape(-1)[:n_C_params]
                cR = float(np.asarray(cR).reshape(-1)[0])
                row += "," + ",".join(f"{v:.8e}" for v in cC) + f",{cR:.8e}"
            csv_file.write(row + "\n")
        csv_file.flush()
        if total < best_loss:
            best_loss = total
            cell_C_flat, cell_rho = params
            np.savez(out / "optimized_params.npz",
                     cell_C_flat=np.asarray(cell_C_flat),
                     cell_rho=np.asarray(cell_rho))
            if theta is not None:
                save_theta(theta, str(out / "best_weights.npz"), opt_state=opt_state)
            tr_msg = ("" if transmission_ratio is None
                      else f", transmission_ratio {transmission_ratio:.4f}")
            print(f"    ✓ New best loss {total:.4e} at step {step}{tr_msg}, "
                  f"saved params")

        if plot_every > 0 and step % plot_every == 0:
            cell_C_flat, cell_rho = params
            plot_profiles(
                np.asarray(cell_C_flat), np.asarray(cell_rho),
                n_x, n_y, n_C_params,
                save_path=str(profile_dir / f"profiles_step_{step:04d}.png"),
                step=step,
            )

    try:
        if method == "neural_topo":
            print(f"Using topology neural reparameterization (method={method})")
            result = solve_optimization_neural_topo(config, step_callback=_log_step)
        elif method == "neural":
            print(f"Using neural reparameterization (method={method})")
            result = solve_optimization_neural(config, step_callback=_log_step)
        else:
            print(f"Using raw parameter optimization (method={method})")
            result = solve_optimization(config, step_callback=_log_step)
    finally:
        csv_file.close()

    print(f"\nOptimisation done. {len(result.loss_history)} iterations.")
    print(f"  Loss: {result.loss_history[0]:.4e} → {result.loss_history[-1]:.4e}")
    print(f"  Best loss: {best_loss:.4e}")
    print(f"  Loss log: {loss_csv}")

    # Plot loss curves
    plot_loss(result, save_path=str(out / "loss_curves.pdf"))
    print(f"  Loss plot: {out / 'loss_curves.pdf'}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/cell_based.yaml"
    main(path)
