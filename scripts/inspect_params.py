#!/usr/bin/env python3
"""Inspect optimized cell parameters: find negative values and plot them.

Usage::

    python scripts/inspect_params.py best_configs/optimized_params.npz best_configs/cauchy_tri_top.yaml
    python scripts/inspect_params.py best_configs/optimized_params.npz   # uses configs/cell_based.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent.parent))

from rayleigh_cloak.config import DerivedParams, SimulationConfig, load_config


def load_cell_grid(npz_path: str, config_path: str, n_x: int = 50, n_y: int = 50):
    """Load optimized params and build cell-center coordinates."""
    data = np.load(npz_path)
    cell_C_flat = data["cell_C_flat"]  # (n_cells, n_C_params)
    cell_rho = data["cell_rho"]        # (n_cells,)

    cfg = load_config(config_path)
    dp = DerivedParams.from_config(cfg)

    # Cloak bounding box (same as CellDecomposition)
    x_c = dp.x_c
    y_top = dp.y_top
    c = dp.c
    b = dp.b
    x_min, x_max = x_c - c, x_c + c
    y_min, y_max = y_top - b, y_top
    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y

    cx = x_min + (np.arange(n_x) + 0.5) * cell_dx
    cy = y_min + (np.arange(n_y) + 0.5) * cell_dy

    return cell_C_flat, cell_rho, cx, cy, dp


def print_summary(cell_C_flat, cell_rho, dp):
    """Print a summary of negative/unphysical values."""
    n_cells = len(cell_rho)
    rho0 = dp.rho0
    n_params = cell_C_flat.shape[1]

    print(f"Total cells: {n_cells}")
    print(f"C params per cell: {n_params}")
    print(f"Background: rho0={rho0:.0f}, lambda={dp.lam:.2e}, mu={dp.mu:.2e}")
    print()

    # Density
    neg_rho = cell_rho < 0
    print(f"Density rho:")
    print(f"  range: [{cell_rho.min():.1f}, {cell_rho.max():.1f}]")
    print(f"  negative: {neg_rho.sum()} / {n_cells}")
    if neg_rho.any():
        print(f"  worst negative: {cell_rho[neg_rho].min():.1f}")
    print()

    # C_flat columns
    labels = {2: ["lambda", "mu"], 6: ["C11", "C12", "C22", "C33", "C34", "C44"]}
    names = labels.get(n_params, [f"p{i}" for i in range(n_params)])
    for i, name in enumerate(names):
        col = cell_C_flat[:, i]
        neg = col < 0
        print(f"  {name}: [{col.min():.2e}, {col.max():.2e}]  negative: {neg.sum()}")


def plot_params(cell_C_flat, cell_rho, cx, cy, dp, out_dir: Path):
    """Plot spatial maps of cell parameters, highlighting negatives."""
    n_x, n_y = len(cx), len(cy)
    n_params = cell_C_flat.shape[1]
    rho0 = dp.rho0

    labels = {2: [r"$\lambda$", r"$\mu$"], 6: ["C11", "C12", "C22", "C33", "C34", "C44"]}
    names = labels.get(n_params, [f"p{i}" for i in range(n_params)])

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Combined overview ---
    n_plots = 1 + n_params  # rho + each C param
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    def imshow_diverging(ax, data_2d, title, unit=""):
        vmin, vmax = data_2d.min(), data_2d.max()
        has_neg = vmin < 0
        if has_neg:
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=max(vmax, abs(vmin) * 0.1))
            cmap = "RdBu_r"
        else:
            norm = None
            cmap = "viridis"
        im = ax.imshow(data_2d.T, origin="lower", aspect="auto",
                        extent=[cx[0], cx[-1], cy[0], cy[-1]],
                        cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if unit:
            cbar.set_label(unit)
        if has_neg:
            n_neg = (data_2d < 0).sum()
            ax.set_title(f"{title}\n({n_neg} negative)", fontsize=10, color="red")

    # Density
    rho_2d = cell_rho.reshape(n_x, n_y)
    imshow_diverging(axes[0], rho_2d, r"$\rho$", "kg/m³")

    # C params
    for i, name in enumerate(names):
        c_2d = cell_C_flat[:, i].reshape(n_x, n_y)
        imshow_diverging(axes[i + 1], c_2d, name, "Pa")

    fig.suptitle("Optimized cell parameters", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_params_overview.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'cell_params_overview.png'}")
    plt.close(fig)

    # --- Negative-only mask plot ---
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    def imshow_mask(ax, data_2d, title):
        mask = (data_2d < 0).astype(float)
        if mask.any():
            im = ax.imshow(mask.T, origin="lower", aspect="auto",
                            extent=[cx[0], cx[-1], cy[0], cy[-1]],
                            cmap="Reds", vmin=0, vmax=1)
            n_neg = int(mask.sum())
            ax.set_title(f"{title}: {n_neg} negative", color="red")
        else:
            ax.imshow(np.zeros_like(data_2d).T, origin="lower", aspect="auto",
                       extent=[cx[0], cx[-1], cy[0], cy[-1]],
                       cmap="Greys", vmin=0, vmax=1)
            ax.set_title(f"{title}: all positive", color="green")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    imshow_mask(axes[0], rho_2d, r"$\rho$")
    for i, name in enumerate(names):
        c_2d = cell_C_flat[:, i].reshape(n_x, n_y)
        imshow_mask(axes[i + 1], c_2d, name)

    fig.suptitle("Negative parameter locations", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cell_params_negatives.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'cell_params_negatives.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect optimized cell parameters")
    parser.add_argument("npz", help="Path to optimized_params.npz")
    parser.add_argument("config", nargs="?", default="configs/cell_based.yaml",
                        help="Config YAML (default: configs/cell_based.yaml)")
    parser.add_argument("--n-x", type=int, default=50)
    parser.add_argument("--n-y", type=int, default=50)
    parser.add_argument("-o", "--output", default="output/inspect",
                        help="Output directory (default: output/inspect)")
    args = parser.parse_args()

    cell_C_flat, cell_rho, cx, cy, dp = load_cell_grid(
        args.npz, args.config, args.n_x, args.n_y)

    print_summary(cell_C_flat, cell_rho, dp)
    plot_params(cell_C_flat, cell_rho, cx, cy, dp, Path(args.output))


if __name__ == "__main__":
    main()
