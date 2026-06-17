"""Visualization script for cellular automata chiral unit cells.

Usage:
    python -m rayleigh_cloak.dataset.cellular_chiral.visualize [--n_samples 16] [--seed 0] [--outdir output/ca_chiral]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .generator import CAConfig, generate_chiral_unit_cell, generate_quadrant


def plot_quadrant_and_cell(
    quadrant: np.ndarray,
    unit_cell: np.ndarray,
    seed: int,
    save_path: Path | None = None,
) -> None:
    """Plot a single quadrant alongside its assembled chiral unit cell."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(quadrant, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title(f"Quadrant (seed={seed})")
    axes[0].axis("off")

    axes[1].imshow(unit_cell, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_title(f"Chiral unit cell (seed={seed})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_grid(
    unit_cells: list[np.ndarray],
    seeds: list[int],
    save_path: Path | None = None,
    ncols: int = 4,
) -> None:
    """Plot a grid of chiral unit cells."""
    n = len(unit_cells)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < n:
            ax.imshow(
                unit_cells[idx], cmap="gray_r", interpolation="nearest", vmin=0, vmax=1
            )
            ax.set_title(f"seed={seeds[idx]}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Chiral CA Unit Cells", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_ca_progression(
    seed: int = 42,
    config: CAConfig | None = None,
    save_path: Path | None = None,
) -> None:
    """Visualize the full CA pipeline: initial seed → CA steps → reverse map → connected → chiral."""
    if config is None:
        config = CAConfig()

    from .generator import (
        _assemble_chiral,
        _connect_regions,
        _initialize_grid,
        _reverse_map,
        _step_ca,
    )

    rng = np.random.default_rng(seed)
    N = config.grid_size

    grid, frozen = _initialize_grid(N, config.live_fraction, config.gate_width, rng)
    stages = [("Initial seed", grid.copy())]

    for i in range(config.num_iterations):
        grid = _step_ca(grid, frozen, config)
        if i in (0, 2, config.num_iterations - 1):
            stages.append((f"CA iter {i + 1}", grid.copy()))

    material = _reverse_map(grid)
    stages.append(("Reverse map", material.copy()))

    connected = _connect_regions(material, config.bridge_width)
    stages.append(("Connected", connected.copy()))

    chiral = _assemble_chiral(connected)
    stages.append(("Chiral cell", chiral.copy()))

    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle(f"CA Pipeline (seed={seed})", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_tiled(
    unit_cell: np.ndarray,
    seed: int,
    repeats: int = 3,
    save_path: Path | None = None,
) -> None:
    """Show the unit cell tiled in a repeats×repeats periodic arrangement."""
    tiled = np.tile(unit_cell, (repeats, repeats))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(tiled, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Tiled {repeats}×{repeats} (seed={seed})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize CA chiral unit cells")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument(
        "--outdir", type=str, default="output/ca_chiral", help="Output directory"
    )
    parser.add_argument(
        "--grid_size", type=int, default=25, help="Quadrant grid size"
    )
    parser.add_argument(
        "--gate_width", type=int, default=5, help="Gate width in pixels"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = CAConfig(grid_size=args.grid_size, gate_width=args.gate_width)

    # 1. Pipeline progression for first seed
    print(f"Generating CA progression (seed={args.seed})...")
    plot_ca_progression(seed=args.seed, config=config, save_path=outdir / "progression.png")

    # 2. Generate grid of samples
    unit_cells = []
    seeds = list(range(args.seed, args.seed + args.n_samples))
    print(f"Generating {args.n_samples} chiral unit cells...")
    for s in seeds:
        uc, _ = generate_chiral_unit_cell(config=config, seed=s)
        unit_cells.append(uc)

    plot_grid(unit_cells, seeds, save_path=outdir / "grid.png")

    # 3. Individual samples with quadrant
    (outdir / "individual").mkdir(exist_ok=True)
    for s in seeds[:4]:
        uc, q = generate_chiral_unit_cell(config=config, seed=s)
        plot_quadrant_and_cell(q, uc, s, save_path=outdir / f"individual/seed_{s}.png")

    # 4. Tiled view for a few samples
    for s in seeds[:4]:
        uc, _ = generate_chiral_unit_cell(config=config, seed=s)
        plot_tiled(uc, s, repeats=3, save_path=outdir / f"individual/tiled_{s}.png")

    print(f"Saved visualizations to {outdir}/")


if __name__ == "__main__":
    main()
