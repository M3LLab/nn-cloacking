#!/usr/bin/env python
"""Visualise cell decomposition and per-cell material properties.

Usage
-----
    python scripts/visualize_cells.py                        # default config
    python scripts/visualize_cells.py configs/default.yaml   # explicit config
    python scripts/visualize_cells.py --params output/optimized_params.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig, load_config
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import C_iso, CellMaterial


def _build_objects(cfg: SimulationConfig):
    params = DerivedParams.from_config(cfg)
    geometry = TriangularCloakGeometry.from_params(params)
    cell_decomp = CellDecomposition(geometry, cfg.cells.n_x, cfg.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(geometry, C0, params.rho0, cell_decomp,
                            n_C_params=cfg.cells.n_C_params)
    return params, geometry, cell_decomp, cell_mat


def _cell_rectangles(cell_decomp: CellDecomposition):
    """Return list of (4,2) vertex arrays for each cell."""
    dx, dy = cell_decomp.cell_dx, cell_decomp.cell_dy
    rects = []
    for cx, cy in cell_decomp.cell_centers:
        rects.append(np.array([
            [cx - dx / 2, cy - dy / 2],
            [cx + dx / 2, cy - dy / 2],
            [cx + dx / 2, cy + dy / 2],
            [cx - dx / 2, cy + dy / 2],
        ]))
    return rects


def plot_cell_layout(cell_decomp: CellDecomposition,
                     geometry: TriangularCloakGeometry,
                     save_path: str | None = None):
    """Plot cell grid coloured by cloak membership, with cloak outline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    rects = _cell_rectangles(cell_decomp)
    colors = ["#4a90d9" if m else "#dddddd" for m in cell_decomp.cloak_mask]
    coll = PolyCollection(rects, facecolors=colors, edgecolors="k",
                          linewidths=0.4, alpha=0.6)
    ax.add_collection(coll)

    # Cloak triangles
    g = geometry
    inner_tri = np.array([
        [g.x_c - g.c, g.y_top],
        [g.x_c, g.y_top - g.a],
        [g.x_c + g.c, g.y_top],
    ])
    outer_tri = np.array([
        [g.x_c - g.c, g.y_top],
        [g.x_c, g.y_top - g.b],
        [g.x_c + g.c, g.y_top],
    ])
    ax.add_patch(Polygon(inner_tri, closed=True, fill=False,
                         edgecolor="red", linewidth=2, label="Inner (defect)"))
    ax.add_patch(Polygon(outer_tri, closed=True, fill=False,
                         edgecolor="darkred", linewidth=2, linestyle="--",
                         label="Outer (cloak)"))

    # Cell centres
    cloak_centers = cell_decomp.cell_centers[cell_decomp.cloak_mask]
    ax.scatter(cloak_centers[:, 0], cloak_centers[:, 1], s=6, c="k", zorder=3)

    ax.set_xlim(cell_decomp.x_min - cell_decomp.cell_dx,
                cell_decomp.x_max + cell_decomp.cell_dx)
    ax.set_ylim(cell_decomp.y_min - cell_decomp.cell_dy,
                cell_decomp.y_max + cell_decomp.cell_dy)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Cell layout ({cell_decomp.n_x}x{cell_decomp.n_y}, "
                 f"{cell_decomp.n_cloak_cells} in cloak)")
    ax.legend(loc="lower right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_cell_materials(cell_decomp: CellDecomposition,
                        cell_C_flat: np.ndarray,
                        cell_rho: np.ndarray,
                        n_C_params: int = 4,
                        save_path: str | None = None):
    """Plot per-cell material properties as coloured patches."""
    cell_C_flat = np.asarray(cell_C_flat)
    cell_rho = np.asarray(cell_rho)
    rects = _cell_rectangles(cell_decomp)
    mask = cell_decomp.cloak_mask

    n_plots = min(n_C_params, 4) + 1  # up to 4 C components + rho
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # C components
    c_labels = ["C[0000]", "C[1111]", "C[0101]", "C[0011]"]
    for k in range(min(n_C_params, 4)):
        ax = axes[k]
        vals = cell_C_flat[:, k]
        cloak_vals = vals[mask]
        vmin, vmax = float(cloak_vals.min()), float(cloak_vals.max())
        if np.isclose(vmin, vmax):
            vmin -= abs(vmin) * 0.1 + 1
            vmax += abs(vmax) * 0.1 + 1
        norm_vals = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        cmap = plt.cm.viridis
        colors = [cmap(norm_vals[i]) if mask[i] else (0.9, 0.9, 0.9, 1.0)
                  for i in range(len(mask))]
        coll = PolyCollection(rects, facecolors=colors, edgecolors="k",
                              linewidths=0.3)
        ax.add_collection(coll)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=ax, fraction=0.046)
        ax.set_title(c_labels[k] if k < len(c_labels) else f"C[{k}]")
        ax.set_xlim(cell_decomp.x_min, cell_decomp.x_max)
        ax.set_ylim(cell_decomp.y_min, cell_decomp.y_max)
        ax.set_aspect("equal")

    # rho
    ax = axes[-1]
    cloak_rho = cell_rho[mask]
    vmin, vmax = float(cloak_rho.min()), float(cloak_rho.max())
    if np.isclose(vmin, vmax):
        vmin -= abs(vmin) * 0.1 + 1
        vmax += abs(vmax) * 0.1 + 1
    norm_vals = np.clip((cell_rho - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.cm.magma
    colors = [cmap(norm_vals[i]) if mask[i] else (0.9, 0.9, 0.9, 1.0)
              for i in range(len(mask))]
    coll = PolyCollection(rects, facecolors=colors, edgecolors="k",
                          linewidths=0.3)
    ax.add_collection(coll)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, fraction=0.046)
    ax.set_title("rho_eff")
    ax.set_xlim(cell_decomp.x_min, cell_decomp.x_max)
    ax.set_ylim(cell_decomp.y_min, cell_decomp.y_max)
    ax.set_aspect("equal")

    fig.suptitle("Per-cell material properties")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualise cell decomposition")
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to YAML config (default: built-in defaults)")
    parser.add_argument("--params", default=None,
                        help="NPZ file with optimized cell_C_flat and cell_rho")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output plots")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else SimulationConfig()
    if not cfg.cells.enabled:
        cfg.cells.enabled = True
        print("Note: cells.enabled was False, enabling for visualization.")

    params, geometry, cell_decomp, cell_mat = _build_objects(cfg)

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    plot_cell_layout(cell_decomp, geometry, save_path=str(out / "cells_layout.png"))

    if args.params:
        data = np.load(args.params)
        cell_C_flat = data["cell_C_flat"]
        cell_rho = data["cell_rho"]
    else:
        cell_C_flat = cell_mat.cell_C_flat
        cell_rho = cell_mat.cell_rho

    plot_cell_materials(cell_decomp, cell_C_flat, cell_rho,
                        n_C_params=cfg.cells.n_C_params,
                        save_path=str(out / "cells_material.png"))

    plt.show()


if __name__ == "__main__":
    main()
