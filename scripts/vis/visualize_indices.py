#!/usr/bin/env python
"""Visualize node index selection for optimization boundary functions.

Shows which nodes are selected by:
  - get_top_surface_beyond_cloak_indices  (measurement region)
  - get_outside_cloak_indices             (full exterior field)

Usage
-----
    python scripts/visualize_indices.py                        # default config
    python scripts/visualize_indices.py configs/optimize.yaml  # explicit config
    python scripts/visualize_indices.py --mesh full            # show full mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rayleigh_cloak import load_config
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import (
    get_outside_cloak_indices,
    get_top_surface_beyond_cloak_indices,
)
from rayleigh_cloak.solver import _create_geometry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", nargs="?", default="configs/default.yaml")
    p.add_argument(
        "--mesh", choices=["cloak", "full", "both"], default="both",
        help="Which mesh to show (default: both)",
    )
    p.add_argument("--save", metavar="FILE", help="Save figure to file instead of showing")
    return p.parse_args()


def plot_mesh_indices(ax, pts, triangles, label_sets, title):
    """Plot mesh with highlighted node sets."""
    pts = np.asarray(pts)

    # Draw mesh triangles lightly
    if triangles is not None:
        from matplotlib.collections import PolyCollection
        polys = pts[triangles][:, :, :2]
        col = PolyCollection(polys, facecolor="none", edgecolor="#cccccc", linewidth=0.2)
        ax.add_collection(col)

    # Plot all nodes as small grey dots
    ax.scatter(pts[:, 0], pts[:, 1], s=1, c="#bbbbbb", zorder=1, label="_all")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (indices, legend_label), color in zip(label_sets, colors):
        if len(indices) == 0:
            continue
        selected = pts[indices]
        ax.scatter(
            selected[:, 0], selected[:, 1],
            s=12, c=color, zorder=3, label=legend_label,
        )

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, markerscale=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def compute_index_sets(mesh, geometry, dp):
    pts = mesh.points
    x_left = dp.x_off
    x_right = dp.x_off + dp.W

    top_idx = get_top_surface_beyond_cloak_indices(
        pts, geometry, dp.y_top, x_left, x_right,
    )
    outside_idx = get_outside_cloak_indices(
        pts, geometry, dp.x_off, dp.y_off, dp.W, dp.H,
    )
    return top_idx, outside_idx


def main():
    args = parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)

    print("Generating meshes…")
    full_mesh = generate_mesh_full(config, dp, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

    show_meshes = {
        "cloak": [(cloak_mesh, "Cloak mesh")],
        "full":  [(full_mesh, "Full mesh")],
        "both":  [(cloak_mesh, "Cloak mesh"), (full_mesh, "Full mesh")],
    }[args.mesh]

    ncols = len(show_meshes)
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), squeeze=False)

    for ax, (mesh, mesh_title) in zip(axes[0], show_meshes):
        print(f"Computing indices for {mesh_title}…")
        top_idx, outside_idx = compute_index_sets(mesh, geometry, dp)

        n_top = len(top_idx)
        n_out = len(outside_idx)
        print(f"  top_surface_beyond_cloak : {n_top} nodes")
        print(f"  outside_cloak            : {n_out} nodes")

        triangles = getattr(mesh, "cells", None)
        if triangles is not None and hasattr(triangles, "tolist"):
            triangles = np.asarray(triangles)

        label_sets = [
            (top_idx,     f"top_surface_beyond_cloak ({n_top})"),
            # (outside_idx, f"outside_cloak ({n_out})"),
        ]
        plot_mesh_indices(ax, mesh.points, triangles, label_sets, f"{mesh_title}")

    fig.suptitle("Node index selection — optimize boundary functions", fontsize=12)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
