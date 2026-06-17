"""Visualize the polar cell decomposition of the Nassar lattice cloak.

Produces:
  1. Polar cell grid overlaid on the annular cloak region
  2. Per-cell deformation parameter f
  3. Per-cell aspect ratio a/b
  4. Per-cell density ρ_eff
  5. Comparison: polar vs Cartesian cell grids
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.cells_polar import PolarCellDecomposition
from rayleigh_cloak.config import DerivedParams, load_config
from rayleigh_cloak.nassar import NassarPolarMaterial, nassar_init_from_lame
from rayleigh_cloak.solver import _create_geometry


def plot_polar_grid(pd: PolarCellDecomposition, geometry, output_dir: str):
    """Plot the polar cell grid with sector/layer boundaries."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ri, rc = pd.ri, pd.rc
    x_c, y_c = pd.x_c, pd.y_c

    # Draw radial edges
    for r in pd.r_edges:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                '-', color='steelblue', lw=0.5, alpha=0.7)

    # Draw angular edges
    for phi in pd.phi_edges[:-1]:  # skip duplicate 2π = 0
        ax.plot([ri * np.cos(phi), rc * np.cos(phi)],
                [ri * np.sin(phi), rc * np.sin(phi)],
                '-', color='steelblue', lw=0.5, alpha=0.7)

    # Draw cell centres
    cx = pd.cell_centers[:, 0] - x_c
    cy = pd.cell_centers[:, 1] - y_c
    ax.scatter(cx, cy, s=3, c='red', zorder=5, label='cell centres')

    # Inner and outer circles
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ri * np.cos(theta), ri * np.sin(theta),
            'k-', lw=2, label=f'$r_i$ = {ri*1e3:.0f} mm')
    ax.plot(rc * np.cos(theta), rc * np.sin(theta),
            'k--', lw=2, label=f'$r_c$ = {rc*1e3:.0f} mm')

    ax.set_aspect('equal')
    ax.set_title(f'Polar cell grid: {pd.N} sectors × {pd.M} layers '
                 f'= {pd.n_cells} cells')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(fontsize=8, loc='upper right')

    path = os.path.join(output_dir, 'polar_grid.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_cell_scalar(pd: PolarCellDecomposition, values: np.ndarray,
                     title: str, fname: str, output_dir: str,
                     cmap='viridis'):
    """Plot a per-cell scalar field on the polar grid using wedge patches."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ri, rc = pd.ri, pd.rc
    M, N = pd.M, pd.N

    patches = []
    for j in range(M):
        for k in range(N):
            r_inner = pd.r_edges[j]
            r_outer = pd.r_edges[j + 1]
            phi_start = np.degrees(pd.phi_edges[k])
            phi_end = np.degrees(pd.phi_edges[k + 1])
            wedge = mpatches.Wedge(
                (0, 0), r_outer, phi_start, phi_end,
                width=r_outer - r_inner,
            )
            patches.append(wedge)

    pc = PatchCollection(patches, cmap=cmap, edgecolors='grey',
                         linewidths=0.3)
    pc.set_array(np.asarray(values))
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.7, label=title)

    # Inner/outer circles
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ri * np.cos(theta), ri * np.sin(theta), 'k-', lw=1.5)
    ax.plot(rc * np.cos(theta), rc * np.sin(theta), 'k--', lw=1.5)

    margin = 0.02
    lim = rc + margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_grid_comparison(pd: PolarCellDecomposition, cd: CellDecomposition,
                         geometry, output_dir: str):
    """Side-by-side comparison of polar vs Cartesian cell grids."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ri, rc = pd.ri, pd.rc
    x_c, y_c = pd.x_c, pd.y_c
    theta = np.linspace(0, 2 * np.pi, 200)

    # ── Left: Polar ──
    for r in pd.r_edges:
        ax1.plot(r * np.cos(theta), r * np.sin(theta),
                 '-', color='steelblue', lw=0.4, alpha=0.6)
    for phi in pd.phi_edges[:-1]:
        ax1.plot([ri * np.cos(phi), rc * np.cos(phi)],
                 [ri * np.sin(phi), rc * np.sin(phi)],
                 '-', color='steelblue', lw=0.4, alpha=0.6)
    ax1.plot(ri * np.cos(theta), ri * np.sin(theta), 'k-', lw=2)
    ax1.plot(rc * np.cos(theta), rc * np.sin(theta), 'k--', lw=2)
    ax1.set_title(f'Polar: {pd.N}×{pd.M} = {pd.n_cells} cells')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    lim = rc * 1.15
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)

    # ── Right: Cartesian ──
    # Grid lines
    for ix in range(cd.n_x + 1):
        x = cd.x_min + ix * cd.cell_dx - x_c
        ax2.plot([x, x], [cd.y_min - y_c, cd.y_max - y_c],
                 '-', color='coral', lw=0.4, alpha=0.6)
    for iy in range(cd.n_y + 1):
        y = cd.y_min + iy * cd.cell_dy - y_c
        ax2.plot([cd.x_min - x_c, cd.x_max - x_c], [y, y],
                 '-', color='coral', lw=0.4, alpha=0.6)
    # Highlight cloak cells
    for i in cd.cloak_cell_indices:
        cx = cd.cell_centers[i, 0] - x_c
        cy = cd.cell_centers[i, 1] - y_c
        ax2.plot(cx, cy, '.', color='coral', ms=2)
    ax2.plot(ri * np.cos(theta), ri * np.sin(theta), 'k-', lw=2)
    ax2.plot(rc * np.cos(theta), rc * np.sin(theta), 'k--', lw=2)
    ax2.set_title(f'Cartesian: {cd.n_x}×{cd.n_y} = {cd.n_cells} cells '
                  f'({cd.n_cloak_cells} in cloak)')
    ax2.set_aspect('equal')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)

    fig.suptitle('Polar vs Cartesian cell decomposition', fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, 'grid_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/nassar_polar.yaml'
    cfg = load_config(config_path)
    params = DerivedParams.from_config(cfg)
    geometry = _create_geometry(cfg, params)

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Polar decomposition
    pd = PolarCellDecomposition(
        ri=params.ri, rc=params.rc,
        x_c=geometry.x_c, y_c=geometry.y_c,
        N=cfg.nassar.lattice_N, M=cfg.nassar.lattice_M,
    )
    print(f'Polar grid: {pd.N} sectors × {pd.M} layers = {pd.n_cells} cells')
    print(f'  r_edges: [{pd.r_edges[0]*1e3:.1f}, ..., {pd.r_edges[-1]*1e3:.1f}] mm')
    print(f'  Cell a range: [{pd.cell_a.min()*1e3:.2f}, {pd.cell_a.max()*1e3:.2f}] mm')
    print(f'  Cell b range: [{pd.cell_b.min()*1e3:.2f}, {pd.cell_b.max()*1e3:.2f}] mm')

    # Material
    nassar_mat = NassarPolarMaterial(
        geometry, params.lam, params.mu, params.rho0, pd,
    )

    print('\nPlotting...')
    plot_polar_grid(pd, geometry, output_dir)
    plot_cell_scalar(pd, pd.cell_f, 'Deformation $f = (r - r_i)/r$',
                     'polar_f.png', output_dir)
    plot_cell_scalar(pd, np.asarray(nassar_mat.cell_aspect),
                     'Aspect ratio $a/b$', 'polar_aspect.png', output_dir)
    plot_cell_scalar(pd, np.asarray(nassar_mat.cell_rho),
                     r'Density $\rho_{eff}$ [kg/m²]', 'polar_rho.png',
                     output_dir, cmap='inferno')

    # Comparison with Cartesian grid (same total cells approximately)
    n_cart = int(np.sqrt(pd.n_cells * 4 / np.pi))  # rough equivalent
    cd = CellDecomposition(geometry, n_cart, n_cart)
    print(f'\nCartesian grid: {cd.n_x}×{cd.n_y} = {cd.n_cells} cells '
          f'({cd.n_cloak_cells} in cloak)')
    plot_grid_comparison(pd, cd, geometry, output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
