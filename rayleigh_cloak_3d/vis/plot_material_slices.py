"""Slice visualizations of optimized cell material fields along x direction.

For each x-index, renders a yz-plane heatmap showing λ, μ, ρ components,
masked to the cloak region only. Used during optimization via step_callback.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def extract_cell_params(mf, theta):
    """Extract cell-centre material parameters from MLP weights.

    Replicates the cell-centre evaluation from CellDecomposedNeuralField.evaluate
    but stops before expanding to quadrature points. Returns NumPy arrays.

    Args:
        mf: CellDecomposedNeuralField instance
        theta: MLP weights (list of dicts)

    Returns:
        (cell_C_flat, cell_rho): shape (n_cells, n_C_params) and (n_cells,)
    """
    from rayleigh_cloak_3d.neural import mlp_forward

    raw = mlp_forward(theta, mf._cell_features)          # (n_cells, n_C+1)
    rel_C = raw[:, : mf.n_C_params] * mf.output_scale
    rel_rho = raw[:, mf.n_C_params] * mf.output_scale
    mask = mf._cloak_mask_j.astype(raw.dtype)
    cell_C_flat = mf._cell_C_flat_init * (1.0 + rel_C * mask[:, None])
    cell_rho = mf._cell_rho_init * (1.0 + rel_rho * mask)
    return np.asarray(cell_C_flat), np.asarray(cell_rho)


def plot_x_slices(cell_C_flat, cell_rho, mf, step, out_dir):
    """Plot yz-plane heatmaps for each x-slice of the cell grid.

    For each ix (x-index), creates one PNG showing n_C_params + 1 subplots
    (C parameters + density), with cells outside the cloak masked as NaN
    (appears blank).

    Args:
        cell_C_flat: shape (n_cells, n_C_params)
        cell_rho: shape (n_cells,)
        mf: CellDecomposedNeuralField instance (provides cell_decomp, n_C_params)
        step: optimization step number (for filename)
        out_dir: output directory (Path or str)
    """
    cd = mf.cell_decomp
    n_x, n_y, n_z = cd.n_x, cd.n_y, cd.n_z
    n_C_params = mf.n_C_params
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    C_flat_grid = cell_C_flat.reshape(n_x, n_y, n_z, n_C_params)
    rho_grid = cell_rho.reshape(n_x, n_y, n_z)
    cloak_mask_3d = np.asarray(mf._cloak_mask_j).reshape(n_x, n_y, n_z)

    # Cell center coordinates
    cx = cd.x_min + (np.arange(n_x) + 0.5) * cd.cell_dx
    cy = cd.y_min + (np.arange(n_y) + 0.5) * cd.cell_dy
    cz = cd.z_min + (np.arange(n_z) + 0.5) * cd.cell_dz

    # Parameter labels
    _LABELS = {
        2: [r"$\lambda$", r"$\mu$"],
        3: [r"$C_{11}$", r"$C_{33}$", r"$C_{13}$"],
        9: [r"$C_{11}$", r"$C_{22}$", r"$C_{33}$", r"$C_{23}$", r"$C_{13}$",
            r"$C_{12}$", r"$C_{66}$", r"$C_{55}$", r"$C_{44}$"],
        21: [f"$C_{i},{j}$" for i in range(1, 7) for j in range(i, 7)],
    }
    param_labels = _LABELS.get(n_C_params,
                               [f"p{i}" for i in range(n_C_params)])

    # Plot one PNG per x-slice, each showing all C params + rho as subplots
    for ix in range(n_x):
        x_val = cx[ix]
        fig, axes = plt.subplots(1, n_C_params + 1, figsize=(5 * (n_C_params + 1), 5))
        axes = np.atleast_1d(axes)

        # C components
        for k in range(n_C_params):
            ax = axes[k]
            vals = C_flat_grid[ix, :, :, k].copy()  # (n_y, n_z)
            mask = cloak_mask_3d[ix, :, :]           # (n_y, n_z)
            vals = vals.astype(float)
            vals[~mask] = np.nan                      # mask out-of-cloak as NaN

            im = ax.pcolormesh(cy, cz, vals.T / 1e9, shading='auto')
            fig.colorbar(im, ax=ax, label="GPa")
            ax.set_title(f"{param_labels[k]} — x={x_val:.3f}")
            ax.set_xlabel("y [m]")
            ax.set_ylabel("z [m]")
            ax.set_aspect('equal')

        # Density
        ax = axes[n_C_params]
        vals = rho_grid[ix, :, :].copy()  # (n_y, n_z)
        mask = cloak_mask_3d[ix, :, :]
        vals = vals.astype(float)
        vals[~mask] = np.nan

        im = ax.pcolormesh(cy, cz, vals.T, shading='auto')
        fig.colorbar(im, ax=ax, label="kg/m³")
        ax.set_title(r"$\rho$ — x={:.3f}".format(x_val))
        ax.set_xlabel("y [m]")
        ax.set_ylabel("z [m]")
        ax.set_aspect('equal')

        fig.suptitle(f"Material fields — step {step}, x-slice {ix}", fontsize=14)
        fig.tight_layout()

        fname = out_dir / f"step_{step:04d}_x{ix:02d}.png"
        fig.savefig(fname, dpi=100)
        plt.close(fig)
