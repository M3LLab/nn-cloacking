"""Visualise the initial density field right after dataset matching.

Runs Steps 1–6 of the topo pipeline (no FEM solve, no optimisation),
builds pixel targets from matched geometries, and saves a density plot.

Usage:
    python scripts/plot_matched_density.py configs/triangular_optimize_neural_topo.yaml
"""

import sys
import numpy as np
import jax.numpy as jnp

from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.materials import (
    C_iso, C_to_flat2, symmetrize_stiffness,
    C_eff as compute_C_eff, rho_eff as compute_rho_eff,
)
from rayleigh_cloak.dataset_init import (
    build_pixel_targets, load_dataset, match_cells_to_dataset,
)
from rayleigh_cloak.solver import _create_geometry
from rayleigh_cloak.neural_reparam_topo import plot_density_grid


def main(config_path: str) -> None:
    config = load_config(config_path)
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)
    topo_cfg = config.optimization.topo_neural

    # Coarse cell decomposition
    coarse_decomp = CellDecomposition(
        geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)

    print(f"Coarse grid: {coarse_decomp.n_cells} cells, "
          f"{coarse_decomp.n_cloak_cells} in cloak")

    # Compute target (λ, μ) per coarse cell
    coarse_lam_mu = np.zeros((coarse_decomp.n_cells, 2))
    coarse_rho = np.zeros(coarse_decomp.n_cells)
    for i, center in enumerate(coarse_decomp.cell_centers):
        if coarse_decomp.cloak_mask[i]:
            x = jnp.array(center)
            C_i = compute_C_eff(x, geometry, C0)
            C_sym = symmetrize_stiffness(C_i)
            coarse_lam_mu[i] = np.asarray(C_to_flat2(C_sym))
            coarse_rho[i] = float(compute_rho_eff(x, geometry, params.rho0))
        else:
            coarse_lam_mu[i] = np.array([params.lam, params.mu])
            coarse_rho[i] = params.rho0

    # Match to dataset
    dataset = load_dataset(topo_cfg.dataset_path)
    print(f"Dataset: {len(dataset.geometries)} entries")

    matched_geoms, matched_idx = match_cells_to_dataset(
        coarse_lam_mu, coarse_rho, coarse_decomp.cloak_mask,
        dataset, rho_weight=topo_cfg.rho_weight,
    )
    print(f"Matched {len(matched_idx)} cloak cells to dataset entries")

    # Build fine pixel grid
    ppc = topo_cfg.pixel_per_cell
    n_fine_x = config.cells.n_x * ppc
    n_fine_y = config.cells.n_y * ppc
    fine_decomp = CellDecomposition(geometry, n_fine_x, n_fine_y)
    print(f"Fine grid: {n_fine_x}×{n_fine_y} = {fine_decomp.n_cells} pixels")

    pixel_targets = build_pixel_targets(
        matched_geoms, config.cells.n_x, config.cells.n_y,
        ppc, coarse_decomp.cloak_mask,
    )

    # Reshape to 2D for plotting (same logic as NeuralReparamTopo.get_density_grid)
    density_grid = pixel_targets.reshape(n_fine_x, n_fine_y).T  # (n_y, n_x)

    cloak_mask_2d = np.array(fine_decomp.cloak_mask).reshape(
        n_fine_x, n_fine_y).T  # (n_y, n_x)

    save_path = f"{config.output_dir}/matched_density_init.png"
    plot_density_grid(density_grid, step=0, save_path=save_path,
                      cloak_mask_2d=cloak_mask_2d)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/triangular_optimize_neural_topo.yaml"
    main(cfg)
