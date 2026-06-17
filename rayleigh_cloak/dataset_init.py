"""Dataset-based initialisation for topology neural reparameterisation.

Matches each coarse cloak cell's target (C_eff, rho_eff) from transformational
elasticity to the nearest entry in the pre-computed HDF5 dataset.  The matched
geometry masks are downsampled and tiled into a fine pixel grid that the MLP is
pre-trained to reproduce.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

class DatasetEntries(NamedTuple):
    """Loaded dataset arrays (numpy)."""
    geometries: np.ndarray   # (N_samples, H, W) int8
    C_effs: np.ndarray       # (N_samples, 4, 4) float64
    rho_effs: np.ndarray     # (N_samples,) float64


def load_dataset(path: str | Path) -> DatasetEntries:
    """Load geometry, C_eff, and rho_eff from an HDF5 dataset."""
    with h5py.File(path, "r") as f:
        geometries = f["geometry"][:]      # (N, H, W) int8
        C_effs = f["C_eff"][:]             # (N, 4, 4)
        rho_effs = f["rho_eff"][:]         # (N,)
    return DatasetEntries(geometries, C_effs, rho_effs)


# ---------------------------------------------------------------------------
# Extract isotropic (λ, μ) from 4×4 augmented Voigt matrices
# ---------------------------------------------------------------------------

def _extract_lam_mu_from_voigt4(C4: np.ndarray) -> np.ndarray:
    """Extract (λ, μ) from 4×4 augmented Voigt C_eff matrices.

    For an isotropic tensor: λ = C_1122, μ = (C_1212 + C_1221) / 2.
    Works on a single (4,4) or a batch (N, 4, 4).

    Returns shape (..., 2) with columns [λ, μ].
    """
    # Augmented Voigt pairs: 0=(0,0), 1=(1,1), 2=(0,1), 3=(1,0)
    # C_1122 = M[0,1],  C_1212 = M[2,2],  C_1221 = M[2,3]
    lam = C4[..., 0, 1]
    mu = 0.5 * (C4[..., 2, 2] + C4[..., 2, 3])
    return np.stack([lam, mu], axis=-1)


# ---------------------------------------------------------------------------
# Nearest-neighbour matching on (λ, μ)
# ---------------------------------------------------------------------------

def match_cells_to_dataset(
    cell_lam_mu: np.ndarray,
    cell_rho: np.ndarray,
    cloak_mask: np.ndarray,
    dataset: DatasetEntries,
    *,
    rho_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each cloak cell to the nearest dataset entry in (λ, μ, ρ) space.

    Both target and dataset properties are normalised by the dataset range
    so that matching compares dimensionless fractions.

    Parameters
    ----------
    cell_lam_mu : (n_cells, 2) — target [λ, μ] per coarse cell
    cell_rho : (n_cells,) — target ρ per coarse cell
    cloak_mask : (n_cells,) bool
    dataset : loaded HDF5 dataset
    rho_weight : relative weight for density in the distance metric

    Returns
    -------
    matched_geoms : (n_cloak_cells, H, W) float32 — geometry masks
    matched_indices : (n_cloak_cells,) int — dataset indices
    """
    cloak_idx = np.where(cloak_mask)[0]
    n_cloak = len(cloak_idx)
    H, W = dataset.geometries.shape[1], dataset.geometries.shape[2]

    # Extract (λ, μ) from dataset 4×4 Voigt matrices
    ds_lam_mu = _extract_lam_mu_from_voigt4(dataset.C_effs)  # (N_ds, 2)
    ds_rho = dataset.rho_effs  # (N_ds,)

    # Normalise by dataset range so distance is dimensionless
    lam_mu_scale = np.maximum(ds_lam_mu.max(axis=0) - ds_lam_mu.min(axis=0), 1e-12)
    rho_scale = max(ds_rho.max() - ds_rho.min(), 1e-12)

    matched_geoms = np.zeros((n_cloak, H, W), dtype=np.float32)
    matched_indices = np.zeros(n_cloak, dtype=np.int32)

    for i, ci in enumerate(cloak_idx):
        target_lm = cell_lam_mu[ci]  # (2,)
        target_rho = cell_rho[ci]

        dlm = (ds_lam_mu - target_lm[None]) / lam_mu_scale[None]
        drho = (ds_rho - target_rho) / rho_scale

        dist = np.sum(dlm ** 2, axis=1) + rho_weight * drho ** 2
        idx = int(np.argmin(dist))

        matched_geoms[i] = dataset.geometries[idx].astype(np.float32)
        matched_indices[i] = idx

    return matched_geoms, matched_indices


# ---------------------------------------------------------------------------
# Pixel-grid target construction
# ---------------------------------------------------------------------------

def _downsample_block(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Downsample a 2D image by block-averaging to (target_h, target_w).

    Returns float values in [0, 1] representing the fraction of solid pixels
    in each block.
    """
    H, W = img.shape
    bh = H // target_h
    bw = W // target_w
    # Crop to exact multiple
    cropped = img[:bh * target_h, :bw * target_w]
    return cropped.reshape(target_h, bh, target_w, bw).mean(axis=(1, 3))


def build_pixel_targets(
    matched_geoms: np.ndarray,
    n_coarse_x: int,
    n_coarse_y: int,
    pixel_per_cell: int,
    cloak_mask_coarse: np.ndarray,
) -> np.ndarray:
    """Tile matched geometries into a full fine-pixel grid.

    Each matched geometry (50×50) is downsampled to
    ``pixel_per_cell × pixel_per_cell`` and placed into the corresponding
    coarse-cell position.  Non-cloak cells are filled with 1.0 (solid =
    background material).

    Parameters
    ----------
    matched_geoms : (n_cloak, H_ds, W_ds) float32 — matched geometry masks
    n_coarse_x, n_coarse_y : coarse cell grid dimensions
    pixel_per_cell : number of fine pixels per coarse cell edge
    cloak_mask_coarse : (n_coarse_cells,) bool

    Returns
    -------
    pixel_grid : (n_fine_x * n_fine_y,) float32 — flattened fine-pixel
        density targets, ordered to match ``CellDecomposition`` indexing
        (ix varies slowest: idx = ix * n_fine_y + iy).
    """
    ppc = pixel_per_cell
    n_fine_x = n_coarse_x * ppc
    n_fine_y = n_coarse_y * ppc

    # Start with all-solid (background)
    grid = np.ones((n_fine_x, n_fine_y), dtype=np.float32)

    ds_h, ds_w = ppc, ppc
    cloak_idx = np.where(cloak_mask_coarse)[0]

    for i, ci in enumerate(cloak_idx):
        # Coarse cell (ix, iy) in the coarse grid
        ix_coarse = ci // n_coarse_y
        iy_coarse = ci % n_coarse_y

        # Downsample the matched geometry
        geom = matched_geoms[i]  # (H, W) from dataset
        ds = _downsample_block(geom, ds_h, ds_w)

        # Place into the fine grid
        # Note: dataset geometry has row 0 = top = high y, but
        # CellDecomposition has iy increasing upward (low to high y).
        # Flip the downsampled geometry vertically.
        ds_flipped = ds[::-1, :]  # flip y

        ix_start = ix_coarse * ppc
        iy_start = iy_coarse * ppc
        # ds_flipped[px, py] -> grid[ix_start + px, iy_start + py]
        # But ds is (rows=y, cols=x) while grid is (ix, iy)
        # ds_flipped shape is (ppc_y, ppc_x), grid indexing is (ix, iy)
        # Transpose: ds[row, col] = ds[y, x] -> grid[x, y]
        grid[ix_start:ix_start + ppc, iy_start:iy_start + ppc] = ds_flipped.T

    return grid.ravel()


# ---------------------------------------------------------------------------
# MLP pre-training
# ---------------------------------------------------------------------------

def pretrain_mlp(
    theta: list[dict],
    cell_features: jnp.ndarray,
    cloak_idx: jnp.ndarray,
    target_densities: jnp.ndarray,
    n_steps: int = 500,
    lr: float = 1e-3,
) -> list[dict]:
    """Pre-train MLP weights so sigmoid(mlp(features)) ≈ target densities.

    Parameters
    ----------
    theta : MLP parameters (list of {W, b})
    cell_features : (n_pixels, n_features) Fourier features for ALL pixels
    cloak_idx : (n_cloak_pixels,) indices of cloak pixels
    target_densities : (n_cloak_pixels,) target density values in [0, 1]
    n_steps : number of gradient descent steps
    lr : learning rate

    Returns
    -------
    theta : updated MLP parameters
    """
    from rayleigh_cloak.neural_reparam import mlp_forward

    cloak_features = cell_features[cloak_idx]  # (n_cloak_pixels, n_features)
    targets = target_densities  # (n_cloak_pixels,)

    def loss_fn(theta):
        raw = mlp_forward(theta, cloak_features)  # (n_cloak_pixels, n_out)
        pred = jax.nn.sigmoid(raw)                  # (n_cloak_pixels, n_out)
        # All channels share the same density target at init
        return jnp.mean((pred - targets[:, None]) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    for step in range(n_steps):
        loss_val, grads = loss_and_grad(theta)
        theta = jax.tree.map(lambda p, g: p - lr * g, theta, grads)
        if step % 100 == 0 or step == n_steps - 1:
            print(f"  Pretrain step {step:4d} | loss = {loss_val:.6f}")

    return theta
