# Surrogate

Neural surrogate mapping cell-wise microstructure `(C_eff, ρ_eff)` + frequency `f*` to scalar transmission loss, trained on FEM samples from `dataset_gen/`.

## Architecture

The cell grid is treated as a multi-channel image (`n_C_params + 1` channels: stiffness params stacked with density), so a 2D CNN trunk can exploit local spatial structure and translation equivariance across cells. The trunk compresses geometry into a single latent `z`, decoupling the expensive convolution from the cheap query. Frequency enters separately through a sinusoidal Fourier encoding, which lets a small MLP decoder produce a smooth, continuous response over `f*` without retraining per frequency and enables evaluating a whole spectrum from one trunk pass (`forward_spectrum`).

Two backbones are selectable via `ForwardFEM_CNN(backbone=...)`, both trained from scratch:

- `small_convnext` (default, ~1.6M params) — `SmallConvNeXt`, a ConvNeXt-style trunk sized for coarse cell grids. Stride-1 stem, three stages of 2 blocks with stride-2 downsamples between (spatial flow 10→10→5→2 / 20→20→10→5 / 50→50→25→12 before GAP). Works for 10×10–50×50 grids.
- `convnext_tiny` (~28M params) — torchvision ConvNeXt-Tiny with the stem swapped for `n_C_params+1` channels and the classifier head → `z_dim`. Works on 50×50 but **fails on ≤20×20** because ConvNeXt's 32× cumulative downsampling collapses the spatial dim past the last stage.

## Files

- `dataset.py` — `SurrogateDataset` (reads HDF5), `SurrogateSample` / `SurrogateBatch` (pydantic + jaxtyping), `collate_surrogate`, `make_dataloader`. Reshapes the flat cell axis to `(n_x, n_y)`; with `cloak_only=True` non-cloak cells are zeroed.
- `model.py` — `ForwardFEM_CNN`: CNN trunk over `(C, ρ)` channels → latent `z` (default `SmallConvNeXt`, opt-in `convnext_tiny`); sinusoidal `FourierFeatures(f)`; MLP `SpectrumDecoder` on `concat(z, ff(f))`. Two modes:
  - `forward_at(batch, f=None)` → `(B,)` one frequency per sample.
  - `forward_spectrum(batch, f_grid)` → `(B, F)` shared grid.
- `train.py` — `TrainConfig` (pydantic) + `train()`. AdamW + MSE, writes `config.json`, `history.json`, `best.pt`, `last.pt`, periodic `epoch-NNNN.pt` to `out_dir`. Supports `resume_from`.
- `loss.py` — MSE helper (minimax / mean-spectrum variants stubbed).

## HDF5 input

Produced by `dataset_gen/`:

| dataset | shape | meaning |
|---|---|---|
| `cell_C_flat` | `(N, n_cells, n_C_params)` | stiffness params per cell |
| `cell_rho` | `(N, n_cells)` | density per cell |
| `f_star` | `(N,)` | normalised frequency |
| `loss` | `(N,)` | transmission loss (target) |
| `sample_type` | `(N,)` | `"init"` / `"random*"` / `"smooth*"` / `"opt*"` |
| `cloak_mask` | `(n_cells,)` | shared boolean cloak-cell mask |

Cell index layout: `idx = ix * n_y + iy`.

## Usage

```bash
conda activate jax-fem-env
python -m surrogate.train output/surrogate_dataset.h5
```

Programmatic:

```python
from surrogate.train import TrainConfig, train
model, history = train(TrainConfig(data_path="output/surrogate_dataset.h5"))
```
