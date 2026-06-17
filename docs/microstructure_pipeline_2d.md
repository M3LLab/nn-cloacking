# Microstructure-grounded cloak pipeline

End-to-end flow: generate a CA microstructure dataset → homogenise it → fit a
GMM density on (λ, μ, ρ) → optimise a cloak with the GMM as a flat-top prior →
snap each cloak cell to its nearest dataset entry → validate at pixel level.

All commands assume repo root is the working directory.

## 1. Generate CA unit cells

Random squared-assembly cellular-automata cells, one big memmapped `.npy`.

```bash
python -m dataset.cellular_chiral.bulk_generate \
    -n 1000000 \
    -o output/ca_bulk_squared
```

Outputs: `output/ca_bulk_squared/{cells.npy, live_fractions.npy}`.

## 2. FEM homogenisation

Periodic FEM on every unique cell → effective Lamé (λ, μ), density ρ, full
stiffness `C_eff`. Exact + fuzzy dedup, per-sample timeout, multi-process.

```bash
python -m dataset.cellular_chiral.bulk_stiffness \
    -i output/ca_bulk_squared \
    -o output/ca_bulk_squared/stiffness.h5 \
    -j 8
```

Resume after interruption with `--resume`.

## 3. Fit the GMM density

Gaussian mixture on standardised (λ, μ, ρ). Saves means/precisions and the
log-p quantiles used to set the flat-top threshold τ.

```bash
python -m dataset.cellular_chiral.fit_gmm \
    -i output/ca_bulk_squared/stiffness.h5 \
    -o output/ca_bulk_squared/gmm_lambda_mu_rho.npz \
    -K 16 \
    --threshold-percentile 0.25
```

## 4. Construct YAML config

Initialize with cement material or with the dataset centroid:

```yaml
cells:
  init: dataset_centroid
  init_path: output/ca_bulk_squared/gmm_lambda_mu_rho.npz
```

Add the GMM prior:
```yaml
loss:
  regularizations:
    material_cement_GMM:
      enabled: true
      path: output/ca_bulk_squared/gmm_lambda_mu_rho.npz
      weight: 1.0
      quantile: 0.25     # flat-top quantile in [0.01, 0.75]; null → use τ from .npz
```

Lower quantile → looser margin (fewer cells penalised); higher → stricter
(push deeper into the dataset manifold).

## 5. Run the optimisation

```bash
python run_optimize.py configs/cell20_cement_reg2_threshold50.yaml
```

Outputs go to `<output_dir>/optimized_params.npz` (cell_C_flat, cell_rho).

## 6. Snap cells to nearest microstructure & tile

For each cloak cell, find the dataset entry with the closest standardised
(λ, μ, ρ) and tile the matched 50×50 binaries.

```bash
python -m scripts.vis.tile_matched_microstructure \
    -p output/<run>/optimized_params.npz \
    -d output/ca_bulk_squared/stiffness.h5 \
    -c output/<run>/config.yaml \
    -o output/<run>/tiled_microstructure.png
```

Side-car `.npz` carries the per-cell match index + std-L2 distance.

## 7. Pixel-level frequency-sweep validation

Re-solves the cloak with material assigned at the pixel level on a refined
mesh, then plots ⟨|u|⟩/⟨|u_ref|⟩ vs. f\* alongside any existing
obstacle/ideal/optimised CSVs.

```bash
python scripts/frequency_sweep_validated.py \
    configs/multifreq.yaml \
    output/<run>/optimized_params.npz \
    --fmin 0.7 --fmax 3.3 --fstep 0.1 \
    --refinement-factor 25
```

The default refinement (25) gives ~1 FEM element per micro-pixel side; lower
values alias the microstructure. CSVs and the comparison plot land in the
params directory.
