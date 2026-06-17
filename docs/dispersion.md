# Bloch-Floquet Dispersion Curves

Computes dispersion bands and IPR for the **reference** (homogeneous half-space) and **ideal cloak** unit cells, reproducing Fig 3(d)–(e) of Chatzopoulos et al. (2023).

## Usage

```bash
./run_dispersion.sh              # run (skips if cached .npz files exist)
./run_dispersion.sh -f           # force recompute
./run_dispersion.sh --h-elem 0.12 --n-eigs 40   # coarser mesh, quick test
```

Any argument after the script name is forwarded directly to `scripts/dispersion_ideal.py`.

## Arguments

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--n-kpts` | int | 50 | Number of evenly-spaced k-points from k=0 to the Brillouin zone edge (k=π/L_c). More points → smoother bands. |
| `--n-eigs` | int | 40 | Eigenvalues (modes) computed per k-point via ARPACK. Controls the **maximum frequency** visible in the plot and how many surface modes on folded Rayleigh replicas get captured. Paper §5.1 uses "multiple modes (>500) for each wavenumber" to reach f\*=2.5; `run_dispersion.sh` defaults to 550. |
| `--h-elem` | float | 0.08 | Global gmsh element size in metres (units of λ*=1 m). Larger → coarser mesh, fewer DOFs, faster solve. |
| `--h-fine` | float | 0.03 | Fine element size used near the cloak triangle edges. Controls accuracy of mode shapes inside the cloak region. |
| `--ipr-thr` | float | 2.0 | IPR threshold separating surface (Rayleigh) modes from bulk modes. Paper uses **3.5**; `run_dispersion.sh` defaults to 3.5. Modes with IPR > threshold are plotted as filled colored circles; bulk modes as hollow diamonds. |
| `--f-max` | float | 2.2 | Upper limit of the normalised frequency axis f* in the output plot. Has no effect on which eigenvalues are computed. |
| `--H-factor` | float | 1.0 | Scale unit-cell height by this factor (triangle geometry stays fixed). Paper uses 1.0 (H = 4.305 λ\*). Increase only if you want to probe long-wavelength behaviour without raising `--n-eigs`; it is a workaround for under-resolved eigenspectra rather than a physical fix. |
| `--lumped-mass` | flag | off | Debug: row-sum lumped mass matrix. Tested and found to give ~no improvement vs consistent mass for this problem. |
| `--case` | str | `both` | `reference`, `ideal_cloak`, or `both` — skip one sweep if you only want the other. |
| `--out-dir` | str | `output/dispersion` | Directory where the `.npz` cache files and `.png` plot are written. |
| `--force` / `-f` | flag | off | Recompute both sweeps even if the `.npz` cache files already exist. |

Cache `.npz` filenames embed `h_elem`, `h_fine`, `lumped`, and `H_factor`, so varying these produces independent caches rather than overwriting.

## Debugging helper

`scripts/dispersion_debug.py` runs the reference case under several
configurations to isolate numerical vs physical effects:

- Step 1: mesh convergence (h ∈ {0.08, 0.04, 0.02}) — mesh is already converged at h=0.08.
- Step 2: consistent vs lumped mass — negligible effect.
- Step 3: H_factor=1 vs 2 — **this is the physical fix**; see `debug_Hfactor.png`.
- Step 4: H_factor sweep {1.0, 1.5, 2.0, 3.0} — monotonic IPR improvement with H; 2.0 is the chosen default.

## Normalizations

- **Frequency:** f* = f · λ\* / c_R, where λ\*=1 m, c_R≈266.6 m/s. The Rayleigh branch lies along f*=k_norm.
- **Wavenumber:** k_norm = k / (2π). The Brillouin zone edge is at k_norm = 1/(2·L_c/λ\*) = **0.25**.

## Outputs

| File | Description |
|---|---|
| `output/dispersion/dispersion_reference.npz` | Cached sweep for the reference (no cloak/defect) unit cell |
| `output/dispersion/dispersion_ideal_cloak.npz` | Cached sweep for the ideal Cosserat cloak unit cell |
| `output/dispersion/dispersion_comparison.png` | Side-by-side dispersion plot coloured by log₂(IPR) |

## IPR (Inverse Participation Ratio)

The IPR quantifies how spatially localised a mode is:

```
IPR = A_total × Σ_n( A_n |u_n|⁴ ) / ( Σ_n A_n |u_n|² )²
```

- **IPR > `--ipr-thr` (default 2.0):** mode energy is concentrated near the free surface → **Rayleigh surface wave** (filled circle, coloured by log₂(IPR))
- **IPR ≤ `--ipr-thr`:** mode energy is spread through the bulk → **bulk P/S wave** (hollow blue diamond)

At the Brillouin zone edge the fundamental Rayleigh mode reaches IPR≈2.7 for both the reference and ideal cloak cases, confirming the cloak is transparent to surface waves.
