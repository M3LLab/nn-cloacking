# Bloch–Floquet Dispersion

Computes dispersion bands and IPR for a vertical strip unit cell, Bloch-periodic
along the surface direction, fixed at the bottom and traction-free on top —
the construction of Fig. 3(d)–(e) in Chatzopoulos et al. (2023), and Fig. 8 of
the Wave Motion paper.

Three cell types are available:

| Case | Cell | Material |
|---|---|---|
| `reference` | flat rectangle, no notch | homogeneous isotropic |
| `ideal_cloak` | triangular void cut out | analytic transformation elasticity (`C_eff`, `rho_eff`) |
| `optimized_cloak` | triangular void cut out | per-cell values from `optimized_params.npz` |

## Scripts

| Script | Use |
|---|---|
| `scripts/dispersion/dispersion_jaxfem.py` | **The one to use.** Config-driven; supports all three cases. |
| `scripts/dispersion/plot_dispersion_overlay.py` | Overlays several cached sweeps in one figure (e.g. ideal vs optimised). |
| `scripts/dispersion/dispersion_ideal.py` | Legacy. Hardcoded parameters, `reference`/`ideal_cloak` only. |
| `scripts/dispersion/dispersion_debug.py` | Mesh/mass/H-factor convergence diagnostics. |

`dispersion_ideal.py` hardcodes `rho0=1600`, `cs=300`, `H=4.305λ*`, `a=0.0774H`,
`b=3a`, `c=0.1545H`, `L_c=2λ*` and the same `cR` formula, so for a config
carrying those values it is numerically **redundant** with
`dispersion_jaxfem.py --case both`. Prefer the config-driven script: it cannot
drift out of sync with the run you are analysing.

> The old `scripts/dispersion/shell_commands/*.sh` wrappers were removed: all
> four had a fatal `cd "$(dirname "$0")/.."` that resolved one level short of
> the repo root, and `run_dispersion_diagnose.sh` additionally omitted
> `--n-cells-x/--n-cells-y` (silently producing a wrong figure — see Gotcha 2).
> Recover with `git checkout -- scripts/dispersion/shell_commands/` if needed.

## Setup

```bash
cd /home/m3l/workspace/nn-cloaking
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax-fem-env
```

All commands below assume `PYTHONPATH=.` from the repo root.

---

## Example 1 — Smoke test (~8 s)

Verifies the whole pipeline end-to-end. Writes to `/tmp` so it can never
pollute a real sweep cache.

```bash
rm -rf /tmp/disp_smoke

PYTHONPATH=. python scripts/dispersion/dispersion_jaxfem.py \
  output/B_multifreq_14x10_nogmm_restart2/config.yaml \
  --case optimized \
  --params-npz output/B_multifreq_14x10_nogmm_restart2/optimized_params.npz \
  --n-kpts 3 --n-eigs 60 \
  --h-elem 0.08 --h-fine 0.03 \
  --f-max 1.5 --ipr-thr 2.5 \
  --workers 3 \
  --out-dir /tmp/disp_smoke
```

**What to check:**

- `Cell grid: 14x10, n_C_params=4 (config; matches npz ✓)` — the grid was read
  from `cells:` and cross-checked against `cell_C_flat.shape`
- `a = 0.3332, b = 0.9996, c = 0.6651`, `BZ edge: k_norm = 0.250`
- **No** `⚠ Clamping …` lines — the material is untouched (all 140 cells are
  positive-definite, so nothing is silently altered)
- `n_eigs=60` reaches only `f*≈1.48`, hence `--f-max 1.5`

Only 3 k-points, so the plot is three vertical stripes — this checks that it
*runs*, not that the physics is resolved.

## Example 2 — Preview run (~4.5 min)

Full frequency range at production `n_eigs`, but a coarse mesh and coarse
k-sampling. This is the one to run when you want to *see* the bands before
committing an hour.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python scripts/dispersion/dispersion_jaxfem.py \
  output/B_multifreq_14x10_nogmm_restart2/config.yaml \
  --case optimized \
  --params-npz output/B_multifreq_14x10_nogmm_restart2/optimized_params.npz \
  --n-kpts 16 --n-eigs 400 \
  --h-elem 0.08 --h-fine 0.03 \
  --f-max 3.5 --ipr-thr 2.5 \
  --workers 8 --force \
  --out-dir /tmp/disp_preview
```

Measured: 8 k-points at these settings took **133 s** → 16 k-points ≈
**4.5 min**, 24 k-points ≈ 7 min. See [Why `--workers` barely
helps](#why---workers-barely-helps) for the env vars.

The mesh is deliberately the coarse one — at `h_elem=0.08` the far-field
resolution is only ~4.2 points per wavelength at `f*=3` for linear TRI3, so band
positions carry a few percent of numerical stiffening. Fine for shape, not for
publication.

## Example 3 — Production run (~23 min per case)

Refined mesh, full k-sampling, both the reference and the optimised cell.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. nohup python -u scripts/dispersion/dispersion_jaxfem.py \
  output/B_multifreq_14x10_nogmm_restart2/config.yaml \
  --case optimized_vs_ref \
  --params-npz output/B_multifreq_14x10_nogmm_restart2/optimized_params.npz \
  --n-kpts 50 --n-eigs 400 \
  --h-elem 0.05 --h-fine 0.025 \
  --f-max 3.5 --ipr-thr 2.5 \
  --workers 8 --force \
  > /tmp/bloch_fig8.log 2>&1 &

tail -f /tmp/bloch_fig8.log | grep --line-buffered "\[k \|Saved\|Plot"
```

Use `--case optimized` to skip the reference sweep and halve the time.

**Timing.** Measured directly at these settings: 4 k-points in **114 s** with
`--workers 8` and BLAS pinned → **~23 min per case**, so `optimized_vs_ref` is
about **46 min**. Without pinning, budget ~28 min per case. Memory is ~1.9 GB per
worker (~15 GB at `--workers 8`).

### Sizing rationale

- **`--n-eigs 400`** — dominant cost knob. Mode count below `f*` grows as `f*²`;
  60 eigenvalues reach `f*=1.48`, and 400 was measured to reach **`f*=4.01`**,
  comfortably above `--f-max 3.5`. The 550 previously used reaches `f*≈4.5` and
  costs roughly twice as much for nothing.
- **`--h-elem 0.05 --h-fine 0.025`** — the 14×10 macro cells are 0.095 wide, so
  `h_fine=0.025` puts ~3.8 elements across each one (a piecewise-constant
  material needs at least 2–3), and far-field resolution is ~6.7 points per
  wavelength at `f*=3`. The 0.08/0.03 default gives ~3 elements per cell and 4.2
  points per wavelength — under-resolved.
- **`--workers 8` + pinned BLAS** — the best measured configuration, but worth
  only ~4% over serial; pinning is what matters (15% over unpinned). See below.

---

## Arguments

### Correctness-critical — must match the optimisation run

| Argument | Meaning |
|---|---|
| `config` (positional) | Source of the physical parameters (`rho0`, `cs`, geometry factors, `H_factor`) **and** the cell decomposition. Use the config **from the run's output directory**, not the one in `configs/`, so it reflects what was actually trained. |
| `--params-npz` | Per-cell `cell_C_flat` and `cell_rho`. Required for `optimized` / `optimized_vs_ref`. |
| `--case` | Which cells to compute — see below. |

The cell decomposition is **read from the config** (`cells.n_x`, `cells.n_y`,
`cells.n_C_params`), so for a normal run you do not pass it at all. The script
prints what it resolved and cross-checks it against the `.npz`:

```
Cell grid: 14x10, n_C_params=4 (config; matches npz ✓)
```

A mismatch is a hard error, not a wrong figure:

```
ERROR: cell grid does not match output/…/optimized_params.npz
  using : 50x50 = 2500 cells x 4 params
  npz has: 140 cells x 4 params
  Fix cells.n_x / n_y / n_C_params in output/…/config.yaml, or pass
  --n-cells-x / --n-cells-y / --n-C-params explicitly.
```

| Override | Default |
|---|---|
| `--n-cells-x` | `cells.n_x` from the config |
| `--n-cells-y` | `cells.n_y` from the config |
| `--n-C-params` | `cells.n_C_params` from the config |

Pass these only to deliberately analyse a grid other than the one that was
trained; the same consistency check still applies, and the banner names what you
overrode:

```
Cell grid: 14x10, n_C_params=4 (config, overridden: --n-cells-x; matches npz ✓)
```

**All of this is checked before any sweep starts.** A missing `--params-npz`, a
path that does not exist, an `.npz` lacking `cell_C_flat`/`cell_rho`, or a grid
mismatch all exit within a second — rather than after the reference sweep has
already burned half an hour. Passing `--params-npz` to a case that ignores it
(`reference`, `ideal_cloak`, `both`) prints a `NOTE` and continues.

`--case` values:

| Value | Computes | Produces |
|---|---|---|
| `reference` | reference only | single plot |
| `ideal_cloak` | analytic cloak only | single plot |
| `optimized` | optimised cloak only | single plot |
| `both` | reference + ideal | comparison + 2 singles |
| `optimized_vs_ref` | reference + optimised | comparison + 2 singles |

There is no built-in `ideal_vs_optimized` — see [Combining sweeps](#combining-sweeps-in-one-figure).

### Cost knobs — affect runtime and accuracy

| Argument | Default | Meaning |
|---|---|---|
| `--n-kpts` | 50 | k-points from ~0 to the Brillouin-zone edge (π/L_c). Cost is linear. Controls band smoothness only. |
| `--n-eigs` | 40 | Eigenvalues per k-point (ARPACK shift-invert at σ=0). **Sets the maximum `f*` reached** and dominates cost (between linear and quadratic). Does not affect accuracy of the modes it does find. |
| `--h-elem` | 0.08 | Global gmsh element size, in units of λ*=1 m. Sets far-field resolution. |
| `--h-fine` | 0.03 | Element size near the cloak triangle edges. Must resolve the macro-cell material jumps. |
| `--workers`, `-j` | 1 | k-points solved concurrently. Each holds its own LU factorisation (~1.9 GB here). **Scales far worse than linearly** — see below. |
| `--H-factor` | 1.0 | Scales cell height; triangle dimensions stay fixed. Leave at 1.0 — it is a workaround for under-resolved eigenspectra, not a physical fix. |
| `--lumped-mass` | off | Row-sum lumped mass. Tested: negligible effect. Debug only. |
| `--force` | off | Recompute even when a matching cache exists. Rarely needed now that caches are fingerprinted — see Gotcha 1. |
| `--out-dir` | `<config output_dir>/dispersion` | Where `.npz` and `.png` land. |

### Why `--workers` barely helps

Counter-intuitively, raising `--workers` buys almost nothing, and raising it
*without* pinning BLAS makes things worse. Measured, 8 k-points at
`h=0.08/0.03`, `n_eigs=400`, on a 32-core box:

| Setting | Wall time |
|---|---|
| `--workers 8`, `OMP/OPENBLAS/MKL_NUM_THREADS=1` | **133 s** |
| `--workers 1` (BLAS free to use all cores) | 139 s |
| `--workers 8`, BLAS unpinned | 157 s |

Eight-way concurrency is worth **4%** over serial, and unpinned it is 13%
*slower* than serial. The cause is structural: `run_sweep` uses a
`ThreadPoolExecutor`, and each k-point solve is
`eigsh(..., OPinv=LinearOperator(matvec=lu.solve))` — so ARPACK calls back into
**Python** for every matrix-vector product. Those callbacks take the GIL, which
serialises the workers. Unpinned, the nested BLAS threads then oversubscribe the
machine on top of that.

Practical guidance:

- Always pin BLAS when using `--workers > 1`:
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
- `--workers 8` pinned is the best of the measured configurations; going higher
  costs memory (~1.9 GB each) for no gain.
- Do not budget as though `--workers N` divides the runtime by N.

Real parallelism would need processes rather than threads (each k-point is fully
independent — same `K`, `M`, only `k` differs), or an ARPACK path whose matvec
stays in C. Neither is implemented.

### Plot-only — free to change, reuse the cache

Changing only these and dropping `--force` re-renders in seconds.

| Argument | Default | Meaning |
|---|---|---|
| `--ipr-thr` | 2.0 | IPR above which a mode is drawn as a large outlined marker (surface/Rayleigh) instead of a small dot (bulk). Chatzopoulos et al. use **3.5**; 2.5 is a good middle ground. Purely visual — every mode's IPR is in the `.npz` regardless. |
| `--f-max` | 2.2 | Upper limit of the `f*` axis. Does not change which eigenvalues are computed. |

Re-plot at a different threshold — same command, **drop `--force`**, change only
`--ipr-thr`:

```bash
PYTHONPATH=. python scripts/dispersion/dispersion_jaxfem.py \
  output/B_multifreq_14x10_nogmm_restart2/config.yaml \
  --case optimized_vs_ref \
  --params-npz output/B_multifreq_14x10_nogmm_restart2/optimized_params.npz \
  --n-kpts 50 --n-eigs 400 --h-elem 0.05 --h-fine 0.025 \
  --f-max 3.5 --ipr-thr 3.5
```

It prints `Loading cached …`. Keep `--n-kpts/--n-eigs/--h-elem/--h-fine`
identical to the run that produced the cache — all four are part of the cache
identity, so changing any of them recomputes rather than silently reloading the
wrong sweep (see Gotcha 1).

---

## Combining sweeps in one figure

`dispersion_jaxfem.py` can only ever pair the reference with *one* cloak, so
ideal-vs-optimised needs a separate step. No shell script is required — both
scripts write the same `.npz` schema (`ks`, `fs`, `iprs`), so
`plot_dispersion_overlay.py` reads any set of caches and overlays them.

**Step 1 — compute all three cells into one directory, at one mesh setting:**

```bash
OUT=output/B_multifreq_14x10_nogmm_restart2/dispersion
CFG=output/B_multifreq_14x10_nogmm_restart2/config.yaml
NPZ=output/B_multifreq_14x10_nogmm_restart2/optimized_params.npz
COMMON="--n-kpts 50 --n-eigs 400 --h-elem 0.05 --h-fine 0.025 \
        --f-max 3.5 --ipr-thr 2.5 --workers 8 --force --out-dir $OUT"

# reference + ideal analytic cloak
PYTHONPATH=. python scripts/dispersion/dispersion_jaxfem.py "$CFG" --case both $COMMON

# optimised cloak (same mesh -> same cache tag -> comparable)
PYTHONPATH=. python scripts/dispersion/dispersion_jaxfem.py "$CFG" --case optimized \
    --params-npz "$NPZ" $COMMON
```

Both invocations must share `--h-elem/--h-fine`, since the cache tag — and
therefore comparability — is built from those. You end up with:

```
dispersion_reference_h0.05_hf0.025_k50_e400.npz
dispersion_ideal_cloak_h0.05_hf0.025_k50_e400.npz
dispersion_optimized_cloak_h0.05_hf0.025_k50_e400.npz
```

**Step 2 — overlay.** Each positional argument is `Label=path.npz`, or a bare
path (label inferred from the filename):

```bash
# ideal vs optimised, one solid colour each
PYTHONPATH=. python scripts/dispersion/plot_dispersion_overlay.py \
  Ideal=$OUT/dispersion_ideal_cloak_h0.05_hf0.025_k50_e400.npz \
  Optimised=$OUT/dispersion_optimized_cloak_h0.05_hf0.025_k50_e400.npz \
  --f-max 3.5 --ipr-thr 2.5 \
  --out $OUT/ideal_vs_optimized.png

# all three, coloured by IPR with bulk modes shown
PYTHONPATH=. python scripts/dispersion/plot_dispersion_overlay.py \
  Reference=$OUT/dispersion_reference_h0.05_hf0.025_k50_e400.npz \
  Ideal=$OUT/dispersion_ideal_cloak_h0.05_hf0.025_k50_e400.npz \
  Optimised=$OUT/dispersion_optimized_cloak_h0.05_hf0.025_k50_e400.npz \
  --color-by ipr --bulk --f-max 3.5 \
  --out $OUT/three_way.png
```

It reads only caches, so it is instant and safe to re-run while tuning.

| Argument | Default | Meaning |
|---|---|---|
| `SPEC…` | — | One or more `Label=path.npz` (or bare paths). |
| `--out` | required | Output PNG. |
| `--color-by` | `dataset` | `dataset`: one colour-blind-safe colour each — clearest for 2–3 sweeps. `ipr`: turbo colormap by localisation + colourbar, matching `dispersion_jaxfem.py`. |
| `--ipr-thr` | 2.5 | Surface/bulk split, as above. |
| `--f-max` | 3.5 | `f*` axis limit. |
| `--ipr-cap` | 15 | Clips the IPR colour scale to stop saturation. |
| `--bulk` | off | Also draw sub-threshold bulk modes as faint dots. |
| `--k-edge` | 0.25 | Brillouin-zone edge in `k_norm`, equal to `1/(2·L_c)`. 0.25 is correct for `L_c = 2λ*` **with λ\* = 1 m**, which every config here uses; scale it if `domain.lambda_star` differs. |
| `--no-rayleigh` | off | Omit the folded analytic Rayleigh guide lines. |
| `--title` | none | Axes title. Omit for paper figures whose caption carries the description. |

Marker *shape* always encodes the dataset (so the figure survives greyscale),
and later datasets are drawn slightly smaller so overlapping branches from
earlier ones stay visible — relevant because reference and ideal-cloak branches
sit almost exactly on top of each other.

### Using `dispersion_ideal.py` instead

Only if you want the legacy hardcoded-parameter path. It emits the same schema,
so its output drops straight into the overlay:

```bash
PYTHONPATH=. python scripts/dispersion/dispersion_ideal.py \
  --n-kpts 50 --n-eigs 400 --h-elem 0.05 --h-fine 0.025 \
  --f-max 3.5 --ipr-thr 2.5 --workers 8 --force \
  --out-dir /tmp/disp_legacy --case ideal_cloak
```

Note it writes only the `.npz` for a single case — no PNG — which is all the
overlay needs.

It has no `--case optimized`, no `--params-npz`, and ignores your config — so
if the config's material or geometry ever differs from its hardcoded values, the
comparison is meaningless. Cross-check `rho0`, `cs` and the geometry factors
before trusting an overlay built this way.

It also keeps the **old cache scheme**: its filenames encode only `h_elem` and
`h_fine`, and its `.npz` carries no fingerprint. Gotcha 1 therefore still applies
to it — changing `--n-kpts`/`--n-eigs` silently reloads the previous sweep. Pass
`--force`, or use a fresh `--out-dir`, every time you change either.

---

## Gotchas

**1. Caches are fingerprinted — a stale one is recomputed, not reused.** The
filename now encodes the mesh *and* the sweep size:

```
dispersion_optimized_cloak_h0.05_hf0.025_k50_e400.npz
                           └ h_elem └ h_fine └ n_kpts └ n_eigs
```

and every `.npz` additionally stores a `fingerprint` recording the geometry
(`H`, `L_c`, `a`, `b`, `c`), the material (`rho0`, `cs`), the mesh, the sweep
size, and — for the optimised case — the cell grid plus a SHA of the
`optimized_params.npz` it was built from. On load the fingerprint must match
exactly or the sweep is recomputed, naming what differed:

```
Recomputing dispersion_optimized_cloak_…npz: params_sha: cached=d8219e79… now=85366517…
```

This closes three holes a filename alone cannot: `--H-factor` (changes `H`, not
the name), an edited config, and — the nastiest — pointing at a *different*
optimisation run whose grid and mesh happen to match. Caches written before this
existed are detected (`cache predates fingerprinting`) and recomputed.

`--ipr-thr` and `--f-max` are deliberately **not** fingerprinted: they only
affect rendering, which is what makes the cheap re-plot above possible.

**2. The cell grid comes from the config — but only since it was fixed.** It
used to default to 50×50 with `--n-C-params 2`, and omitting the flags against a
14×10 grid silently mapped every element to the wrong cell: a plausible-looking
wrong figure, no error. That is what made the deleted
`run_dispersion_diagnose.sh` unusable. It is now read from `cells.n_x/n_y/
n_C_params` and cross-checked against `cell_C_flat.shape`, so a mismatch exits
with an error. If the banner says `overridden: …` when you did not intend an
override, check your command.

**3. The `n_C_params=4` "singular stiffness" warning is a false alarm.** flat4 is
orthotropic Cauchy, whose augmented 4×4 Voigt matrix is rank-3 — but the
background isotropic `C_iso` is rank-3 for exactly the same reason (a Cauchy
material produces no stress under infinitesimal rotation). The weak form
contracts `C` with the full displacement gradient, so this reduces to standard
linear elasticity, and the fixed-bottom Dirichlet condition leaves K
positive-definite. Ignore it.

**4. What is *not* a false alarm: `⚠ Clamping …`.** For the eigenvalue problem,
K and M must be positive-definite, so `element_materials_optimized` floors `rho`
at 1% of `rho0` (and `mu` at 1% of background, for `n_C_params=2` only). If those
lines appear, the dispersion is being computed on a *modified* material — the
optimisation produced non-physical cells. For this run they do not appear: all
140 cells are positive-definite, because it was trained with
`optimization.neural.constrained: true`.

---

## Normalisations

- **Frequency:** `f* = f·λ*/c_R`, with λ*=1 m and `c_R = cs·(0.826+1.14ν)/(1+ν)`
  ≈ 266.64 m/s for ν=0.25, cs=300. The Rayleigh branch lies along `f* = k_norm`.
- **Wavenumber:** `k_norm = k/(2π)`. With `L_c = 2λ*` the Brillouin-zone edge is
  at `k_norm = 0.25`.
- Bands fold at the zone edge, so the analytic guide lines are
  `f* = 2m·k_edge + k` and `f* = 2(m+1)·k_edge − k` for m = 0, 1, 2, …

## IPR (inverse participation ratio)

```
IPR = A_total · Σ_n( A_n |u_n|⁴ ) / ( Σ_n A_n |u_n|² )²
```

where `A_n` is the nodal area. It measures spatial localisation: a mode spread
uniformly over the cell has IPR ≈ 1, a mode confined to the free surface has
IPR ≫ 1. Modes above `--ipr-thr` are drawn as large outlined markers (surface /
Rayleigh), below it as small dots (bulk P/S). At the zone edge the fundamental
Rayleigh mode reaches IPR ≈ 2.7 for both the reference and ideal cloak.

## Outputs

| File | Contents |
|---|---|
| `dispersion_<case>_h<h_elem>_hf<h_fine>_k<n_kpts>_e<n_eigs>.npz` | Cached sweep: `ks`, `fs`, `iprs` (one entry per mode per k-point) + `fingerprint` |
| `dispersion_comparison[_optimized].png` | Reference overlaid with the cloak |
| `dispersion_<case>_vs_rayleigh.png` | Single case with the folded Rayleigh lines |
| `ideal_vs_optimized.png` | Whatever `plot_dispersion_overlay.py --out` names |

## Debugging helper

`scripts/dispersion/dispersion_debug.py` runs the reference case under several
configurations to separate numerical from physical effects:

- mesh convergence (h ∈ {0.08, 0.04, 0.02}) — converged by h=0.08
- consistent vs lumped mass — negligible
- `H_factor` 1 vs 2, then a sweep {1.0, 1.5, 2.0, 3.0} — monotonic IPR
  improvement with H
