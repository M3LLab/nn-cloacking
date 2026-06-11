# Choosing the FEM mesh — a practical guide

How to pick a *trustworthy* mesh for this pipeline, with the dimensionless rules,
the failure modes, and the experiments that prove each rule. It covers **two
distinct problems** that need opposite intuitions:

- **Regime A — homogenisation:** compute the effective stiffness `C` of one
  pixel/voxel microstructure. Governed by the **feature/pixel** size.
  Evidence: [`results/mesh_artifact/SUMMARY.md`](../results/mesh_artifact/SUMMARY.md).
- **Regime B — macro wave solve:** solve the cloak elastodynamics on a field of
  per-cell *homogenised* `C`. Governed by the **wavelength**, the macro-cell
  material jumps, and the **positive-definiteness** of the material.
  Evidence: [`results/mesh_tri6_validation/VALIDATION.md`](../results/mesh_tri6_validation/VALIDATION.md),
  [`docs/mesh_2d_benchmark_homogenised.md`](mesh_2d_benchmark_homogenised.md).

> **Decision in one line.** Are you resolving a *microstructure geometry* (pixels)
> or a *wave* in a homogenised medium? Pixels → Regime A. Wave → Regime B.
> Never resolve pixels in the macro wave solve (see the scale-separation check).

---

## 1. Length scales and the dimensionless numbers

| symbol | meaning | canonical value (cell20_phys, f\*=2) |
|---|---|---|
| `λ_star` | reference wavelength (sets the unit) | 1.0 |
| `f_star` | dimensionless frequency | 2.0 |
| `λ_R` | **Rayleigh** wavelength = `2π·c_R/ω` = **`λ_star / f_star`** | **0.50** |
| `δ` | pixel size = (cell physical size) / N_pix | ~2.6e-3 (50 px) |
| `d_cell` | macro-cell size = (cloak extent) / n_cells | 0.04–0.13 |
| `h` | FEM element size | swept |
| `e/px` | **elements per pixel** = `δ/h` | Regime A knob |
| `ppw` | **elements per wavelength** = `λ_R/h` | Regime B knob |
| `S` | **pixels per wavelength** = `λ_R/δ` (scale separation) | **190–600** |

Two master formulas you will actually use:

```
λ_R = λ_star / f_star                     # shorter waves at higher f_star
h_max ≤ λ_R / ppw_target = λ_star / (f_star · ppw_target)
```

`c_R < c_s < c_p`, so the **Rayleigh** branch is the shortest relevant wave and
sets the resolution. Because element count ∝ `(domain/h)²`, **cost grows like
`f_star²`** — doubling the frequency quadruples the mesh.

---

## 2. Regime A — homogenisation (pixels → effective `C`)

QoI: the effective stiffness `C` of a 50×50-pixel microstructure (porous cement,
*isotropic* solid phase). Two independent error sources, both biasing `C` **too
stiff**:

### 2a. Material aliasing — mesh must be pixel-aligned, ≥ 1 e/px
An unstructured mesh that samples the material pointwise and is **coarser than a
pixel** misses the thin solid ligaments. At **0.5 e/px** the effective
`C22/C12/C66` collapse by ~1000× (the load paths are severed); at **1 e/px on a
pixel-aligned structured mesh** the stored dataset `C` is reproduced to 0.0%.
*Rule:* use a **structured mesh whose resolution is a multiple of the pixel grid**
(e.g. N ∈ {50,100,150,…} for a 50-px cell). Non-multiples re-render the geometry
and inject spurious outliers.
*Proof:* `scripts/_mesh_artifact_homog_convergence.py`; SUMMARY Findings 1–2.

### 2b. Element-order over-stiffness — TRI3 ≈ 2× worse than TRI6
Linear TRI3 converges to stiffness **from above**; a ligament one element wide
cannot bend, so it is over-stiff — and worse the **thinner** (lower volume
fraction) the cell. Quadratic **TRI6 ≈ 2× more accurate per linear resolution**
(`TRI6@N ≈ TRI3@2N`). Max rel. error in `C` vs a converged TRI6 reference:

| vol. frac. | TRI3 @1/px (= dataset) | TRI6 @1/px | TRI6 @2/px |
|---|---|---|---|
| 0.24 (thin) | **148 %** | 37 % | 23 % |
| 0.46 (mid)  | 66 %  | 18 % | 12 % |
| 0.84 (bulky)| 21 %  |  8 % |  6 % |

The dataset `C` (TRI3 @1/px) is over-stiff by **+37 % (C11) to +134 % (C22)** for
the thin (vf=0.24) cell; Richardson extrapolation (`C(h)=C∞+a·hᵖ`, p≈1.3) confirms
this is real, not an un-converged reference.
*Proof:* `scripts/_mesh_artifact_richardson.py`; SUMMARY Finding 4 + `tri3_vs_tri6_convergence.png`.

### Regime-A recommendation
- **Element type: TRI6**, always (halves the cost of a given accuracy).
- **Pixel-aligned structured mesh**, resolution a multiple of the pixel grid.
- **`e/px` by volume fraction** (target ≲ 10 %): bulky `vf≳0.8` → **2 e/px**;
  mid `vf≈0.5` → **2–3 e/px**; **thin `vf≲0.25` → ≥ 4 e/px** (these are
  intrinsically hard — stress concentration at necks; budget for them).
- Per-cell cost is seconds–minutes; recomputing all unique cells offline is cheap.

### Do NOT pixel-mesh the macro domain (scale separation)
`S = λ_R/δ ≈ 190–600` here (e.g. 375 px/λ at 20×15). With **hundreds of pixels per
wavelength** the wave only ever sees the *homogenised* medium, so a converged
pixel-level full-domain solve would just reproduce the homogenised macro solve
(to within an `(d_cell/λ_R)² ≈ 1–2 %` correction) — at 16–64 M DOF it is both
**infeasible and redundant**. Homogenise per cell (Regime A), then solve the macro
problem (Regime B). *Proof:* SUMMARY "why brute-force is the wrong tool".

---

## 3. Regime B — macro wave solve (homogenised `C` → cloaking field)

QoI: the cloaking field / surface `transmission_ratio`. Builder:
[`rayleigh_cloak/mesh_uniform.py`](../rayleigh_cloak/mesh_uniform.py)
(`mesh.builder = uniform_tri6`, uniform `h_in` inside the cloak) or the legacy
graded builder (`mesh.builder = legacy`).

### 3a. Resolve the wavelength — `ppw`, and it is set by the FAR FIELD
`h ≤ λ_R/ppw_target`, with `ppw_target = 10–20` for **TRI3 (p=1)** and **`5–6`
for TRI6 (p=2)**. `ppw` is governed by the **coarsest** elements — the far field
(`h_out`, knob `refinement_factor_outside`) — *not* by the cloak refinement.
TRI3 and TRI6 **converge to the same limit** (`ratio_area ≈ 0.980`, agreeing
< 1 % at matched refinement); TRI6 reaches it at fewer ppw but ~4× the nodes at a
given triangulation, so it only pays off when the far field is *coarsened*.
*Proof:* the two-builder overlay, `--builders legacy_tri3,uniform_tri6`:

| rf_cloak | TRI3 `ratio_area` | TRI6 `ratio_area` |
|---|---|---|
| 2 | 0.9748 | 0.9622 |
| 3 | 0.9805 | 0.9743 |
| 4 | **0.9806** (converged) | (see §3c) |

### 3b. The metric is dispersion-robust — you may coarsen the far field
The cloaking metric is a **phase-invariant magnitude ratio** `⟨|u_cloak|⟩/⟨|u_ref|⟩`;
the cloak and reference share the same far-field mesh, so far-field numerical
dispersion (a *phase* error) largely **cancels**. Consequence: for *ranking*
designs you can run the far field well below the textbook `ppw` (the Phase-B sweep
used `ppw_far ≈ 3.4` with TRI6 and still produced a correct, mesh-converged
ranking). Use a fine far field only when you need the *absolute* ratio.
*Proof:* `output/A_single_frequency/` (economical mesh) + §3d convergence below.

### 3c. **Material must be positive-definite** (the subtle, important one)
The formulation contracts `C` with the **full** displacement gradient. A converged,
trustworthy solve requires every cell's `C` to be **positive-definite**:
`C11, C22, C66 > 0` **and** `detN = C11·C22 − C12² > 0`. If a cell is **non-PD**
(loss of strong ellipticity — a *negative-stiffness* inclusion), the operator is
locally **non-coercive / ill-posed**: its unstable mode is *masked by a coarse
mesh* and *resolved with growing amplitude as `h→0`*, so the result **diverges
instead of converging**. This is a **material** pathology, *not* a mesh or
element-order defect — a finer mesh is correctly exposing it.

*Proof (decisive).* An unconstrained-fit material with 6/300 non-PD cells
(`C22` down to −2.5e6, `detN` down to −3e16) gave a sharp localised blow-up at the
worst cell exactly when the mesh was refined to rf=4:

| rf=4 solve (same mesh) | `max\|u\|/p95` | reading |
|---|---|---|
| reference (isotropic) | 4.0 | healthy |
| initial push-forward material | 4.1 | healthy |
| **optimised, non-PD** material | **14.2** @ the non-PD cell | spike → ratio 0.98→0.92 |

while a **PD** material (the Phase-B 10×8 optimum, `detN ≥ 5.6e16`) is flat under
the same refinement — `transmission_ratio = 0.975 ± 0.001` and `max|u|/p95 ≈ 3.3`
across rf_cloak = 2,3,4,5 (33k→78k nodes), **no spike**. Always optimise with
`optimization.neural.constrained: True` (keeps every cell PD).
*Proof:* `results/mesh_tri6_validation/VALIDATION.md`.

### 3d. Resolve the macro-cell material jumps
The material is piecewise-constant per macro cell, with size `d_cell = (cloak
extent)/n_cells`. Put **≥ 2–3 elements across the smallest macro cell**:
`h_in ≤ d_cell/2 → 3`. An element straddling a cell boundary contributes only the
usual `O(h)` unfitted-coefficient error, which vanishes under refinement; the
**uniform** `h_in` builder keeps every cell discretised identically (the legacy
graded field re-weights node density across refinements). Combine with
`ppw`: `h_in ≤ min(λ_R/ppw_target, d_cell/2)`.

### 3e. Gotchas
- **`embed_macro_grid` is incompatible with the uniform builder** — its point
  embedding lands on the uniform nodes → zero-area slivers / duplicate nodes →
  *singular* matrix. The uniform `h_in` resolves each cell without it; the builder
  ignores the flag. (Use it only with the legacy graded builder.)
- **Plotting / fixed-grid metrics need 3-node connectivity.** TRI6 cells are
  6-node; the benchmark splits each TRI6 into 4 corner+midside sub-triangles
  (`_tri3_view`) and `plot.py` corner-slices — already handled, but reuse those
  helpers in new analysis code.
- Keep `refinement_factor_outside ≤ refinement_factor_cloak`, else the surface
  threshold inverts (`SizeMax < SizeMin`) and the cell count explodes.

---

## 4. How to KNOW you are converged (verification protocol)

1. **Fix the material, refine the mesh** (a `rf_cloak` sweep) and watch a
   **mesh-independent** QoI — `ratio_area` / `transmission_ratio`. Require the
   last refinement to change by **≲ 1–3 %**.
2. **Watch `max|u|/p95`.** If it *grows* with refinement, your **material is
   non-PD** (red flag on the design, §3c) — do not "fix" it with a coarser mesh.
3. **Check `ppw` ≥ target** in the region of interest (raise
   `refinement_factor_outside` if the far field is marginal *and* you need the
   absolute ratio).
4. Tool: `scripts/mesh_2d_benchmark_homogenised.py <cfg> <params.npz>
   --builders legacy_tri3,uniform_tri6 --cloak 2,3,4 --outside 1` → CSV +
   per-builder figures + a TRI3-vs-TRI6 overlay (`<stem>_builders.png`).

---

## 5. Quick reference

| you are computing | regime | element | alignment / size rule | watch out for |
|---|---|---|---|---|
| effective `C` of a pixel microstructure | A | **TRI6** | pixel-aligned, `e/px` = 2 (bulky) → ≥4 (thin) | aliasing if < 1 e/px or unaligned; TRI3 over-stiff |
| cloak wave field / ratio, **absolute** | B | TRI3 or TRI6 | `h ≤ λ_R/ppw`, ppw 10–20 (p1) / 5–6 (p2), **far field binds** | non-PD material → divergence; resolve cell jumps |
| cloak design **ranking** (e.g. cell sweep) | B | TRI6 | coarse far field OK (metric is dispersion-robust); `h_in ≤ d_cell/2` | keep material PD (`constrained: True`) |
| *pixels in the macro domain* | — | — | **don't** — `S = λ_R/δ ≫ 1` ⇒ homogenise instead | infeasible + redundant |

**Worked sizing (cell20_phys, f\*=2 → λ_R=0.5):** TRI6 macro solve → `h_out ≤
0.5/5 = 0.10` (far field), `h_in ≤ min(0.10, d_cell/2)`; at 20×15 cells
`d_cell≈0.066` ⇒ `h_in ≤ 0.033`. At f\*=4 every `h` halves and the mesh ~4×.

## 6. Tools & evidence
- Builders: `rayleigh_cloak/mesh_uniform.py` (uniform/TRI6), `rayleigh_cloak/mesh.py` (legacy graded); selected by `mesh.builder`.
- Macro convergence: `scripts/mesh_2d_benchmark_homogenised.py` (+ `docs/mesh_2d_benchmark_homogenised.md`).
- Homogenisation convergence: `scripts/_mesh_artifact_homog_convergence.py`, `scripts/_mesh_artifact_richardson.py`.
- Reports: `results/mesh_artifact/SUMMARY.md` (Regime A), `results/mesh_tri6_validation/VALIDATION.md` (Regime B + the PD diagnosis), `output/A_single_frequency/RECOMMENDATION.md` (a worked Regime-B ranking).
