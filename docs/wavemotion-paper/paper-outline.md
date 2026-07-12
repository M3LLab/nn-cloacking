# Neural-field design of broadband Rayleigh-wave carpet cloaks under microstructure realisability constraints

*Structural extraction of `elsarticle-template-num.tex` — sections, subsections, and a paragraph-level summary of each. Wave Motion submission.*

---

## Abstract

- Transformation elasticity gives **exact** material distributions for elastodynamic cloaks, but the required stiffness tensors violate the **minor symmetries** of classical Cauchy elasticity → hard to realize.
- Existing approaches restore symmetry by symmetrizing the transformed tensor → only **approximate** cloak.
- **This work:** formulate 2D Rayleigh-wave carpet-cloak design as a **PDE-constrained optimization** — coordinate neural network + differentiable FEM optimizing symmetric stiffness and density fields by minimizing wave-field distortion.
- Both **single-frequency** and **broadband** (multi-frequency) optimization.
- Physical realizability via a **database of homogenized microstructures** (conditional diffusion, autoencoder latent optimization, nearest-neighbour selection).
- Result: unconstrained design ≈ ideal transformation cloaking; best realizable design retains ~X% of that performance.

**Keywords:** elastic cloaking · Rayleigh waves · transformation elasticity · neural field · implicit neural representation · metamaterials

---

## 1. Introduction  `sec:intro`

- **¶1 — Transformation elasticity.** Coordinate map steers elastic waves around an object; mechanical counterpart of transformation optics. The **"carpet" variant** (hides a surface defect) is the most experimentally accessible.
- **¶2 — BGM push-forward.** Coating material follows Brun–Guenneau–Movchan relations. Resulting tensor keeps **major** symmetry but breaks **minor** symmetries → ideal cloak is a **polar (Cosserat)** medium, outside ordinary Cauchy materials. Turning this non-realisable prescription into a buildable cloak is the problem.
- **¶3 — Three realisation families for Rayleigh waves.** (i) Symmetrise into a Cauchy material (approximate cloaking); (ii) purpose-built polar/Cosserat lattice (Nassar 2018); (iii) select orthotropic Cauchy cells from a database (Wang 2022, quasi-static). Chatzopoulos 2023 compared triangular vs semi-circular carpets via arithmetic-mean symmetrisation.
- **¶4 — Why triangular + our approach.** Triangular transformation is a constant-gradient shear → homogeneous, non-singular, frequency-independent ideal tensor. **Contribution:** differentiable JAX-FEM pipeline; first search for the single best realisable orthotropic Cauchy moduli `(C11,C12,C22,C66)`; then a **coordinate neural field** assigning each cell its own moduli+density; optimise single-frequency then broadband. Neural field = implicit material representation (NeRF/SIREN/Fourier-feature lineage); relates to neural reparameterisation in topology optimisation, but here outputs **anisotropic stiffness + density** against an elastodynamic objective.
- **¶5 — Explicit realisation.** Fill each cell with a two-phase (solid/void) microstructure from a homogenised database (Yang 2024), all sharing **one base material**, differing only in geometry. Claims first Rayleigh-wave elastodynamic cloak realised this way from a common material; reports gains + fraction of ideal performance surviving homogenisation.
- **¶6 — Paper organisation.** §2 setup, §3 neural-field pipeline + single-freq, §4 broadband, §5 microstructure constraint, §6 discussion.

---

## 2. Problem setup and FEM baseline  `sec:setup`

*Fig. 1 (`fig:setup`): computational domain + cloak geometry inset.*

### 2.1 Governing equations  `subsec:governing`
- Isotropic half-space `(λ,μ,ρ)`, Navier elastodynamics, plane strain, isotropic Voigt tensor `C_IJ`.
- Pointwise invertible transformation `Ξ`; transformation gradient `F`, `J=det F`.
- Navier equation not form-invariant → gauge choice. `A=F` = Willis setting (symmetric stress, needs coupling tensors + tensorial density, narrow-band). **They adopt Cosserat setting `A=I`** (Norris 2011, Brun 2009) → Navier form retained with BGM parameters.
- **BGM push-forward** `c_eff = J⁻¹ C F F`, `ρ_eff = ρ J⁻¹`. Preserves major symmetry, breaks minor symmetries → non-symmetric polar medium; stress acts on full displacement gradient. Frequency-domain convention, `u_tt → −ω²u`.

### 2.2 Transformation elasticity for the triangular carpet cloak
- Pinched-carpet construction (Chatzopoulos 2023): inner/outer triangles `z1,z2`; shear map compresses region above `z2` onto cloak region, opening the notch.
- Piecewise-constant gradient `F` (`F21=sign(X1)·a/c`, `F22=(b−a)/b=J`).
- Effective stiffness `c_eff` (4×4 augmented Voigt), constant density `ρ_eff = ρ b/(b−a)`.
- Tensor is **non-singular** and uniform within each half → ideal cloak = **two mirror-image phases**. Three minor-symmetry pairs broken independently — exactly what symmetrisation averages away.

### 2.3 Cell discretisation
- Regular Cartesian grid `nx×ny` square cells; cells between inner/outer triangles get piecewise-constant stiffness + density. Affine transform → no rotation to local frame.
- Ideal tensor = full 4×4 polar, **10 independent components**. Real materials are Cauchy → per-cell design variable = **4 free orthotropic Cauchy moduli `(C11,C12,C22,C66)` + scalar density**, searched by the differentiable optimiser.
- **14×10 grid** chosen as optimal resolution for all single-frequency experiments. *(Fig. 2 `fig:cartesian_cells`.)*

### 2.4 Simulation domain and FEM solver  `subsec:domain`
- Rectangular domain `W=12.5b`, `H=4.305b`; traction-free top, PML on laterals + bottom. Notch `a=0.0774H`, `c=0.1545H`, cloak `b=3a`. Background `ρ=1600`, `c_s=300`, `ν=1/4`, plane strain.
- Time-harmonic vertical point force → Rayleigh waves. Dimensionless frequency `f* = f b/c_R`; **design frequency f*=2**.
- Frequency-domain, complex DOFs, Rayleigh-damping PML, **JAX-FEM** (Xu 2020), triangular mesh ~246k elements / 124k nodes. **Fully differentiable via implicit adjoint** through the linear solve.

### 2.5 Cloaking metrics
- **`D_r`** — right-boundary (transmission-side) distortion, relative L² vs defect-free reference.
- **`D_out`** — outside-cloak distortion (global wavefield perturbation; related to Wang 2022 objective).
- **Cloak ratio `η`** — `⟨|u|⟩/⟨|u_ref|⟩` on free surface beyond cloak (Chatzopoulos 2023); objective used in microstructure pipeline.

*Fig. 3 (`fig:nf-arch`): neural-field pipeline schematic.*

---

## 3. Neural-field design pipeline  `sec:method`

### 3.1 Coordinate-conditioned material network
- MLP `Φ_θ: (x,y) → (C_4CM, ρ)`. 4-layer, 256 hidden units (~200k weights), `tanh`, **Fourier positional encoding** (Tancik 2020) for sharp material transitions. Evaluated at cell centres → piecewise-constant → expanded to quadrature points. Gradients by backprop through JAX-FEM adjoint.

### 3.2 Transformation-elasticity prior
- Network doesn't learn from scratch: moduli initialised at a starting point, then **freely optimised** (only explicit drift penalty constrains). Tried two starting points (arithmetic-mean symmetrised ideal tensor; dataset mean) → **no significant difference** → design driven by objective, not init.
- Last layer scaled ~0; output applied as **multiplicative residual** (scale ε=0.1) → spatially uniform relative perturbation across differing magnitudes. Additive variant implemented but worse-conditioned.

### 3.3 Loss function
- `L = L_cloak + λ_l2 · L_l2`. Cloak term = squared right-boundary distortion; `L_l2` = normalised drift penalty toward TE init. No neighbour-smoothness term needed (neural field is spatially continuous by construction).

### 3.4 Comparison against the ideal cloak
- Compares raw per-cell baseline, neural-field reparam, and continuous `c_eff` reference (all 14×10). *(Table 1 `tab:opt_results` — several **[FILL]** values; Fig. 5 `fig:convergence_comparison`.)*
- **Note:** many placeholder `[FILL]` / `TODO confirm` entries remain in this section (init `D_r`, per-cell + neural-field results).

### 3.5 Mesh-refinement convergence  `subsec:mesh_conv`
- Continuous cloak on meshes `N_e ∈ {6,12,25,50,100,200}k`. Distortions vs `N_e^(−1/2)`. Linear fit extrapolates to `D_r,∞ ≈ 19±5%`, `D_out,∞ ≈ 0.75%`. → Residual right-boundary distortion is a **finite-mesh artefact** of corner singularities, not an optimisation artefact. *(Fig. 6 `fig:mesh_conv`.)*

### 3.6 Wavefield and material visualisations  `subsec:wavefield`
- Three diagnostics: `Re[u]` field overlay (Fig. 7 `fig:reu_comparison`); magnitude `‖u‖` invariant to phase, exposes residual scattering (Fig. 8 `fig:unorm`); per-cell material heatmap `(C11,C12,C22,C66,ρ)` (Fig. 9 `fig:matheatmap`).

### 3.7 Direct search over realisable Cauchy moduli  `subsec:flat4`
- Cloak parameterised by **4 free orthotropic Cauchy moduli (4CM) + density**. Justified by microstructures: square two-phase cells with `D2` point-group symmetry → normal-shear couplings vanish (`C16=C26=0`) → block-diagonal orthotropic form.
- Keep weaker `D2` (not `D4`) so `C11≠C22` can match ideal cloak anisotropy. 4CM is a **deliberately minimal, narrower** class than the 6-component symmetrised tensor. Full 10-component polar tensor (**10-PT**) retained only as ideal reference (not a design parameterisation). *(Table 2 `tab:flat4` — **[FILL]**; Fig. 10 `fig:flat4_vs_flat10`.)*

### 3.8 Symmetrised-tensor reference  `subsec:symmetrised`
- Arithmetic-mean symmetrisation (Chatzopoulos 2023) = closed-form Cauchy reference. It's a **6-component anisotropic** Cauchy material vs their **4-component orthotropic** 4CM — different materials. Their 4CM (independent numerical search) can match/improve on the closed-form point within the realisable class. *(Fig. 11 `fig:nassar_repro`.)*

---

## Single frequency - NEW

### §4.1 — Single material beats symmetrisation (A_symmetrized, A_1x1flat4_vs_symmetrized)
A single searched orthotropic material (1×1) reaches η=0.85, far better than Chatzopoulos's closed-form symmetrisation (η=0.63) — "much better cloaking, but not perfect" (vs ideal η=0.999). Table 1 + frequency-sweep figure.

### §4.2 — Four materials reach 95% cloaking (A_partition_pushed)
A 2×2 decomposition = 4 materials → η=0.962 (within 4% of perfect). Presented as the requested table (1×1…3×2), plus the 1-vs-4 field comparison.

### §4.3 — Grid selection by cell sweep (A_cell_sweep)
Sweep of 5×4…32×24 shows η plateaus at ~0.99 by 14×10; finer grids give no gain — "converging, stable, fast." I explicitly highlight that the 14×10 grid = 140 cells but only 46 lie in the cloak region and carry design variables (bolded in text, a dedicated "cloak cells" table column, and in the histogram's in-cloak counts).

### §4.4 — Near-perfect cloaking at 14×10 (A_single_freq_14x10)
The 46-material design reaches η=0.990, shown reconstructing the reference and matching the ideal continuous polar cloak in the 4-panel field comparison, plus the optimised per-cell material tensor.



## 4. Broadband multifrequency optimisation  `sec:multifreq`

### 4.1 Single-frequency overfitting
- Single-freq cloak overfits: sharp loss minimum at `f*`, degrades away. Runs at `f*=2.5`, `3.0` nearly disjoint outside `Δf*≈0.2`. Analogous to narrow-band topology-optimised acoustic cloaks → motivates multifrequency objective. *(Fig. 12 `fig:freq_overfitting`; grid noted as 50×50 with a TODO to confirm.)*

### 4.2 Reference sweeps  `subsec:ref_sweeps`
- Two reference sweeps on 50×50 cells: ideal continuous `c_eff` (lower envelope) and uncoated obstacle (upper envelope). Bound the optimisation target. *(Fig. 13 `fig:reference_sweeps`.)*

### 4.3 Min–max/mean objective
- `L_band = α·max_i L(f_i) + (1−α)·mean_i L(f_i)`, `α` annealed 1→0. Pure min-max prevents trading frequencies; relaxing sharpens final solution. Flat low loss over `f*∈[1,4]`. *(Fig. 14 `fig:multifreq_sweep`.)*

### 4.4 Cloaking ratio across the band  `subsec:ratio_chart`
- Cloak ratio `η(f)` reported across band. Optimised cloak holds `η≈1`; obstacle far from unity; ideal cloak near-but-not-1 (mesh-convergence floor). Same ratio = objective in §5. *(Fig. 15 `fig:u_ratio`.)*

### 4.5 Floquet–Bloch dispersion  `subsec:bloch`
- Bloch dispersion of a vertical strip cut through the optimised cloak (Bloch-periodic along surface, traction-free top, fixed bottom). Dispersion **real-valued** across band → supports propagating surface modes, **no reliance on locally resonant bandgaps**. *(Fig. 16 `fig:bloch`.)*

### 4.6 Results
- Neural-field reparam gives flat, low response; band-averaged metric **0.985**. Sharp designs with fine spatial features (high-order Fourier). Bloch confirms expected propagation regime at band centre. *(Fig. 17 `fig:sweep-compare`; Fig. 18 `fig:mat-compare`.)*

---

## 5. Microstructure-constrained optimisation  `sec:micro`

- Motivation: §4 cloaks are tensor-valued (no guaranteed realisable microstructure). Constrain neural field to a **manifold of realisable two-phase microstructures** (Yang 2024 strategy).

### 5.1 Dataset and homogenisation
- `N ~ 10⁶` binary 50×50 unit cells via squared-assembly **cellular automaton** on cement/void substrate. Periodic-FEM homogenisation → `(λ_eff, μ_eff)`, `ρ_eff`, full `C_eff`. Follows Yang 2024 in philosophy but specialised to elastodynamic cloak target.

### 5.2 Dataset distribution  `subsec:dataset_dist`
- Marginal/pairwise distributions of `(λ,μ,ρ)` with single-freq optimum trajectory overlaid. Dataset covers a band-limited region; cloak-relevant analytical `c_eff` locus **partially exits** the envelope → anticipates projection error. *(Fig. 19 `fig:dataset_dist`.)*

### 5.3 GMM prior and constrained loss
- Gaussian mixture fit to standardised `(λ,μ,ρ)`; flat-top log-density threshold `τ` defines realisable manifold `M`. Dataset is intrinsically Cauchy → **4CM parameterisation**, summarised by per-cell isotropic descriptor.
- **Manifold penalty** `L_GMM = Σ_c max(0, τ − log p_GMM(...))` pulls cells into `M` without forcing a specific entry. After optimisation, **snap** each cell to nearest dataset entry (Euclidean in standardised `λ,μ,ρ`); tile microstructures for pixel-level validation on refined mesh. *(Fig. 20 `fig:micro_pipeline`.)*

### 5.4 Diagnostic experiment  `subsec:micro_results`
- Separates **matching error** (projection onto discrete dataset) from **pixel-vs-homogenised error** via an intermediate stage (matched `λ,μ,ρ` but still homogenised FEM). Cloak ratio at `f*=2` on 20×15 macro grid, 100 cloak cells. *(Table 3 `tab:micro_results`: optimised ≈1.00 → matched 0.94–0.95 → pixel-level ≈0.70.)*
  - **~5% matching gap is benign** — price of discrete projection; improvable via bigger dataset, reweighted distance (so μ isn't ignored), or generative match.
  - **~25% pixel-vs-homogenised gap is the real problem** — collapses to ~0.70, mesh refinement doesn't close it. Likely cause: **hidden anisotropy** (isotropic `λ,μ` projection of an orthotropic stiffness). Secondary: Willis dynamic-homogenisation, pixel-level interface scattering. Proposes disambiguating experiment reinserting full anisotropic `C`.

### 5.5 Displacement-field validation  `subsec:disp_validation`
- Matched homogenised vs pixel-level fields. Transmission-side amplitude well reconstructed by homogenised solve; pixel-level carries higher-order interface features the `(λ,μ,ρ)` descriptor can't encode → visual correlate of the ~25% gap. *(Fig. 21 `fig:disp_validation`.)*

### 5.6 Microstructure gallery  `subsec:micro_gallery`
- Representative matched cement/void microstructures tiled across cloak; outline colour encodes host cell position → gradient from sheared low-density (near defect) to near-bulk (outer boundary). *(Fig. 22 `fig:micro_gallery`.)*

---

## 6. Broadband microstructure-constrained design  `sec:micro_band`

- Combines GMM manifold penalty (§5) with multifrequency objective (§4). Keeps 4CM. Trains single neural field against `L_band+micro = L_band + λ_GMM·L_GMM + λ_l2·L_l2`. Manifold penalty annealed from small value.
- **Snap, validate, sweep:** snap to nearest dataset entry, frequency sweep of matched homogenised cloak. Broadband micro-constrained cloak stays roughly flat (GMM doesn't collapse to single-freq optimum) but carries the same residual amplitude penalty from §5.4. *(Fig. 23 `fig:micro_band_sweep`.)*

---

## 7. Discussion  `sec:discussion`

- **Shared architecture** (coordinate neural field + differentiable FEM) beats hand-crafted symmetrisation in three regimes: single-freq (direct 4CM search), multifreq (implicit smoothness + band objective flattens loss), microstructure-constrained (projection exposes a hidden homogenisation gap).
- **Comparison with prior work.** Wang 2022 (elasto-static): 17.2% → 4% (circular void, orthotropic database). Here: continuous `c_eff` reduces outside distortion ~5% → extrapolated ~0.75%; optimised 14×10 neural field ~1.7%. Not apples-to-apples (elastodynamic polar/orthotropic-Cauchy vs quasi-static Cauchy) but order-of-magnitude encouraging.
- **Limitations.** Piecewise-constant square cells can't conform to inclined interfaces (residual `D_r` dominated by FEM error); microstructure pipeline limited to binary substrate + isotropic `(λ,μ,ρ)` descriptor.
- **Future work.** (i) Learn a **generative model** of the realisable manifold (guided diffusion, Yang 2024) instead of discrete snapping; (ii) incorporate **dynamic/Willis homogenisation** to close the gap; (iii) extend to **3D**; (iv) design beyond orthotropic Cauchy via **Cosserat/polar microstructures** to realise the broken minor symmetry directly.

---

## 8. Conclusion  `sec:conclusion`

- Unified differentiable pipeline for 2D Rayleigh-wave carpet cloaks: per-cell material = **4 free orthotropic Cauchy moduli + density**, searched by a coordinate neural field through end-to-end differentiable JAX-FEM.
- An alternative to closed-form symmetrisation that directly explores the realisable Cauchy space; ideal 10-component polar tensor kept only as mathematical reference.
- Recovers near-perfect single-frequency cloaking, flattens loss across a band (min–max/mean), and — when projected onto a realisability manifold of homogenised binary microstructures — **exposes a hidden-anisotropy gap not previously isolated in the elastodynamic regime**.

---

## Front-matter TODOs still open in the source
- Title-page author/affiliation blocks empty.
- Abstract: best realizable design "retains ~**X%**" unfilled.
- §3.4 Table 1 & Fig. 5: multiple `[FILL]` (init `D_r`, per-cell + neural-field results).
- §3.7 Table 2: 4CM `D_r`, `D_out` `[FILL]`.
- §4.1–4.4 grid noted as 50×50 with repeated `TODO confirm grid (14×10?)`.
- §3.2 init source and §4.6 band-averaged "0.985" flagged "confirm 4CM".
- Leading "Paper Plan" enumerate block is a to-remove-before-submission scaffold.
