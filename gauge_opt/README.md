# `gauge_opt` — gauge-optimizing the cloak target into the D2 reachable set

Exploit the gauge freedom of transformation elasticity to move the *target*
material onto what D2-symmetric unit-cell blocks can actually build — at zero
cost in ideal cloaking performance, because the exterior scattering is exactly
invariant along the gauge orbit.

Formulas live in **[`theory.md`](theory.md)**. This file is the pipeline.

Does **not** use the Nassar polar lattice (`rayleigh_cloak/nassar.py`); the
target class here is rotated-orthotropic Cauchy, which is what a D2 block gives.
The surrogate is not used either — every number below comes from closed-form
tensor algebra or the already-fitted GMM.

```bash
PY=/root/miniconda3/envs/jax-fem-env/bin/python
$PY -m gauge_opt.test_gauge        # 10/10 verification identities
$PY -m gauge_opt.sweep_gauge       # stage 1 — the trade-off table
$PY -m gauge_opt.check_reachable   # stage 2 — feasibility vs the D2 catalogue
```

---

## What the gauge buys, in one table

Sweeping `A(s) = (F^T)^s` from the identity gauge (`s=0`) to the Cauchy gauge
(`s=1`), on the triangular cloak's actual `F`:

| `s` | minor-sym violation | density anisotropy `log(ρ₁/ρ₂)` | orthotropy residual |
|---|---|---|---|
| 0.0 | **0.340** | 0.000 | 5.3e-3 |
| 0.5 | 0.144 | 0.729 | 9.2e-4 |
| 0.9 | 0.025 | 1.302 | 2.9e-5 |
| 1.0 | **2.4e-17** | 1.443 | 5.4e-17 |

Three results, all verified in `test_gauge.py`:

1. **A minor-symmetric gauge exists.** `A = c·Fᵀ` makes the stiffness *exactly*
   Cauchy — violation at machine epsilon, not approximately. No symmetrization
   projection needed, unlike `materials.symmetrize_stiffness`, which discards a
   34 % residual.
2. **It is unique** up to the scalar `c` (theory.md §4.0b). Demanding exact minor
   symmetry spends the entire gauge freedom.
3. **It costs density anisotropy**, ratio 4.23, and nothing else on this geometry —
   because the triangular map is piecewise affine (`geometry/triangular.py:53`),
   so `∇A = 0` and the Willis coupling is **identically zero**, not merely small.

Bonus: the Cauchy-gauge target is *exactly orthotropic* in the eigenframe of
`B = FFᵀ` (C16 = C26 = 0 to 5e-17), with closed-form moduli — i.e. it lands
exactly in the D2 class by construction, no fitting involved.

## The catch, measured

Exact minor symmetry pins two modulus ratios to host-determined constants that
no gauge can move (theory.md §4.3). Against the fitted reachable set
(`dataset/gmm/gmm_flat4_squared_2m.npz`, 906k cells):

| cumulative condition | share of reachable set |
|---|---|
| anisotropy ≈ 17.9 | 1.21 % |
| + `C12/C66 = 1` | 0.079 % |
| + `C11·C22/(C12·C66) = 9` | **0.005 %** (28 of 520k) |

Non-empty but deep in a smoothed GMM tail. And the catalogue's cells run at
1000–2560 m/s against a 300 m/s substrate — it was homogenized for a base
material 5–8× too stiff.

**So: don't target `s = 1`.** Scoring the family against the GMM already shows the
best margin at **s ≈ 0.9**, an interior optimum — which is the whole premise of
optimizing over the gauge rather than picking a canonical one.

---

## Pipeline

### Stage 0 — renormalize the catalogue *(not started; blocking)*
The base-material mismatch above dominates every other term. Re-express the GMM
in dimensionless form (`C_eff/C_base`, `ρ_eff/ρ_base`) so cell *geometry* is
separated from base-material choice, then re-attach the substrate's
`(λ₀, μ₀, ρ₀)`. Until this is done, `check_reachable`'s absolute `log p` values
are not meaningful — only the *shape* of the curve across `s` is.

### Stage 1 — gauge family and trade-off  ✅ `sweep_gauge.py`
`gauge.py` implements `C_gauge`, `rho_gauge`, `willis_S/W` and the family
`A(s,c) = c·(Fᵀ)^s`. Verified against `materials.C_eff` at `s=0`.

### Stage 2 — reachability scoring  ✅ `check_reachable.py`
`reachable.ReachableGMM` wraps the fitted 16-component GMM and exposes
`log_prob` / `reachable` / `margin`. `margin` is smooth in the moduli, so it is
directly usable as the objective term of theory.md eq (6.4) — no hard box.
`fit_D2` extracts `(θ, C11, C12, C22, C66)` plus both discarded residuals.

### Stage 3 — optimize the gauge fields *(next)*
Promote `s → s(x)`, `c → c(x)` on the cell grid and minimize eq (6.5). Needs:
- a JAX-traceable 2×2 matrix power to replace `gauge.gauge_power`'s
  `scipy.expm/logm` — closed form is available since `spec(F) = {1, (b−a)/b}`;
- gauge BCs `s = 0`, `c = 1` on Γ_out (theory.md §3.1), which a piecewise-constant
  gauge violates. A transition layer of thickness ℓ carries `S = O(1/ℓ)`; its
  width is itself a design variable.

### Stage 4 — joint optimization over the map χ *(next)*
Because §4.0b caps what the gauge alone can reach, the map is the other lever.
Relax the triangular map's affine parameters; note that a non-affine map
reintroduces `∇F ≠ 0` and hence Willis coupling that D2 cells **cannot**
represent (centrosymmetry ⇒ odd-rank tensors vanish), so penalize `‖∇F‖` hard.

### Stage 5 — inverse-design the cells *(deferred)*
Feed the per-cell `(θ, C11, C12, C22, C66)` to the existing D2 generator. The
conditioning variables match exactly — see the four
`microstructure_generation_2d/scaler_*` files. Deferred until the surrogate is
better trained, as requested.

### Stage 6 — FEM confirmation *(deferred)*
Re-run `run.py` on the gauge-optimized target. The gauge is scattering-invariant
only in the continuum limit; discretization error is what makes stage 3
meaningful, so this closes the loop.

**FEM caveat:** the augmented 4×4 Voigt matrix is singular for *every* gauge
(theory.md §4.0a — Sylvester's law of inertia; the isotropic host has no
rotational stiffness and no gauge can invent one). The full-gradient Cosserat
formulation must therefore not be used near `s = 1`; switch to the
symmetric-strain path (`n_C_params=6cauchy`). `objectives.positive_definite`
checks the 3×3 block for this reason; `objectives.couple_stiffness` reports the
4×4 null eigenvalue as a diagnostic.

---

## Files

| file | role |
|---|---|
| `theory.md` | derivation and all formulas |
| `gauge.py` | `C_gauge`, `rho_gauge`, `willis_S/W`, gauge families |
| `reachable.py` | D2 class, `fit_D2`, `ReachableGMM`, closed-form eq (4.5) |
| `objectives.py` | asymmetry, density anisotropy, Willis, D2 defect, stability |
| `sweep_gauge.py` | stage 1 |
| `check_reachable.py` | stage 2 |
| `test_gauge.py` | 10 verification identities (theory.md §7) |

`__init__.py` enables JAX x64 — in float32 the exactness claims floor at ~1e-7
and are indistinguishable from modelling error.
