# Gauge freedom in transformation elasticity — formulas

Companion to `README.md` (pipeline). This file holds the derivation and every
formula the code implements. Notation: capital indices `I,J,K,L` live in the
**virtual** (undeformed) domain, lowercase `i,j,k,l` in the **physical**
(cloak) domain. Summation convention throughout.

---

## 1. Virtual problem

Time-harmonic elastodynamics in the virtual domain, isotropic host
$(\lambda_0,\mu_0,\rho_0)$:

$$
\frac{\partial}{\partial X_J}\!\left(C^0_{IJKL}\,\frac{\partial u_K}{\partial X_L}\right) + \rho_0\,\omega^2 u_I = 0,
\qquad
C^0_{IJKL} = \lambda_0\,\delta_{IJ}\delta_{KL} + \mu_0(\delta_{IK}\delta_{JL} + \delta_{IL}\delta_{JK}).
$$

## 2. Map

$x = \chi(X)$, deformation gradient and Jacobian

$$
F_{iI} = \frac{\partial x_i}{\partial X_I}, \qquad J = \det F .
$$

Chain rule $\partial/\partial X_J = F_{jJ}\,\partial/\partial x_j$, and the **Piola identity**
for any field $V_J$:

$$
\frac{\partial V_J}{\partial X_J} \;=\; J\,\frac{\partial}{\partial x_j}\!\left(J^{-1}F_{jJ}V_J\right).
$$

Applying it with $V_J = C^0_{IJKL}\,F_{lL}\,\partial u_K/\partial x_l$ and dividing by $J$:

$$
\boxed{\;
\frac{\partial}{\partial x_j}\!\left(\mathcal{C}_{IjKl}\,\frac{\partial u_K}{\partial x_l}\right) + J^{-1}\rho_0\,\omega^2 u_I = 0,
\qquad
\mathcal{C}_{IjKl} := J^{-1} F_{jJ}\,F_{lL}\,C^0_{IJKL}. \;}
\tag{2.1}
$$

$\mathcal{C}$ has **major** symmetry ($\mathcal{C}_{IjKl} = \mathcal{C}_{KlIj}$) but its
first slot is a *virtual* index while its second is *physical* — that mismatch is
the entire source of the trouble.

### 2.1 Identity gauge (what the repo currently does)

Relabel $I\to i$, $K\to k$. Then $(2.1)$ *is* an elastodynamic equation with

$$
C'_{ijkl} = J^{-1} F_{jJ}F_{lL}\,C^0_{iJkL}, \qquad \rho' = \rho_0/J \;\;(\text{scalar}).
$$

This is exactly `rayleigh_cloak/materials.py::C_eff`
(`einsum("jJ,lL,iJkL->ijkl", F, F, C0) / J`) and `rho_eff`. It is the
Brun–Guenneau–Movchan / Cosserat realization: major-symmetric, **not**
minor-symmetric, so the stress is non-symmetric and the medium must transmit
couples.

Expanded for isotropic $C^0$:

$$
J\,C'_{ijkl} = \lambda_0\,F_{ji}F_{lk} \;+\; \mu_0\,\delta_{ik}\,(FF^{T})_{jl} \;+\; \mu_0\,F_{li}F_{jk}.
\tag{2.2}
$$

---

## 3. The gauge

Introduce an invertible matrix field $\mathbf{A}(x)\in GL^{+}(2)$ and redefine the
displacement:

$$
u_I(X) \;=\; A_{Ik}(x)\,\tilde{u}_k(x).
\tag{3.1}
$$

Substitute into $(2.1)$ and **left-multiply by $A_{Ii}$** (i.e. by $\mathbf{A}^{T}$).
That choice is not arbitrary: it is the unique multiplier that keeps the
resulting operator self-adjoint, i.e. that preserves reciprocity and gives a
real energy functional. Using
$A_{Ii}\partial_j(\Xi_{Ij}) = \partial_j(A_{Ii}\Xi_{Ij}) - (\partial_j A_{Ii})\Xi_{Ij}$:

$$
\frac{\partial}{\partial x_j}\!\left(C^{\mathcal A}_{ijkl}\frac{\partial \tilde u_k}{\partial x_l} + S_{ijk}\,\tilde u_k\right)
\;-\; T_{ikl}\frac{\partial \tilde u_k}{\partial x_l}
\;-\; W_{ik}\,\tilde u_k
\;+\; \omega^2 \rho^{\mathcal A}_{ik}\,\tilde u_k \;=\; 0
\tag{3.2}
$$

with

| symbol | definition | meaning |
|---|---|---|
| $C^{\mathcal A}_{ijkl}$ | $A_{Ii}\,\mathcal{C}_{IjKl}\,A_{Kk} \;=\; J^{-1}A_{Ii}F_{jJ}\,C^0_{IJKL}\,F_{lL}A_{Kk}$ | stiffness (major-symmetric always) |
| $\rho^{\mathcal A}_{ik}$ | $J^{-1}\rho_0\,(A^{T}A)_{ik}$ | density — **tensorial**, symmetric PD |
| $S_{ijk}$ | $A_{Ii}\,\mathcal{C}_{IjKl}\,\partial_l A_{Kk}$ | Willis-type coupling, $\propto \nabla A$ |
| $T_{ikl}$ | $(\partial_j A_{Ii})\,\mathcal{C}_{IjKl}\,A_{Kk}$ | adjoint of $S$ (major symmetry $\Rightarrow$ self-adjoint pair) |
| $W_{ik}$ | $(\partial_j A_{Ii})\,\mathcal{C}_{IjKl}\,(\partial_l A_{Kk})$ | zeroth-order correction, $\propto \nabla A \otimes \nabla A$ |

**Every $\mathbf{A}$ gives the same exterior scattering.** In 2D that is 4 free
scalar fields.

### 3.1 Boundary conditions on the gauge — what makes it a symmetry

Outside the cloak $\mathbf{A}=\mathbf{I}$ and $\tilde u = u$. For the physical
displacement to be continuous across the outer cloak boundary $\Gamma_{\rm out}$
(as it must be in a bonded solid), $(3.1)$ forces

$$
\boxed{\;\mathbf{A}\big|_{\Gamma_{\rm out}} = \mathbf{I}\;}
\tag{3.3}
$$

On the inner (void) boundary $\Gamma_{\rm in}$ the gauge is **free**: the traction
transforms by the invertible factor $\mathbf{A}^{T}$, so $t=0 \Leftrightarrow \tilde t=0$.

So the gauge group is

$$
\mathcal{G} = \{\, \mathbf{A}\in C^1(\bar\Omega_{\rm cloak}, GL^{+}(2)) \;:\; \mathbf{A}|_{\Gamma_{\rm out}} = \mathbf{I} \,\}.
$$

Infinite-dimensional, connected (since $GL^+(2)$ is), but the *admissible* subset
(positive-definite $C^{\mathcal A}$, $\rho^{\mathcal A}\succ 0$, bounded moduli) is a
semialgebraic set with boundary — constrained optimization, not free
optimization on a manifold.

### 3.2 Isotropic-host expansion

Let $G := FA$ (so $G_{ji} = F_{jJ}A_{Ji}$), $B_A := A^{T}A$, $B_F := FF^{T}$. Then

$$
\boxed{\;J\,C^{\mathcal A}_{ijkl} \;=\; \lambda_0\,G_{ji}G_{lk} \;+\; \mu_0\,(B_A)_{ik}\,(B_F)_{jl} \;+\; \mu_0\,G_{li}G_{jk}. \;}
\tag{3.4}
$$

Setting $A=I$ recovers $(2.2)$. ✔

---

## 4. Can a gauge make the stiffness minor-symmetric? **Yes.**

Take

$$
\boxed{\;\mathbf{A} = c(x)\,\mathbf{F}^{T}, \qquad c(x) > 0 \;\text{ an arbitrary scalar field.}\;}
\tag{4.1}
$$

Then $G = c\,FF^{T} = c\,\mathbf{B}$ and $B_A = c^2\,\mathbf{B}$ where
$\mathbf{B} := FF^{T}$ is the (symmetric, PD) left Cauchy–Green tensor. Substituting
into $(3.4)$, every $G$ becomes symmetric and $(3.4)$ collapses to

$$
\boxed{\;C^{\mathcal A}_{ijkl} \;=\; \frac{c^2}{J}\Big[\lambda_0\,B_{ij}B_{kl} \;+\; \mu_0\big(B_{ik}B_{jl} + B_{il}B_{jk}\big)\Big]\;}
\tag{4.2}
$$

which is **manifestly minor-symmetric in $ij$ and in $kl$, and major-symmetric** —
i.e. an ordinary Cauchy-elastic stiffness. It is precisely $c^2 J^{-1}$ times the
full push-forward $F_{iI}F_{jJ}F_{kK}F_{lL}C^0_{IJKL}$ of the host tensor.

Accompanying density:

$$
\rho^{\mathcal A}_{ik} = \frac{c^2\rho_0}{J}\,B_{ik} \qquad \text{— anisotropic.}
\tag{4.3}
$$

Willis coupling:

$$
S_{ijk} = A_{Ii}\,\mathcal{C}_{IjKl}\,\partial_l(c\,F^{T})_{Kk}
\;\;\xrightarrow[\;\nabla F = 0\;]{}\;\; \propto\; \partial_l c .
\tag{4.4}
$$

### 4.0a Proposition (rank & inertia are gauge invariants)

Write the augmented Voigt matrix with the pair index $(ij)$ ordered
$(11,22,12,21)$. Then $(3.2)$ is a **congruence**:

$$
M^{\mathcal A} \;=\; J^{-1}\,K^{T} M^{0} K, \qquad K \;=\; \mathbf{A}\otimes \mathbf{F}^{T}, \qquad \det K = (\det A)^2 (\det F)^2 \neq 0 .
$$

By **Sylvester's law of inertia**, $M^{\mathcal A}$ has the same inertia as $M^0$ for
*every* gauge $\mathbf{A}$ and *every* map $F$. An isotropic $M^0$ has shear block
$\begin{psmallmatrix}\mu&\mu\\\mu&\mu\end{psmallmatrix}$, hence inertia
$(3\ \text{positive},\,1\ \text{zero})$. Therefore:

> **The augmented Voigt matrix is rank-3 — singular — for every gauge.**
> Local rotation costs no energy in the isotropic host, and no relabeling of
> the fields can invent a couple-stress stiffness that the host does not have.

Consequences, all verified numerically in `sweep_gauge.py`:

* "Cosserat-ness" of the identity gauge is **not** rotational stiffness. It is
  that the invariant null direction has been *tilted away* from the pure-rotation
  axis $(0,0,1,-1)/\sqrt2$ — which is exactly what makes the stress non-symmetric.
* Couple stiffness is therefore **not** an axis the gauge can trade along. The
  trade in §4.1 is genuinely two-way, not three-way.
* Stability must be checked on the symmetric-strain $3\times3$ block, never on the
  augmented $4\times4$ (`objectives.positive_definite`).

### 4.0b Proposition (the symmetrizing gauge is unique)

Minor symmetry $\Leftrightarrow$ the invariant null direction is rotated back
*onto* the antisymmetric axis. That is one vector condition, and it pins the
gauge down completely:

$$
\boxed{\;\mathcal{E}_{\rm asym}[\mathbf{A}] = 0 \iff \mathbf{A} = c\,\mathbf{F}^{T},\quad c \in \mathbb{R}\setminus\{0\}. \;}
$$

Verified numerically: unconstrained minimization of $\mathcal{E}_{\rm asym}$ over all
four components of $\mathbf{A}$, from random starts, converges to
$\|\mathbf{A} - c\mathbf{F}^{T}\|/\|\mathbf{A}\| \sim 10^{-14}$ every time, with $c$ scattered
(the material depends on $\mathbf{A}$ quadratically, so the sign of $c$ is irrelevant).

**This is the single most important constraint on the pipeline.** Demanding
*exact* minor symmetry consumes the entire gauge freedom apart from one scalar
field $c(x)$ — and $c$ only rescales $(C, \rho)$ by $c^2$, leaving all wave speeds
and all modulus *ratios* untouched. So the two rigid ratios at the end of §4.3
cannot be tuned by any gauge. If the D2 reachable set does not contain them, the
only remaining levers are the **map $\chi$** and the **host material**.

### 4.1 The three-way trade (the actual no-go)

Nothing is free. Comparing the two endpoints:

| gauge | minor symmetry | density | Willis $S$ |
|---|---|---|---|
| $\mathbf{A}=\mathbf{I}$ | ✗ broken (non-symmetric stress) | scalar $\rho_0/J$ | $0$ |
| $\mathbf{A}=\mathbf{F}^{T}$ | ✔ exact Cauchy | tensor $\rho_0 \mathbf{B}/J$ | $0$ **iff** $\nabla F = 0$ |

You cannot have minor symmetry **and** isotropic density **and** $S=0$
simultaneously — the gauge *moves* the exoticism, it does not remove it. Choose
which pathology your microstructure family is best at faking.

For D2 blocks the choice is forced, and in a helpful direction:

* a D2 cell is **centrosymmetric** $\Rightarrow$ every odd-rank effective tensor
  vanishes $\Rightarrow$ it can realize $S = 0$ **only**;
* Cauchy homogenization of a non-micropolar cell $\Rightarrow$ minor symmetry is
  **forced**, so the $\mathbf{A}=\mathbf{I}$ column is simply not in the catalogue;
* anisotropic density is the one exotic ingredient a D2 block can approximate at
  all (via directional mass layout), and even then only weakly.

So the gauge axis that matters here is: **how much minor-symmetry violation you
accept in exchange for how little density anisotropy** — quantified by $s$ in §5.

### 4.2 Why this is unusually favourable for the *triangular* cloak

`rayleigh_cloak/geometry/triangular.py:53` gives a **piecewise-constant** $F$:

$$
F = \begin{pmatrix} 1 & 0 \\ \pm a/c & (b-a)/b \end{pmatrix},
\qquad
J = \frac{b-a}{b},
$$

with the sign selected by $x \gtrless x_c$. The map is piecewise **affine**, so
$\nabla F = \mathbf{0}$ inside each half of the cloak. Taking $c \equiv \text{const}$
in $(4.1)$ therefore gives, exactly and with no approximation:

$$
S = T = W = \mathbf{0} \quad\text{in the cloak interior.}
$$

So on the triangular geometry the $\mathbf{A}=\mathbf{F}^{T}$ gauge yields a
**Willis-free, Cauchy-elastic, piecewise-constant** cloak material. The entire
residual difficulty is pushed into (a) the anisotropic density $(4.3)$ and
(b) the boundary condition $(3.3)$, which a piecewise-constant $\mathbf{A}$ violates
at $\Gamma_{\rm out}$.

### 4.3 Orthotropy — the target lands in the D2 class for free

Work in the eigenframe of $\mathbf{B} = FF^{T}$, where $\mathbf{B} = \mathrm{diag}(b_1,b_2)$.
Then $B_{12}=0$ and $(4.2)$ gives

$$
C_{1112} \propto \lambda_0 B_{11}B_{12} + 2\mu_0 B_{11}B_{12} = 0,
\qquad
C_{2212} \propto \lambda_0 B_{22}B_{12} + 2\mu_0 B_{22}B_{12} = 0,
$$

i.e. $C_{16}=C_{26}=0$: **the material is exactly orthotropic in the principal
frame of $\mathbf{B}$**, with closed-form moduli

$$
\boxed{
\begin{aligned}
C_{11} &= \tfrac{c^2}{J}(\lambda_0 + 2\mu_0)\,b_1^{2}, &\qquad
C_{22} &= \tfrac{c^2}{J}(\lambda_0 + 2\mu_0)\,b_2^{2},\\
C_{12} &= \tfrac{c^2}{J}\,\lambda_0\,b_1 b_2, &\qquad
C_{66} &= \tfrac{c^2}{J}\,\mu_0\,b_1 b_2,
\end{aligned}}
\tag{4.5}
$$

and cell orientation $\theta = $ the principal angle of $\mathbf{B}$.

That is exactly the parametrization the D2 microstructure generator is
conditioned on ($C_{11}, C_{12}, C_{22}, C_{66}$ — see the four
`microstructure_generation_2d/scaler_*` files) plus a rigid rotation, which a D2
cell realizes by construction.

**Two structural constraints fall out of $(4.5)$ and must be checked against the
dataset before anything else:**

$$
\frac{C_{12}}{C_{66}} = \frac{\lambda_0}{\mu_0} \quad (\text{= 1 for } \nu_0 = 0.25),
\qquad
\frac{C_{11}C_{22}}{C_{12}C_{66}} = \frac{(\lambda_0+2\mu_0)^2}{\lambda_0\mu_0}.
$$

Both are **independent of the gauge scalar $c$ and of $F$** — they are rigid
consequences of the host being isotropic (and by §4.0b no other gauge can reach
them either). For the repo's substrate ($\nu_0 = 1/4$, so $\lambda_0=\mu_0$) they read

$$
\frac{C_{12}}{C_{66}} = 1, \qquad \frac{C_{11}C_{22}}{C_{12}C_{66}} = 9 .
$$

**Measured against the fitted D2 reachable set** (`dataset/gmm/gmm_flat4_squared_2m.npz`,
906k homogenized cells; 600k GMM samples drawn):

| condition | fraction of reachable set |
|---|---|
| stiffness anisotropy $\max/\min \in [10,30]$ (target 17.9) | 1.40 % |
| … and $C_{12}/C_{66}\in[0.75,1.25]$ | 0.091 % |
| … and $C_{11}C_{22}/(C_{12}C_{66})\in[6.75,11.25]$ | **0.006 %** (32 of 520k) |

Non-empty, but deep in the tail — and a GMM tail is a smoothed extrapolation,
not evidence that manufacturable cells live there. There is also a scale
mismatch: the dataset's cells run at $\sqrt{C/\rho} \approx 1000\text{–}2560$ m/s
against a substrate at $c_s = 300$ m/s, i.e. the catalogue was built for a base
material roughly 5–8× too stiff.

**Conclusion: do not target $s=1$.** Exact minor symmetry is *possible* but lands
outside what the current D2 catalogue plausibly builds. Optimize $s(x)$ instead
(§5) and buy back reachability with a little residual asymmetry — which is
exactly what the sweep in `sweep_gauge.py` quantifies.

---

## 5. Interpolating family used by the optimizer

A one-parameter path from the identity gauge to the Cauchy gauge, staying in
$GL^{+}$:

$$
\boxed{\;\mathbf{A}(s,c) \;=\; c\,\exp\!\big(s\,\log \mathbf{F}^{T}\big) \;=\; c\,(\mathbf{F}^{T})^{s}, \qquad s\in[0,1].\;}
\tag{5.1}
$$

$s=0 \Rightarrow \mathbf{A}=c\,\mathbf{I}$ (identity gauge, Cosserat);
$s=1 \Rightarrow \mathbf{A}=c\,\mathbf{F}^{T}$ (Cauchy).
$\det \mathbf{A} = c^2 J^{s} > 0$ throughout. Well-defined here because
$\mathrm{spec}(F) = \{1,\,(b-a)/b\}$ is real and positive.

Promoting $s\to s(x)$, $c\to c(x)$ gives a **2-scalar-field** design space, which
is what the pipeline actually optimizes. The boundary condition $(3.3)$ reads
$s = 0$, $c = 1$ on $\Gamma_{\rm out}$; a transition layer of thickness $\ell$ near
$\Gamma_{\rm out}$ carries $\nabla A = O(1/\ell)$ and hence Willis terms
$S = O(1/\ell)$, $W = O(1/\ell^2)$ localized there.

---

## 6. Objective functionals

All norms are Frobenius on the augmented $4\times4$ Voigt matrix
(`materials.py::C_to_voigt4`), basis order $(11, 22, 12, 21)$.

**Minor-symmetry violation ("chirality").** With $\Pi$ the projector implemented
by `materials.py::symmetrize_stiffness`,

$$
\mathcal{E}_{\rm asym}[\mathbf{A}] \;=\; \frac{\big\|\,C^{\mathcal A} - \Pi\,C^{\mathcal A}\,\big\|}{\big\|C^{\mathcal A}\big\|}.
\tag{6.1}
$$

**Density anisotropy.**

$$
\mathcal{E}_{\rho}[\mathbf{A}] \;=\; \log\frac{\Lambda_{\max}(\rho^{\mathcal A})}{\Lambda_{\min}(\rho^{\mathcal A})}.
\tag{6.2}
$$

**Willis magnitude** (non-dimensionalized by a reference length $\ell_{\rm ref}$, e.g. one cell):

$$
\mathcal{E}_{S}[\mathbf{A}] \;=\; \frac{\ell_{\rm ref}\,\|S\|}{\|C^{\mathcal A}\|}.
\tag{6.3}
$$

**Distance to the D2 reachable set.** Let $\Pi$ project onto minor-symmetric,
$R(\theta)$ rotate, and $\mathcal{B}$ be the achievable box in
$(C_{11},C_{12},C_{22},C_{66})$ estimated from the dataset. Then

$$
\mathcal{E}_{\rm D2}[\mathbf{A}] \;=\; \min_{\theta}\;\Big[\underbrace{\big\|R(\theta)\!\cdot\!\Pi C^{\mathcal A}\big\|_{16,26}^{2}}_{\text{orthotropy defect}} \;+\; \underbrace{\mathrm{dist}^2\big(\mathrm{moduli}(\theta),\,\mathcal{B}\big)}_{\text{reachability}}\Big].
\tag{6.4}
$$

**Total, minimized over the gauge fields $s(x), c(x)$ (and optionally the map $\chi$):**

$$
\boxed{\;
\min_{s(\cdot),\,c(\cdot)}\;\int_{\Omega_{\rm cloak}}
\Big[ w_1 \mathcal{E}_{\rm asym} + w_2 \mathcal{E}_{\rho} + w_3 \mathcal{E}_{S} + w_4 \mathcal{E}_{\rm D2}\Big]\,dx
\quad\text{s.t.}\quad s|_{\Gamma_{\rm out}}=0,\; c|_{\Gamma_{\rm out}}=1,\; C^{\mathcal A}\succ0.
\;}
\tag{6.5}
$$

The scattering objective does **not** appear: it is exactly invariant along the
gauge orbit in the continuum limit. It re-enters only after discretization into
finite cells, which is why stage 5 of the pipeline re-runs the FEM.

---

## 7. Verification identities (implemented as tests)

1. $\mathbf{A}=\mathbf{I}$ reproduces `materials.py::C_eff` to machine precision.
2. $C^{\mathcal A}$ has major symmetry for every $\mathbf{A}$.
3. $\mathbf{A}=c\mathbf{F}^{T}$ gives $\mathcal{E}_{\rm asym} = 0$ to machine precision.
4. $\mathbf{A}=c\mathbf{F}^{T}$ gives $C_{16}=C_{26}=0$ in the eigenframe of $FF^{T}$,
   and moduli matching the closed form $(4.5)$.
5. $\rho^{\mathcal A}$ is symmetric positive-definite for every invertible $\mathbf{A}$.
6. With $\nabla A = 0$: $S = T = W = 0$.
7. $M^{\mathcal A}$ has inertia $(3^+, 1^0)$ for every $\mathbf{A}$, including random
   non-family gauges (§4.0a).
8. Unconstrained minimizers of $\mathcal{E}_{\rm asym}$ over $\mathbf{A}\in GL(2)$ all
   satisfy $\mathbf{A} \parallel \mathbf{F}^{T}$ (§4.0b).

**Everything above currently passes** — run `python -m gauge_opt.sweep_gauge` and
`pytest gauge_opt/test_gauge.py`.

> Note: these identities hold to $\sim10^{-16}$ only in float64. JAX defaults to
> float32, where they floor at $\sim10^{-7}$ and look like modelling error;
> `gauge_opt/__init__.py` enables x64 on import.

---

## 8. References

- Milton, Briane & Willis (2006), *On cloaking for elasticity and physical
  equations with a transformation invariant form*, New J. Phys. **8** 248.
- Brun, Guenneau & Movchan (2009), *Achieving control of in-plane elastic waves*,
  Appl. Phys. Lett. **94** 061903. — the identity gauge / Cosserat cloak.
- Norris & Shuvalov (2011), *Elastic cloaking theory*, Wave Motion **48** 525–538.
  — the gauge itself; §3.2 above follows their construction.
- Norris (2008), *Acoustic cloaking theory*, Proc. R. Soc. A **464** 2411.
  — precedent for *optimizing* over the family (minimum-mass cloak).
