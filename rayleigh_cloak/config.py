"""Pydantic configuration models and derived physical parameters.

All user-facing parameters live in ``SimulationConfig`` (serialisable to/from
YAML).  Computed quantities that depend on multiple config fields are collected
in ``DerivedParams`` so they are calculated once and threaded through the code
without module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, model_validator


# ── user-facing config sections ──────────────────────────────────────


class MaterialConfig(BaseModel):
    rho0: float = 1600.0
    cs: float = 300.0


class DomainConfig(BaseModel):
    f_star: float = 2.0
    lambda_star: float = 1.0
    H_factor: float = 4.305  # H = H_factor * lambda_star
    W_factor: float = 12.5   # W = W_factor * lambda_star


class TriangularCloakConfig(BaseModel):
    a_factor: float = 0.0774   # inner depth  = a_factor * H
    b_factor: float = 3.0      # outer depth  = b_factor * a
    c_factor: float = 0.1545   # half-width   = c_factor * H


class CircularCloakConfig(BaseModel):
    ri: float = 0.050          # inner (void) radius [m]
    rc: float = 0.150          # outer (cloak boundary) radius [m]


class AbsorbingConfig(BaseModel):
    L_pml_factor: float = 1.0  # L_pml = factor * lambda_star
    xi_max: float = 4.0
    pml_pow: int = 2
    pml_all_sides: bool = False  # True for plane-wave (PML on all 4 sides)


class MeshConfig(BaseModel):
    """Mesh-generation knobs.

    The cloak mesh is built from three target element sizes derived from a
    common base ``h_elem = min(W_total/nx_total, H_total/ny_total)``:

      - inside the cloak       : ``h_elem / refinement_factor_cloak``
      - far from the surface   : ``h_elem / refinement_factor_outside``
      - near the free surface  : ``h_elem / refinement_factor_surface``

    For backwards compatibility, the ``_cloak`` and ``_surface`` factors fall
    back to the legacy ``refinement_factor`` when unset, and ``_outside``
    defaults to 1.0 (the prior behaviour). Set ``refinement_factor_outside``
    *below* 1.0 to coarsen the mesh away from cloak/surface (e.g. 0.5 → 2×
    larger elements there).
    """
    model_config = {"extra": "ignore"}
    n_pml_x: int = 32
    n_pml_y: int = 32
    nx_phys: int = 200
    ny_phys: int = 60
    refinement_factor: int = 4
    refinement_factor_cloak: float | None = None
    refinement_factor_outside: float = 1.0
    refinement_factor_surface: float | None = None
    # When true, the (n_x-1) + (n_y-1) interior macro-grid lines are embedded
    # as 1D constraints in the gmsh surface, so no FEM element straddles a
    # macro-cell boundary. The element-to-cell map becomes exact, eliminating
    # the alignment-noise that otherwise affects piecewise-constant material
    # coefficients across mesh refinements. Default false (legacy behaviour).
    # Currently only honoured in the full-mesh path (generate_mesh_full); the
    # defect-cutout path silently ignores it.
    embed_macro_grid: bool = False
    ele_type: str = "TRI3"
    # Mesh builder selection. "legacy" (default) uses the graded
    # Distance->Threshold cloak refinement in ``rayleigh_cloak.mesh`` and is
    # bit-for-bit unchanged. "uniform_tri6" uses ``rayleigh_cloak.mesh_uniform``:
    # a uniform element size across the whole cloak bounding box (one size per
    # macro cell) instead of a graded field, intended together with
    # ``ele_type: TRI6`` (quadratic) for low-dispersion macro solves. The
    # builder honours ``ele_type`` for both TRI3 and TRI6.
    builder: Literal["legacy", "uniform_tri6"] = "legacy"


class SourceConfig(BaseModel):
    source_type: str = "gaussian"  # "gaussian" (surface) or "plane_wave" (left boundary)
    wave_type: str = "P"           # "P" or "S" (for plane_wave source)
    frequency_hz: float = 0.0     # Hz; if >0, overrides f_star-based omega
    x_src_factor: float = 0.05    # x_src = factor * W
    sigma_factor: float = 0.01    # sigma = factor * lambda_star
    F0: float = 1.0


class SolverConfig(BaseModel):
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    pc_factor_mat_solver_type: str = ""  # e.g. "mumps" or "mkl_pardiso" for parallel LU


class CellConfig(BaseModel):
    """Cell-based material decomposition of the cloak region."""
    enabled: bool = False
    n_x: int = 10                        # cells in x direction within cloak bbox
    n_y: int = 10                        # cells in y direction
    n_C_params: Literal[2, 4, 6, 10, 16] = 6   # 2=isotropic (λ,μ), 6=block-diag Cosserat (recommended), 16=full voigt4
    symmetrize_init: bool = False        # symmetrize C_eff (minor symmetry) when building init params (pushforward only)
    # "pushforward":      transformation-elasticity seed (anisotropic, off-manifold).
    # "homogeneous":      every cell starts at background (C0, rho0) — pure cement,
    #                     also off-manifold (stiffer than the porous CA cells).
    # "dataset_centroid": cloak cells start at the (λ, μ, ρ) centroid of the dataset
    #                     pulled from `init_path`; near-zero GMM penalty at step 0.
    init: Literal["pushforward", "homogeneous", "dataset_centroid"] = "pushforward"
    # Required when init="dataset_centroid": path to the GMM .npz produced by
    # `python -m dataset.cellular_chiral.fit_gmm` (its `feature_mean` field is
    # the centroid).
    init_path: str | None = None
    # When True, a quadrature point receives a cell's optimised material ONLY if
    # the point itself lies in the cloak annulus (per-point ``in_cloak`` test);
    # points outside the annulus get the background material even inside a cloak
    # cell's grid footprint. Default False keeps the legacy cell-centre masking
    # (material fills each cloak cell's whole grid cell). Set True for coarse
    # grids (esp. 1x1) where a cell's bbox footprint spills far outside the
    # triangular annulus into the bulk half-space.
    confine_to_cloak: bool = False
    # When True (requires n_C_params=6), the cell stiffness is parametrised as the
    # 6-param anisotropic **Cauchy** form [C11,C22,C66,C12,C16,C26] (symmetric
    # shear + normal-shear coupling) instead of the default block-diagonal
    # Cosserat flat6. This is EXACTLY the symmetrized-triangle material class
    # (materials.flat6cauchy); use it with init="pushforward"+symmetrize_init to
    # start at the symmetrized cloak and test whether optimisation improves on it.
    aniso_cauchy: bool = False

    @model_validator(mode="after")
    def _check_pushforward_only_fields(self) -> "CellConfig":
        if self.symmetrize_init and self.init != "pushforward":
            raise ValueError(
                f"symmetrize_init=True has no effect when init='{self.init}'; "
                "it only applies to the 'pushforward' initialisation."
            )
        if self.aniso_cauchy and self.n_C_params != 6:
            raise ValueError(
                f"aniso_cauchy=True requires n_C_params=6 (the anisotropic-Cauchy "
                f"flat6cauchy layout); got n_C_params={self.n_C_params}."
            )
        return self


class MultiFreqConfig(BaseModel):
    """Multi-frequency optimization settings.

    When ``f_stars`` is non-empty, the optimizer evaluates the cloaking loss
    at each listed frequency and sums the weighted contributions.  Each
    frequency gets its own FEM problem, reference solution, and
    ``ad_wrapper``; the material parameters are shared across all of them.

    Forward+adjoint solves at different frequencies are independent and
    are dispatched in parallel via a thread pool (PETSc releases the GIL).

    Two strategies are available:
    - ``"mean"`` (default): weighted sum of losses across all frequencies.
    - ``"minimax"``: at each iteration, only the frequency with the largest
      loss contributes to the gradient update.  Frequencies are specified
      via ``f_min``/``f_max``/``f_step`` (or ``f_stars`` as fallback).
    """
    strategy: Literal["mean", "minimax"] = "mean"
    f_stars: list[float] = []
    weights: list[float] = []      # per-frequency weight; [] → uniform (mean only)
    f_min: float | None = None     # minimax: start of frequency range
    f_max: float | None = None     # minimax: end of frequency range
    f_step: float | None = None    # minimax: frequency step
    max_workers: int = 0           # thread-pool size; 0 → len(f_stars)


class MaterialCementGMMConfig(BaseModel):
    """Dataset-prior penalty over (λ, μ, ρ) using a fitted Gaussian Mixture.

    Pushes the optimiser toward (λ, μ, ρ) regions actually populated by the
    cellular-chiral dataset. The penalty is a flat-top:
    ``max(0, threshold - log p(λ, μ, ρ))`` averaged over cloak cells, with
    threshold τ stored inside the .npz at fit time.

    Fit with ``python -m dataset.cellular_chiral.fit_gmm``. Only cloak cells
    contribute — background cells are physically valid by construction.
    """
    model_config = {"extra": "ignore"}
    enabled: bool = False
    path: str = "output/ca_bulk_squared/gmm_lambda_mu_rho.npz"
    weight: float = 1.0
    # Quantile of the dataset log-p distribution used as τ. Must lie in the
    # range of quantiles stored in the .npz (currently [0.01, 0.75]).
    # Lower → looser flat-top, higher → stricter. None → use τ baked in at fit.
    quantile: float | None = None


class RegularizationsConfig(BaseModel):
    """Sub-section of LossConfig for material/geometry regularisations.

    Distinct from the ``lambda_l2`` / ``lambda_neighbor`` knobs on
    ``OptimizationConfig`` because these are *prior-on-the-output*
    penalties tied to a learned dataset, not gradient-flow regularisers.
    """
    model_config = {"extra": "ignore"}
    material_cement_GMM: MaterialCementGMMConfig = MaterialCementGMMConfig()


class LossConfig(BaseModel):
    """Settings for cloaking loss computation.

    The ``type`` field selects which region and metric to evaluate:

    - ``"right_boundary"`` (default): relative L2 on the right physical boundary.
    - ``"top_surface"``: transmission ratio (ratio - 1)^2 on the free surface
      beyond the cloak footprint.
    - ``"outside_cloak"``: relative L2 over all physical-domain nodes outside
      the cloak.
    - ``"depth_line"``: relative L2 on a horizontal line at depth ``depth``
      below the free surface, restricted to x-positions outside the cloak
      footprint. ``n_eval_points`` controls the line resolution.
    - ``"surface_depth"``: same transmission-ratio metric as ``"top_surface"``
      (``(<|u_cloak|>/<|u_ref|> - 1)^2``), but the ``n_eval_points`` samples keep
      a surface-like x-spread while being pushed to a depth power-law biased
      toward the free surface (see ``alpha``). Samples inside the cloak/defect
      are dropped, so only the field beyond/below the cloak is measured.
    - ``"surface_column"``: phase-invariant per-node magnitude error on a
      2-D cloud built from ``n_eval_points`` surface x-positions (same
      selection as ``"top_surface"``) crossed with ``n_column_samples``
      y-positions evenly spaced from ``y_top - depth`` up to ``y_top``.
      Snaps each ``(x, y)`` pair to the nearest mesh node and deduplicates.
    - ``"magnitude_band_integral"``: mean of ``(|u_cloak|/|u_ref| - 1)^2``
      over all cloak-mesh nodes in the top band ``[y_top - depth, y_top]``,
      excluding the cloak/defect footprint. The x-extent is controlled by
      ``band_x_filter``: ``"downstream"`` (default) restricts to nodes with
      ``x > x_c``, matching ``"top_surface"`` exactly when ``depth == 0``;
      ``"full"`` keeps the entire physical x-range.
    """
    model_config = {"extra": "ignore"}
    type: Literal["right_boundary", "top_surface", "outside_cloak", "depth_line", "surface_depth", "surface_column", "magnitude_band_integral"] = "right_boundary"
    multi_freq: MultiFreqConfig = MultiFreqConfig()
    regularizations: RegularizationsConfig = RegularizationsConfig()
    # Mesh-independent surface metric: when ``n_eval_points > 0``, the
    # validation/benchmark scripts evaluate ``<|u|>`` at this many fixed
    # x-positions on the free surface (interpolated from mesh nodes) rather
    # than averaging over a mesh-dependent set of surface nodes. Default 0
    # falls back to the legacy node-based metric.
    n_eval_points: int = 0
    eval_noise_sigma: float = 0.0   # Gaussian noise on the x-positions, in
                                     # physical units, to break resonance with
                                     # any test wavelength (recommended ~λ*/200).
    eval_noise_seed: int = 0
    # Depth below the free surface (physical units) at which the
    # ``"depth_line"`` loss evaluates u. Must be > 0; depth == a places the
    # line at the triangular defect's apex, depth > b places it strictly
    # below the cloak.
    depth: float = 0.0
    # Power-law exponent for the ``"surface_depth"`` loss depth distribution
    # (physical_y = y_top - H*(1 - U**(1/alpha)), U ~ U(0,1)). alpha == 1 is
    # uniform in depth; larger alpha concentrates samples nearer the surface.
    alpha: float = 4.0
    # Number of y-positions per surface x-column for the ``"surface_column"``
    # loss. Y-positions are evenly spaced in [y_top - depth, y_top]; the
    # total eval-point count before snapping is n_eval_points * n_column_samples.
    n_column_samples: int = 1
    # X-extent selector for the ``"magnitude_band_integral"`` loss:
    # ``"downstream"`` keeps cloak-mesh nodes with x > x_c (matches
    # ``"top_surface"`` when depth == 0); ``"full"`` keeps the entire
    # physical x-range [x_off, x_off + W]. Cloak/defect-interior nodes are
    # always excluded.
    band_x_filter: Literal["downstream", "full"] = "downstream"


class NeuralReparamConfig(BaseModel):
    """Settings for neural reparameterization of material fields."""
    hidden_size: int = 256
    n_layers: int = 4
    n_fourier: int = 32
    seed: int = 42
    output_scale: float = 0.1  # multiplier on MLP residual (controls max step size)
    init_weights: str = ""     # path to .npz with saved MLP weights for warm-start
    # Physically-valid flat4 decode (n_C_params=4). When enabled, the decode
    # parameterizes (C11, C22, C66, C12) so that C11/C22/C66 > 0, the orthotropic
    # block is positive-definite (C11*C22 - C12^2 > 0), and the anisotropy ratio
    # C11/C22 stays in [1/R, R]. Default off keeps the legacy unbounded residual
    # bit-for-bit reproducible.
    constrained: bool = False
    kappa: float = 0.95            # |C12/sqrt(C11*C22)| bound (<1 => positive-definite)
    cap_anisotropy: bool = True    # bound C11/C22 in [1/R, R] (flat4 constrained path)
    anisotropy_ratio: float = 15.0  # R; ~ dataset C11/C22 p99 (12x) + margin

    @model_validator(mode="after")
    def _check_constrained_fields(self) -> "NeuralReparamConfig":
        if self.constrained:
            if not (0.0 < self.kappa < 1.0):
                raise ValueError(
                    f"kappa must be in (0, 1) for a positive-definite C12 "
                    f"coupling; got {self.kappa}."
                )
            if self.anisotropy_ratio <= 1.0:
                raise ValueError(
                    f"anisotropy_ratio must be > 1; got {self.anisotropy_ratio}."
                )
        return self


class TopoNeuralConfig(BaseModel):
    """Settings for topology neural reparameterization (pixel-level density)."""
    hidden_size: int = 256
    n_layers: int = 4
    n_fourier: int = 32
    seed: int = 42
    pixel_per_cell: int = 10       # fine pixels per coarse cell edge
    simp_p: float = 3.0            # SIMP penalisation exponent
    E_cement: float = 30e9         # Pa, solid-phase Young's modulus
    nu_micro: float = 0.2          # solid-phase Poisson's ratio
    rho_cement: float = 2300.0     # kg/m^3, solid-phase density
    lambda_bin: float = 0.01       # binarisation penalty weight
    rho_weight: float = 0.1        # weight for density in dataset matching
    dataset_path: str = "output/dataset_30k.h5"
    output_scale: float = 0.1     # MLP residual scale in logit space
    density_eps: float = 0.01     # clamp targets away from 0/1 for finite logits
    beta_start: float = 1.0       # Heaviside projection sharpness at start
    beta_end: float = 32.0        # Heaviside projection sharpness at end
    fourier_sigma: float = 0.0    # random Fourier bandwidth (0 = axis-aligned linspace)


class OptimizationConfig(BaseModel):
    """Settings for cell-based material optimization."""
    method: Literal["raw", "neural", "neural_topo"] = "raw"
    n_iters: int = 100
    lr: float = 1e-3
    lr_end: float | None = None       # if set, decay lr → lr_end
    lr_schedule: Literal["linear", "cosine"] = "linear"
    lambda_l2: float = 1e-4       # L2 regularization (drift from init)
    lambda_neighbor: float = 1e-3  # neighbor smoothness regularization
    plot_every: int = 1            # plot |Re(u)| every N steps (0 = disabled)
    init_params: str = ""          # path to .npz with cell_C_flat/cell_rho for warm-start
    neural: NeuralReparamConfig = NeuralReparamConfig()
    topo_neural: TopoNeuralConfig = TopoNeuralConfig()


class NassarConfig(BaseModel):
    """Nassar cell parameterization settings."""
    enabled: bool = False
    cell_n_x: int = 20       # cells in x direction (Cartesian grid over annulus bbox)
    cell_n_y: int = 20       # cells in y direction
    polar: bool = False       # use polar cell decomposition (N sectors × M layers)
    lattice_N: int = 75       # angular sectors (paper default)
    lattice_M: int = 22       # radial layers (paper default)


class SimulationConfig(BaseModel):
    is_reference: bool = False
    geometry_type: str = "triangular"
    # When True (continuous, non-reference, non-cell forward solve), the
    # transformation-elasticity stiffness C_eff is symmetrised inside the cloak
    # using the augmented-Voigt recipe (``materials.symmetrize_stiffness``)
    # before solving — i.e. the *ideal* cloak material, but made minor-symmetric
    # (Cauchy) so it is realisable by a classical (non-Cosserat) medium. Default
    # False keeps the exact (asymmetric) push-forward. Has no effect when
    # ``is_reference`` or ``cells.enabled`` is True (cells use
    # ``cells.symmetrize_init`` instead).
    symmetrize_cloak: bool = False

    material: MaterialConfig = MaterialConfig()
    domain: DomainConfig = DomainConfig()
    geometry: TriangularCloakConfig = TriangularCloakConfig()
    circular_geometry: CircularCloakConfig = CircularCloakConfig()
    absorbing: AbsorbingConfig = AbsorbingConfig()
    mesh: MeshConfig = MeshConfig()
    source: SourceConfig = SourceConfig()
    solver: SolverConfig = SolverConfig()
    cells: CellConfig = CellConfig()
    nassar: NassarConfig = NassarConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    loss: LossConfig = LossConfig()
    output_dir: str = "output"


def load_config(path: str | Path) -> SimulationConfig:
    """Load a ``SimulationConfig`` from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimulationConfig(**(data or {}))


# ── derived / computed quantities ────────────────────────────────────


@dataclass(frozen=True)
class DerivedParams:
    """Physical quantities derived from :class:`SimulationConfig`.

    All lengths are in the *extended-mesh* coordinate system (i.e. including
    PML padding).
    """

    # material
    rho0: float
    mu: float
    lam: float
    nu: float
    cs: float
    cp: float
    cR: float

    # frequency
    omega: float

    # domain extents (physical)
    H: float
    W: float
    lambda_star: float

    # PML
    L_pml: float
    xi_max: float
    pml_pow: int

    # extended domain
    W_total: float
    H_total: float
    x_off: float
    y_off: float

    # convenience
    y_top: float
    x_c: float  # cloak centre x

    # cloak geometry (triangular)
    a: float
    b: float
    c: float

    # source
    x_src: float
    x_src_phys: float
    sigma_src: float
    F0: float

    # mesh
    nx_total: int
    ny_total: int

    # cloak geometry (circular) — None when geometry_type != "circular"
    ri: float | None = None
    rc: float | None = None
    y_c: float | None = None       # centre y (x_c already exists)
    pml_all_sides: bool = False

    @staticmethod
    def from_config(cfg: SimulationConfig) -> DerivedParams:
        mat = cfg.material
        dom = cfg.domain
        geo = cfg.geometry
        ab = cfg.absorbing
        msh = cfg.mesh
        src = cfg.source

        rho0 = mat.rho0
        cs = mat.cs
        cp = np.sqrt(3.0) * cs
        mu = rho0 * cs ** 2
        lam = rho0 * cp ** 2 - 2 * mu
        nu = lam / (2 * (lam + mu))
        cR = cs * (0.826 + 1.14 * nu) / (1 + nu)

        lambda_star = dom.lambda_star

        # Frequency: use explicit frequency_hz if set, otherwise f_star.
        # f_star follows the paper convention (Chatzopoulos et al. 2023):
        #   f* = b / λ_R,  where b ≈ λ_star (b_factor * a_factor * H_factor ≈ 1)
        # so ω = 2π·cR·f* / λ_star.
        if src.frequency_hz > 0:
            omega = 2 * np.pi * src.frequency_hz
        else:
            omega = 2 * np.pi * dom.f_star * cR / lambda_star

        # Circular geometry: domain sized around the void
        ri_val = rc_val = y_c_val = None
        pml_all = ab.pml_all_sides
        if cfg.geometry_type == "circular":
            circ = cfg.circular_geometry
            ri_val = circ.ri
            rc_val = circ.rc
            # Domain = 4*rc × 4*rc (large enough for far-field measurement)
            H = 4.0 * rc_val
            W = 4.0 * rc_val
            # Override lambda_star from wavelength at given frequency
            if src.frequency_hz > 0:
                lambda_star = cs / src.frequency_hz
            pml_all = True  # circular geometry always uses all-side PML
        else:
            H = dom.H_factor * lambda_star
            W = dom.W_factor * lambda_star

        a = geo.a_factor * H
        b = geo.b_factor * a
        c = geo.c_factor * H

        L_pml = ab.L_pml_factor * lambda_star
        W_total = 2 * L_pml + W
        x_off = L_pml
        y_off = L_pml
        x_c = x_off + W / 2.0

        if cfg.geometry_type == "circular":
            H_total = 2 * L_pml + H  # PML on both top and bottom
            y_c_val = y_off + H / 2.0
        else:
            H_total = L_pml + H  # PML only on bottom

        y_top = H_total

        x_src_phys = src.x_src_factor * W
        x_src = x_off + x_src_phys
        sigma_src = src.sigma_factor * lambda_star

        nx_total = msh.n_pml_x + msh.nx_phys + msh.n_pml_x
        ny_total = msh.n_pml_y + msh.ny_phys
        if cfg.geometry_type == "circular":
            ny_total = msh.n_pml_y + msh.ny_phys + msh.n_pml_y

        return DerivedParams(
            rho0=rho0, mu=mu, lam=lam, nu=nu, cs=cs, cp=cp, cR=cR,
            omega=omega,
            H=H, W=W, lambda_star=lambda_star,
            L_pml=L_pml, xi_max=ab.xi_max, pml_pow=ab.pml_pow,
            W_total=W_total, H_total=H_total, x_off=x_off, y_off=y_off,
            y_top=y_top, x_c=x_c,
            a=a, b=b, c=c,
            ri=ri_val, rc=rc_val, y_c=y_c_val, pml_all_sides=pml_all,
            x_src=x_src, x_src_phys=x_src_phys,
            sigma_src=sigma_src, F0=src.F0,
            nx_total=nx_total, ny_total=ny_total,
        )
