"""Pixel-level full-structure FEM for the multiscale diffusion pipeline.

Factored out of ``scripts/frequency_sweep_validated.py`` so it can be reused by
``MultiscaleDiffusionModel.predict_structure`` / ``compute_fem_loss`` (and, later,
by the torch<->JAX gradient bridge).

The cloak's macro cells each carry a 50x50 microstructure; those are tiled into
one fine-grained canvas and the macro elastodynamics is solved with material
assigned at the *pixel* level. The one substantive change from the validation
script is that the solid/void switch is replaced by a **SIMP-soft** interpolation
so the FEM is differentiable w.r.t. the (soft) pixel occupancy:

    C(occ) = C0 * (void_ratio + (1 - void_ratio) * occ**simp_p),   occ in [0, 1]

With ``binarize=True`` and ``simp_p=1`` this reduces to the original hard
solid/void assignment (occ thresholded at 0.5), for validation parity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import logging

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import jax
import jax.numpy as jnp
from jax_fem.solver import ad_wrapper, solver as jax_fem_solver

from rayleigh_cloak.absorbing import make_xi_profile
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import (
    find_embedded_eval_node_indices,
    make_fixed_surface_eval_points,
    transmitted_displacement_ratio,
)
from rayleigh_cloak.materials import C_iso
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.problem import (
    RayleighCloakProblem,
    _make_dirichlet_bc,
    _make_top_surface,
)
from rayleigh_cloak.solver import _create_geometry, solve_reference


logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── cloak grid / decomposition ──────────────────────────────────────


@dataclass
class CloakLayout:
    """Macro-grid geometry of the cloak (everything ``predict_structure`` needs)."""
    geometry: object
    dp: DerivedParams
    decomp: CellDecomposition
    n_x: int
    n_y: int
    cloak_bbox: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)


def build_cloak_layout(config) -> CloakLayout:
    """Geometry + regular cell grid + cloak mask for a rayleigh ``SimulationConfig``."""
    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)
    n_x, n_y = int(config.cells.n_x), int(config.cells.n_y)
    decomp = CellDecomposition(geometry, n_x, n_y)
    bbox = (decomp.x_min, decomp.x_max, decomp.y_min, decomp.y_max)
    return CloakLayout(geometry, dp, decomp, n_x, n_y, bbox)


def tile_image(geoms: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    """Tile (n_cells, H, W) into (n_y*H, n_x*W) with y-up, cell_idx = ix*n_y + iy.

    Matches ``CellDecomposition`` flattening and the convention assumed by
    ``PixelMaterialProblem._pixel_at`` (top of image = high y).
    """
    n_cells, H, W = geoms.shape
    assert n_cells == n_x * n_y, f"{n_cells} != {n_x}*{n_y}"
    canvas = np.zeros((n_y * H, n_x * W), dtype=geoms.dtype)
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            row = n_y - 1 - iy
            col = ix
            canvas[row * H:(row + 1) * H, col * W:(col + 1) * W] = geoms[idx]
    return canvas


# ── SIMP-soft material interpolation (shared by forward + differentiable paths) ──


def _simp_frac(occ, void_ratio, simp_p, binarize):
    """Stiffness fraction in ``[void_ratio, 1]`` (SIMP-soft, differentiable in occ)."""
    if binarize:
        occ = jnp.where(occ > 0.5, 1.0, 0.0)
    occ = jnp.clip(occ, 0.0, 1.0)
    return void_ratio + (1.0 - void_ratio) * occ ** simp_p


def _density_frac(occ, void_ratio, binarize):
    """Density fraction in ``[void_ratio, 1]`` (linear in occ; matches ersatz void)."""
    if binarize:
        occ = jnp.where(occ > 0.5, 1.0, 0.0)
    occ = jnp.clip(occ, 0.0, 1.0)
    return void_ratio + (1.0 - void_ratio) * occ


# ── pixel-level FEM problem (SIMP-soft material) ────────────────────


class PixelMaterialProblem(RayleighCloakProblem):
    """Elastodynamics with C(x), rho(x) read from a pixel canvas inside the cloak.

    Class attributes set by ``build_pixel_problem``:
        _canvas_jnp  : (H_pix, W_pix) jnp occupancy in [0, 1]
        _cloak_bbox  : (x_min, x_max, y_min, y_max) physical extent of the canvas
        _C0, _rho0   : solid (cement) stiffness / density
        _void_ratio  : E_void / E_solid (and rho_void / rho_solid)
        _simp_p      : SIMP penalisation exponent
        _binarize    : if True, threshold occupancy at 0.5 (hard solid/void)
        _xi_fn       : absorbing-profile callable
    """

    def custom_init(self):
        geo = self._geometry
        canvas = type(self)._canvas_jnp                  # (H_pix, W_pix)
        x_min, x_max, y_min, y_max = type(self)._cloak_bbox
        H_pix, W_pix = canvas.shape
        xi_fn = type(self).__dict__["_xi_fn"]

        # Precompute the quad-point → pixel gather (constant: depends on the quad
        # points and bbox, not on canvas *values*), so ``set_params(canvas)`` is a
        # plain differentiable gather. Mirrors the y-up convention of ``tile_image``
        # (top of image = high y, hence the ``1 - y_norm`` row flip).
        pts = np.asarray(self.physical_quad_points)      # (n_fem, n_qp, 2)
        inv_dx = 1.0 / (x_max - x_min)
        inv_dy = 1.0 / (y_max - y_min)
        x_norm = (pts[..., 0] - x_min) * inv_dx
        y_norm = (pts[..., 1] - y_min) * inv_dy
        col = np.clip((x_norm * W_pix).astype(np.int32), 0, W_pix - 1)
        row = np.clip(((1.0 - y_norm) * H_pix).astype(np.int32), 0, H_pix - 1)
        self._row_idx = jnp.asarray(row)                 # (n_fem, n_qp) int
        self._col_idx = jnp.asarray(col)

        # In-cloak mask at quad points (constant). Outside the cloak the material
        # is the solid background (C0, rho0) regardless of the canvas.
        self._in_clk_qp = jax.vmap(jax.vmap(geo.in_cloak))(self.physical_quad_points)

        # Absorbing profile (constant), stored separately for set_params.
        self._xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)

        # Build the initial material from the canvas the problem was created with.
        self.set_params(canvas)

    def set_params(self, canvas):
        """Rebuild ``internal_vars`` from a pixel ``canvas`` (differentiable).

        The quad-point→pixel indices, in-cloak mask, and absorbing profile are
        precomputed in ``custom_init``; only the SIMP-soft material assignment
        depends on the canvas, so the gather ``canvas[row_idx, col_idx]`` (constant
        indices) carries gradients w.r.t. the canvas *values*. This mirrors
        ``RayleighCloakProblem.set_params`` so ``ad_wrapper`` differentiates the
        FEM solve w.r.t. the canvas.
        """
        cls = type(self)
        canvas = jnp.asarray(canvas, dtype=jnp.float32)
        C0, rho0 = cls._C0, cls._rho0
        void_ratio, simp_p, binarize = cls._void_ratio, cls._simp_p, cls._binarize

        occ = canvas[self._row_idx, self._col_idx]                  # (n_fem, n_qp)
        in_clk = self._in_clk_qp

        C_pixel = _simp_frac(occ, void_ratio, simp_p, binarize)[..., None, None, None, None] * C0
        C_qp = jnp.where(in_clk[..., None, None, None, None], C_pixel, C0)

        rho_pixel = _density_frac(occ, void_ratio, binarize) * rho0
        rho_qp = jnp.where(in_clk, rho_pixel, rho0)

        self.internal_vars = [C_qp, rho_qp, self._xi_qp]


def build_pixel_problem(
    mesh,
    cfg,
    params: DerivedParams,
    geometry,
    canvas: np.ndarray,
    cloak_bbox: tuple[float, float, float, float],
    void_ratio: float = 1e-6,
    simp_p: float = 3.0,
    binarize: bool = False,
) -> PixelMaterialProblem:
    """Build a ``PixelMaterialProblem`` with pixel-level material from ``canvas``."""
    C0 = C_iso(params.lam, params.mu)
    canvas_jnp = jnp.asarray(canvas, dtype=jnp.float32)

    ProblemCls = type("PixelMaterialProblemInstance", (PixelMaterialProblem,), {
        "_omega":       params.omega,
        "_geometry":    geometry,
        "_is_reference": False,
        "_C0":          C0,
        "_rho0":        params.rho0,
        "_xi_fn":       make_xi_profile(params),
        "_x_src":       params.x_src,
        "_sigma_src":   params.sigma_src,
        "_F0":          params.F0,
        "_cell_decomp": None,
        "_n_C_params":  cfg.cells.n_C_params,
        "_source_type": cfg.source.source_type,
        "_wave_type":   cfg.source.wave_type,
        "_lam_param":   params.lam,
        "_mu_param":    params.mu,
        "_canvas_jnp":  canvas_jnp,
        "_cloak_bbox":  cloak_bbox,
        "_void_ratio":  void_ratio,
        "_simp_p":      simp_p,
        "_binarize":    binarize,
    })
    return ProblemCls(
        mesh=mesh,
        vec=4,
        dim=2,
        ele_type=cfg.mesh.ele_type,
        dirichlet_bc_info=_make_dirichlet_bc(params),
        location_fns=[_make_top_surface(params)],
    )


# ── surface eval indices (mirrors scripts/frequency_sweep_validated.py) ──


def _surface_indices(cloak_mesh, geometry, dp, kept_nodes, loss_cfg=None):
    if loss_cfg is not None and int(loss_cfg.n_eval_points) > 0:
        eval_xs = make_fixed_surface_eval_points(
            geometry, dp, int(loss_cfg.n_eval_points),
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed),
        )
        cs_idx = find_embedded_eval_node_indices(cloak_mesh.points, eval_xs, dp.y_top)
        return cs_idx, kept_nodes[cs_idx]
    x_left = dp.x_off
    x_right = dp.x_off + dp.W
    cs_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top, x_left, x_right,
    )
    return cs_idx, kept_nodes[cs_idx]


# ── top-level: solve the full pixel structure and return the loss ───


def structure_cloaking_loss(
    canvas: np.ndarray,
    config,
    cloak_bbox: tuple[float, float, float, float],
    refinement_factor: int | None = None,
    void_ratio: float = 1e-6,
    simp_p: float = 3.0,
    binarize: bool = False,
    solver_opts: dict | None = None,
):
    """Run the pixel-level full-structure FEM and return the cloaking loss.

    Mirrors a single ``run_validated_sweep`` iteration: per-frequency mesh,
    reference solve, pixel-material cloak solve, then the transmitted-displacement
    ratio on the free surface beyond the cloak.

    Returns ``(loss, u_val, diag)``. ``loss`` is the transmitted-displacement
    ratio (the cloaking objective; lower is better).
    """
    if refinement_factor is not None:
        config = config.model_copy(update={
            "mesh": config.mesh.model_copy(update={"refinement_factor": int(refinement_factor)})
        })
    if solver_opts is None:
        solver_opts = {"petsc_solver": {
            "ksp_type": config.solver.ksp_type,
            "pc_type": config.solver.pc_type,
        }}

    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)

    full_mesh = generate_mesh_full(config, dp, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    ref_result = solve_reference(config, mesh=full_mesh)

    problem = build_pixel_problem(
        cloak_mesh, config, dp, geometry,
        canvas=canvas, cloak_bbox=cloak_bbox,
        void_ratio=void_ratio, simp_p=simp_p, binarize=binarize,
    )
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u_val = np.asarray(sol_list[0])

    cs_idx, rs_idx = _surface_indices(cloak_mesh, geometry, dp, kept_nodes, loss_cfg=config.loss)
    loss = transmitted_displacement_ratio(u_val, ref_result.u, cs_idx, rs_idx)

    diag = {"n_nodes": len(cloak_mesh.points), "n_cells": int(cloak_mesh.cells.shape[0])}
    return loss, u_val, diag


# ── differentiable objective: loss(canvas) -> (loss, g_canvas) ──────


def _jnp_transmitted_ratio(u_case, u_ref_surf, case_surf_idx):
    """jnp port of ``transmitted_displacement_ratio`` (pure, autodiff-friendly).

    ``u_ref_surf`` is the reference surface displacement already indexed at the
    matching reference nodes (a constant), so only ``u_case`` carries gradients.
    Same surface-mean convention as the numpy metric, so the bridge loss matches
    the ``structure_cloaking_loss`` forward value on the same canvas.
    """
    u_s = u_case[case_surf_idx]
    mag_case = jnp.sqrt(u_s[:, 0]**2 + u_s[:, 1]**2 + u_s[:, 2]**2 + u_s[:, 3]**2)
    mag_ref = jnp.sqrt(
        u_ref_surf[:, 0]**2 + u_ref_surf[:, 1]**2
        + u_ref_surf[:, 2]**2 + u_ref_surf[:, 3]**2
    )
    return jnp.mean(mag_case) / (jnp.mean(mag_ref) + 1e-30)


@dataclass
class PixelFEMObjective:
    """Reusable differentiable pixel-FEM cloaking objective.

    Built once per sampling trajectory: the mesh, geometry, and reference solve
    do not depend on the canvas, so they (and the ``ad_wrapper``'d problem) are
    set up here and only the material (canvas) changes per call. ``__call__``
    returns ``(loss, g_canvas)`` where ``loss`` is the transmitted-displacement
    ratio and ``g_canvas`` is its gradient w.r.t. the pixel canvas.
    """
    config: object
    dp: DerivedParams
    geometry: object
    full_mesh: object
    cloak_mesh: object
    kept_nodes: np.ndarray
    cloak_bbox: tuple
    canvas_shape: tuple
    case_surf_idx: jnp.ndarray
    u_ref_surf: jnp.ndarray
    problem: PixelMaterialProblem
    fwd_pred: object
    value_and_grad_canvas: object

    def __call__(self, canvas):
        """``canvas`` (H_pix, W_pix) jnp → ``(loss, g_canvas)`` (both jnp)."""
        return self.value_and_grad_canvas(canvas)

    def loss_only(self, canvas):
        """Forward-only loss on ``canvas`` (for consistency checks)."""
        sol_list = self.fwd_pred(jnp.asarray(canvas, dtype=jnp.float32))
        return _jnp_transmitted_ratio(sol_list[0], self.u_ref_surf, self.case_surf_idx)

    def physical_ratio(self, canvas) -> float:
        """Transmitted-displacement ratio of the *binarized* (physical) structure.

        Thresholds ``canvas`` at 0.5 and runs the forward FEM only, so the value
        is the true cloaking performance of the binary microstructure — the metric
        ``scripts/frequency_sweep.py`` reports (``<|u|> / <|u_ref|>``; 1.0 = perfect
        cloak) — as opposed to the SIMP-soft ``loss`` the optimiser descends. With
        a ``{0, 1}`` occupancy the SIMP map collapses to hard solid/void for any
        ``simp_p``, so this is the physical structure regardless of how the
        objective was built. Costs one extra forward solve per call.
        """
        hard = jnp.where(jnp.asarray(canvas, dtype=jnp.float32) > 0.5, 1.0, 0.0)
        return float(self.loss_only(hard))


def build_pixel_objective(
    config,
    cloak_bbox: tuple[float, float, float, float],
    canvas_shape: tuple[int, int],
    refinement_factor: int | None = None,
    void_ratio: float = 1e-6,
    simp_p: float = 3.0,
    binarize: bool = False,
    solver_opts: dict | None = None,
) -> PixelFEMObjective:
    """One-time setup of the differentiable pixel-FEM objective.

    Builds the per-frequency mesh, reference solve, surface-eval indices, and the
    ``ad_wrapper``'d ``PixelMaterialProblem`` (with a dummy canvas — the material
    is supplied per call via ``set_params``). ``canvas_shape`` must match the
    canvas the caller will pass (``(n_y*CELL, n_x*CELL)``).
    """
    if refinement_factor is not None:
        config = config.model_copy(update={
            "mesh": config.mesh.model_copy(update={"refinement_factor": int(refinement_factor)})
        })
    if solver_opts is None:
        solver_opts = {"petsc_solver": {
            "ksp_type": config.solver.ksp_type,
            "pc_type": config.solver.pc_type,
        }}

    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)

    full_mesh = generate_mesh_full(config, dp, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    ref_result = solve_reference(config, mesh=full_mesh)

    # Dummy canvas: only its *shape* matters here; set_params supplies real values.
    dummy_canvas = np.zeros(canvas_shape, dtype=np.float32)
    problem = build_pixel_problem(
        cloak_mesh, config, dp, geometry,
        canvas=dummy_canvas, cloak_bbox=cloak_bbox,
        void_ratio=void_ratio, simp_p=simp_p, binarize=binarize,
    )

    cs_idx, rs_idx = _surface_indices(cloak_mesh, geometry, dp, kept_nodes, loss_cfg=config.loss)
    case_surf_idx = jnp.asarray(cs_idx)
    u_ref_surf = jnp.asarray(np.asarray(ref_result.u)[rs_idx])

    fwd_pred = ad_wrapper(problem, solver_opts, solver_opts)

    def loss_fn(canvas):
        sol_list = fwd_pred(canvas)
        return _jnp_transmitted_ratio(sol_list[0], u_ref_surf, case_surf_idx)

    value_and_grad_canvas = jax.value_and_grad(loss_fn)

    return PixelFEMObjective(
        config=config, dp=dp, geometry=geometry,
        full_mesh=full_mesh, cloak_mesh=cloak_mesh, kept_nodes=kept_nodes,
        cloak_bbox=tuple(cloak_bbox), canvas_shape=tuple(canvas_shape),
        case_surf_idx=case_surf_idx, u_ref_surf=u_ref_surf,
        problem=problem, fwd_pred=fwd_pred,
        value_and_grad_canvas=value_and_grad_canvas,
    )
