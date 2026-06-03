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

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import jax
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

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
        C0 = self._C0
        rho0 = self._rho0
        canvas = type(self)._canvas_jnp                  # (H_pix, W_pix)
        x_min, x_max, y_min, y_max = type(self)._cloak_bbox
        H_pix, W_pix = canvas.shape
        void_ratio = type(self)._void_ratio
        simp_p = type(self)._simp_p
        binarize = type(self)._binarize
        xi_fn = type(self).__dict__["_xi_fn"]

        inv_dx = 1.0 / (x_max - x_min)
        inv_dy = 1.0 / (y_max - y_min)

        def _occ_at(x):
            # Canvas tiled y-up (top of image = high y), so flip y for the row.
            x_norm = (x[0] - x_min) * inv_dx
            y_norm = (x[1] - y_min) * inv_dy
            col = jnp.clip((x_norm * W_pix).astype(jnp.int32), 0, W_pix - 1)
            row = jnp.clip(((1.0 - y_norm) * H_pix).astype(jnp.int32), 0, H_pix - 1)
            return canvas[row, col]

        def _frac(occ):
            # SIMP-soft material fraction in [void_ratio, 1]; differentiable in occ.
            if binarize:
                occ = jnp.where(occ > 0.5, 1.0, 0.0)
            occ = jnp.clip(occ, 0.0, 1.0)
            return void_ratio + (1.0 - void_ratio) * occ ** simp_p

        def _C_eff_pt(x):
            in_clk = geo.in_cloak(x)
            C_pixel = _frac(_occ_at(x)) * C0
            return jnp.where(in_clk, C_pixel, C0)

        def _rho_eff_pt(x):
            in_clk = geo.in_cloak(x)
            # Density is linear in occupancy (no SIMP power), matching ersatz void.
            occ = _occ_at(x)
            if binarize:
                occ = jnp.where(occ > 0.5, 1.0, 0.0)
            occ = jnp.clip(occ, 0.0, 1.0)
            rho_pixel = (void_ratio + (1.0 - void_ratio) * occ) * rho0
            return jnp.where(in_clk, rho_pixel, rho0)

        xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)
        self._xi_qp = xi_qp
        self.internal_vars = [
            jax.vmap(jax.vmap(_C_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_eff_pt))(self.physical_quad_points),
            xi_qp,
        ]

    def set_params(self, _params):
        pass


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
