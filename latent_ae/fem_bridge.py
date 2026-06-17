"""Bridge between the PyTorch decoder and the JAX FEM multi-frequency solver.

``build_freq_targets`` constructs one per-frequency FEM closure by reusing the
setup logic from ``rayleigh_cloak/solver.py`` (shared mesh, per-f reference
solve, per-f FEM problem, loss target, ``ad_wrapper``). The closure signature
is ``(C_flat, rho_flat) -> (loss, gC, grho)`` — JAX ``value_and_grad`` over
both cell arrays.

``MultiFreqFEMLoss`` is a ``torch.autograd.Function`` that calls those closures
inside a thread pool and returns per-frequency losses as a torch tensor, with a
standard backward that scales the saved JAX gradients by ``grad_output``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax_fem.solver import ad_wrapper

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.loss import resolve_loss_target
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.neural_reparam import FreqTarget
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _make_config_at_fstar, _petsc_opts


@dataclass
class FEMContext:
    """Everything needed to evaluate a (C, rho) design across frequencies."""

    freq_targets: list[FreqTarget]
    params_init: tuple[jnp.ndarray, jnp.ndarray]
    cloak_mask_flat: jnp.ndarray            # (n_cells,) 0/1 float
    n_x: int
    n_y: int
    n_C_params: int
    value_and_grad_fns: list[Any]           # one per freq
    pool: ThreadPoolExecutor

    def shutdown(self):
        self.pool.shutdown(wait=False)


def _create_geometry(cfg: SimulationConfig, params: DerivedParams):
    # Matches rayleigh_cloak/solver.py:_create_geometry — triangular is what the
    # hi-fi dataset uses. Extend here if other geometries are needed.
    if cfg.geometry_type == "triangular":
        return TriangularCloakGeometry.from_params(params)
    raise ValueError(
        f"latent_ae.fem_bridge: geometry_type={cfg.geometry_type!r} not wired. "
        f"Only 'triangular' is supported."
    )


def build_freq_targets(
    base_config: SimulationConfig,
    f_stars: list[float],
    max_workers: int | None = None,
) -> FEMContext:
    """Build per-frequency FEM closures matching the hi-fi dataset geometry.

    Mirrors ``rayleigh_cloak.solver.solve_optimization_neural`` up to the point
    where ``FreqTarget``s are built, then exits without running any optimization.
    """
    params = DerivedParams.from_config(base_config)
    geometry = _create_geometry(base_config, params)

    full_mesh = generate_mesh_full(base_config, params, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

    cell_decomp = CellDecomposition(
        geometry, base_config.cells.n_x, base_config.cells.n_y,
    )
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=base_config.cells.n_C_params,
        symmetrize_init=base_config.cells.symmetrize_init,
    )
    params_init = cell_mat.get_initial_params()      # (C_flat, rho) as JAX arrays

    # Cloak mask as a (n_cells,) float — layout matches idx = ix*n_y+iy (cells.py).
    cloak_mask_flat = jnp.asarray(cell_decomp.cloak_mask.astype(np.float32))

    solver_opts = _petsc_opts(base_config)

    from rayleigh_cloak.solver import solve_reference  # local import to avoid cycles

    freq_targets: list[FreqTarget] = []
    for f_star in f_stars:
        cfg_f = _make_config_at_fstar(base_config, float(f_star))
        dp_f = DerivedParams.from_config(cfg_f)

        ref_f = solve_reference(cfg_f, mesh=full_mesh)
        problem_f = build_problem(cloak_mesh, cfg_f, dp_f, geometry, cell_decomp)
        fwd_pred_f = ad_wrapper(problem_f, solver_opts, solver_opts)

        indices_f, u_ref_f, loss_fn_f = resolve_loss_target(
            base_config.loss.type, np.asarray(cloak_mesh.points), geometry,
            dp_f, kept_nodes, ref_f.u, loss_cfg=base_config.loss,
        )

        freq_targets.append(FreqTarget(
            f_star=float(f_star),
            weight=1.0,
            fwd_pred=fwd_pred_f,
            u_ref_boundary=u_ref_f,
            boundary_indices=jnp.array(indices_f),
            loss_fn=loss_fn_f,
        ))

    value_and_grad_fns = [_make_param_loss_and_grad(ft) for ft in freq_targets]

    n_freq = len(freq_targets)
    workers = max_workers if (max_workers is not None and max_workers > 0) else n_freq
    pool = ThreadPoolExecutor(max_workers=workers)

    return FEMContext(
        freq_targets=freq_targets,
        params_init=params_init,
        cloak_mask_flat=cloak_mask_flat,
        n_x=base_config.cells.n_x,
        n_y=base_config.cells.n_y,
        n_C_params=base_config.cells.n_C_params,
        value_and_grad_fns=value_and_grad_fns,
        pool=pool,
    )


def _make_param_loss_and_grad(ft: FreqTarget):
    """value_and_grad over (C_flat, rho) for a single frequency.

    Returns (loss_scalar, grad_C_flat, grad_rho) as JAX arrays.
    """
    _fwd = ft.fwd_pred
    _u_ref = ft.u_ref_boundary
    _idx = ft.boundary_indices
    _lfn = ft.loss_fn

    def _loss(C_flat, rho):
        sol_list = _fwd((C_flat, rho))
        return _lfn(sol_list[0], _u_ref, _idx)

    return jax.value_and_grad(_loss, argnums=(0, 1))


def _blend_with_init(
    C_flat: np.ndarray,
    rho_flat: np.ndarray,
    ctx: FEMContext,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Use decoded values inside cloak cells and ``params_init`` outside."""
    C_init, rho_init = ctx.params_init
    mask = ctx.cloak_mask_flat
    C_blend = mask[:, None] * jnp.asarray(C_flat) + (1.0 - mask[:, None]) * C_init
    rho_blend = mask * jnp.asarray(rho_flat) + (1.0 - mask) * rho_init
    return C_blend, rho_blend


class MultiFreqFEMLoss(torch.autograd.Function):
    """per-freq FEM value_and_grad wrapped as a torch op.

    Forward:   (C_grid, rho_grid) -> (n_freq,) torch tensor of per-f losses.
    Backward:  scales saved per-f grads by grad_output and sums.
    """

    @staticmethod
    def forward(ctx_t, C_grid: torch.Tensor, rho_grid: torch.Tensor, fem_ctx: FEMContext):
        """
        Args:
            C_grid: (X, Y, P) — decoder output for one design.
            rho_grid: (X, Y) — decoder output for one design.
            fem_ctx: FEMContext from ``build_freq_targets``.
        Returns:
            (n_freq,) torch tensor of per-frequency losses.
        """
        assert C_grid.ndim == 3 and rho_grid.ndim == 2, "bridge expects single design"
        assert C_grid.shape[:2] == rho_grid.shape, "(X, Y) mismatch"

        C_np = C_grid.detach().cpu().numpy().astype(np.float64)
        rho_np = rho_grid.detach().cpu().numpy().astype(np.float64)
        C_flat = C_np.reshape(-1, C_np.shape[-1])      # (n_cells, P) — idx = ix*n_y+iy
        rho_flat = rho_np.reshape(-1)                  # (n_cells,)

        C_blend, rho_blend = _blend_with_init(C_flat, rho_flat, fem_ctx)

        def work(vg_fn):
            return vg_fn(C_blend, rho_blend)

        results = list(fem_ctx.pool.map(work, fem_ctx.value_and_grad_fns))

        losses = np.array([float(r[0]) for r in results], dtype=np.float64)
        gC_stack = np.stack(
            [np.asarray(r[1][0]) for r in results], axis=0,
        )    # (n_freq, n_cells, P) — grad w.r.t. C_blend
        grho_stack = np.stack(
            [np.asarray(r[1][1]) for r in results], axis=0,
        )    # (n_freq, n_cells)

        # Chain through the mask: d(C_blend)/d(C_flat) = mask, so multiply.
        mask_flat = np.asarray(fem_ctx.cloak_mask_flat)                  # (n_cells,)
        gC_stack = gC_stack * mask_flat[None, :, None]
        grho_stack = grho_stack * mask_flat[None, :]

        ctx_t.save_for_backward(
            torch.from_numpy(gC_stack).to(C_grid.dtype).to(C_grid.device),
            torch.from_numpy(grho_stack).to(rho_grid.dtype).to(rho_grid.device),
        )
        ctx_t.grid_shape = C_grid.shape
        ctx_t.rho_shape = rho_grid.shape
        return torch.from_numpy(losses).to(C_grid.dtype).to(C_grid.device)

    @staticmethod
    def backward(ctx_t, grad_output):
        gC_stack, grho_stack = ctx_t.saved_tensors   # (n_freq, n_cells, P), (n_freq, n_cells)

        # Sum per-freq gradients weighted by grad_output.
        gC_total = (grad_output[:, None, None] * gC_stack).sum(dim=0)   # (n_cells, P)
        grho_total = (grad_output[:, None] * grho_stack).sum(dim=0)      # (n_cells,)

        gC_grid = gC_total.reshape(ctx_t.grid_shape)
        grho_grid = grho_total.reshape(ctx_t.rho_shape)
        return gC_grid, grho_grid, None
