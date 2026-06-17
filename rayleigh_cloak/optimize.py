"""Cell-based material optimisation for the Rayleigh-wave cloak.

Uses JAX-FEM's ``ad_wrapper`` for implicit-adjoint differentiation through the
FEM solve.  The optimisation loop uses a simple Adam implementation (no external
dependency on optax).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.cells import CellDecomposition


# ── loss components ──────────────────────────────────────────────────


def cloaking_loss(
    u_cloak: jnp.ndarray,
    u_ref_boundary: jnp.ndarray,
    boundary_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Relative L2 displacement difference on the right physical boundary.

    Returns ``||u_cloak - u_ref||^2 / ||u_ref||^2`` (scale-invariant).

    Parameters
    ----------
    u_cloak : (n_nodes, 4) solution on the cloak mesh
    u_ref_boundary : (n_boundary, 4) reference displacement at boundary nodes
        (pre-interpolated onto the cloak mesh boundary positions)
    boundary_indices : node indices into u_cloak for the boundary
    """
    diff = u_cloak[boundary_indices] - u_ref_boundary
    ref_norm_sq = jnp.sum(u_ref_boundary ** 2) + 1e-30  # avoid 0/0
    return jnp.sum(diff ** 2) / ref_norm_sq


def l2_regularization(
    params: tuple[jnp.ndarray, jnp.ndarray],
    params_init: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Relative L2 drift from initial material values.

    Each component (C, rho) is normalised by its own initial norm
    so that terms with different scales contribute equally.
    """
    cell_C, cell_rho = params
    cell_C_init, cell_rho_init = params_init
    C_norm = jnp.sum(cell_C_init ** 2) + 1e-30
    rho_norm = jnp.sum(cell_rho_init ** 2) + 1e-30
    return (jnp.sum((cell_C - cell_C_init) ** 2) / C_norm
            + jnp.sum((cell_rho - cell_rho_init) ** 2) / rho_norm)


def neighbor_regularization(
    params: tuple[jnp.ndarray, jnp.ndarray],
    neighbor_pairs: jnp.ndarray,
) -> jnp.ndarray:
    """Relative smoothness penalty between adjacent cells.

    Normalised by the mean squared magnitude so the penalty is
    scale-invariant w.r.t. absolute material values.
    """
    cell_C, cell_rho = params
    if neighbor_pairs.shape[0] == 0:
        return jnp.float32(0.0)
    C_i = cell_C[neighbor_pairs[:, 0]]
    C_j = cell_C[neighbor_pairs[:, 1]]
    rho_i = cell_rho[neighbor_pairs[:, 0]]
    rho_j = cell_rho[neighbor_pairs[:, 1]]
    C_scale = (jnp.mean(C_i ** 2) + jnp.mean(C_j ** 2)) / 2.0 + 1e-30
    rho_scale = (jnp.mean(rho_i ** 2) + jnp.mean(rho_j ** 2)) / 2.0 + 1e-30
    return (jnp.mean((C_i - C_j) ** 2) / C_scale
            + jnp.mean((rho_i - rho_j) ** 2) / rho_scale)


def total_loss(
    params: tuple[jnp.ndarray, jnp.ndarray],
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    fwd_pred,
    u_ref_boundary: jnp.ndarray,
    boundary_indices: jnp.ndarray,
    neighbor_pairs: jnp.ndarray,
    lambda_l2: float,
    lambda_neighbor: float,
    loss_fn=None,
) -> jnp.ndarray:
    """Combined cloaking + regularisation loss."""
    if loss_fn is None:
        loss_fn = cloaking_loss
    sol_list = fwd_pred(params)
    u_cloak = sol_list[0]

    L_cloak = loss_fn(u_cloak, u_ref_boundary, boundary_indices)
    L_l2 = l2_regularization(params, params_init)
    L_nb = neighbor_regularization(params, neighbor_pairs)

    return L_cloak + lambda_l2 * L_l2 + lambda_neighbor * L_nb



# ── simple Adam optimiser ────────────────────────────────────────────


@dataclass
class AdamState:
    m: Any = None  # first moment (pytree)
    v: Any = None  # second moment (pytree)
    t: int = 0


def adam_init(params) -> AdamState:
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return AdamState(m=m, v=v, t=0)


def adam_update(
    grads, state: AdamState, lr: float = 1e-3, beta1: float = 0.9,
    beta2: float = 0.999, eps: float = 1e-8,
) -> tuple:
    """Return (updates, new_state)."""
    t = state.t + 1
    m = jax.tree.map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, state.m, grads)
    v = jax.tree.map(lambda v_, g: beta2 * v_ + (1 - beta2) * g ** 2, state.v, grads)
    m_hat = jax.tree.map(lambda m_: m_ / (1 - beta1 ** t), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1 - beta2 ** t), v)
    updates = jax.tree.map(lambda mh, vh: -lr * mh / (jnp.sqrt(vh) + eps), m_hat, v_hat)
    return updates, AdamState(m=m, v=v, t=t)


# ── optimisation loop ────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """Result of a cell-based material optimisation run."""
    params: tuple[jnp.ndarray, jnp.ndarray]
    loss_history: list[float] = field(default_factory=list)
    cloak_history: list[float] = field(default_factory=list)
    l2_history: list[float] = field(default_factory=list)
    neighbor_history: list[float] = field(default_factory=list)


def get_right_boundary_indices(
    mesh_points: np.ndarray,
    x_right: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return node indices on the right physical boundary (excluding PML)."""
    return np.where(np.abs(mesh_points[:, 0] - x_right) < tol)[0]


def get_circular_boundary_indices(
    mesh_points: np.ndarray,
    x_c: float,
    y_c: float,
    r_measure: float,
    tol_frac: float = 0.05,
) -> np.ndarray:
    """Return node indices near a circle of radius *r_measure* around (x_c, y_c).

    Selects nodes within ``tol_frac * r_measure`` of the measurement circle.
    Useful for cloaking distortion metrics that are less sensitive to
    shadow-side amplitude variations than a single planar boundary.
    """
    pts = np.asarray(mesh_points)
    dx = pts[:, 0] - x_c
    dy = pts[:, 1] - y_c
    r = np.sqrt(dx**2 + dy**2)
    tol = tol_frac * r_measure
    return np.where(np.abs(r - r_measure) < tol)[0]


def get_all_physical_boundary_indices(
    mesh_points: np.ndarray,
    x_off: float,
    y_off: float,
    W: float,
    H: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return node indices on all four physical boundaries (excluding PML)."""
    pts = np.asarray(mesh_points)
    left = np.abs(pts[:, 0] - x_off) < tol
    right = np.abs(pts[:, 0] - (x_off + W)) < tol
    bottom = np.abs(pts[:, 1] - y_off) < tol
    top = np.abs(pts[:, 1] - (y_off + H)) < tol
    return np.where(left | right | bottom | top)[0]


def get_outside_cloak_indices(
    mesh_points: np.ndarray,
    geometry,
    x_off: float,
    y_off: float,
    W: float,
    H: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return node indices in the physical domain but outside the cloak region.

    Selects nodes that are (a) inside the physical domain (excluding PML) and
    (b) not inside the cloak or defect region.  Useful for measuring cloaking
    quality over the entire exterior field.
    """
    pts = np.asarray(mesh_points)
    # Physical domain mask
    in_phys = (
        (pts[:, 0] >= x_off - tol) & (pts[:, 0] <= x_off + W + tol) &
        (pts[:, 1] >= y_off - tol) & (pts[:, 1] <= y_off + H + tol)
    )
    # Vectorised cloak/defect membership using JAX vmap
    import jax
    import jax.numpy as jnp
    pts_jnp = jnp.array(pts)
    in_c = np.asarray(jax.vmap(geometry.in_cloak)(pts_jnp))
    in_d = np.asarray(jax.vmap(geometry.in_defect)(pts_jnp))
    return np.where(in_phys & ~in_c & ~in_d)[0]


def get_top_surface_beyond_cloak_indices(
    mesh_points: np.ndarray,
    geometry,
    y_top: float,
    x_left: float,
    x_right: float,
    tol: float = 1e-6,
) -> np.ndarray:
    """Return node indices on the top surface *downstream* (right) of the cloak.

    Because the cloak outer boundary starts at the free surface (the outer
    triangle base lies on y_top), ``geometry.in_cloak`` evaluated directly on
    surface nodes correctly identifies the full cloak footprint.  No column
    probing is required.

    Parameters
    ----------
    mesh_points : (n_nodes, 2+) mesh coordinates
    geometry : object satisfying the CloakGeometry protocol
    y_top : y-coordinate of the free surface
    x_left : left edge of the physical domain (exclude PML)
    x_right : right edge of the physical domain (exclude PML)
    tol : positional tolerance for surface detection
    """
    import jax
    import jax.numpy as jnp

    pts = np.asarray(mesh_points)
    on_top = np.abs(pts[:, 1] - y_top) < tol
    in_phys_x = (pts[:, 0] >= x_left - tol) & (pts[:, 0] <= x_right + tol)
    top_phys = np.where(on_top & in_phys_x)[0]

    if len(top_phys) == 0:
        return top_phys

    x_surface = pts[top_phys, 0]
    surface_pts = jnp.column_stack([
        jnp.array(x_surface),
        jnp.full(len(top_phys), y_top),
    ])
    above_cloak = np.asarray(jax.vmap(geometry.in_cloak)(surface_pts))
    downstream = x_surface > geometry.x_c + tol

    return top_phys[~above_cloak & downstream]


def cloaking_distortion_percent(
    u_cloak: jnp.ndarray,
    u_ref: jnp.ndarray,
    boundary_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Mean percent distortion on measurement boundaries.

    Returns ``100 * ‖u_cloak − u_ref‖₂ / ‖u_ref‖₂`` on the given nodes.
    """
    diff = u_cloak[boundary_indices] - u_ref[boundary_indices]
    ref_norm_sq = jnp.sum(u_ref[boundary_indices] ** 2) + 1e-30
    return 100.0 * jnp.sqrt(jnp.sum(diff ** 2) / ref_norm_sq)


def interpolate_ref_to_boundary(
    ref_points: np.ndarray,
    ref_u: np.ndarray,
    cloak_points: np.ndarray,
    boundary_indices: np.ndarray,
    x_right: float,
    tol: float = 1e-6,
) -> np.ndarray:
    """Interpolate reference solution onto cloak mesh boundary nodes.

    The reference and cloak meshes are different (reference has no defect
    cutout).  This function interpolates along the y-coordinate at the
    right physical boundary ``x = x_right``.

    Returns
    -------
    u_ref_boundary : (n_boundary, n_dof) array of reference displacement
        values at the cloak mesh boundary node positions.
    """
    from scipy.interpolate import interp1d

    # Reference mesh: find nodes on right boundary
    ref_mask = np.abs(ref_points[:, 0] - x_right) < tol
    ref_y = ref_points[ref_mask, 1]
    ref_u_bnd = ref_u[ref_mask]

    # Sort by y for interpolation
    order = np.argsort(ref_y)
    ref_y = ref_y[order]
    ref_u_bnd = ref_u_bnd[order]

    # Cloak mesh: y-coordinates of boundary nodes
    cloak_y = cloak_points[boundary_indices, 1]

    # Interpolate each DOF column
    n_dof = ref_u.shape[1]
    u_interp = np.zeros((len(boundary_indices), n_dof))
    for d in range(n_dof):
        f = interp1d(ref_y, ref_u_bnd[:, d], kind='linear',
                     fill_value='extrapolate')
        u_interp[:, d] = f(cloak_y)

    return u_interp


def run_optimization(
    fwd_pred,
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    u_ref_boundary: jnp.ndarray,
    boundary_indices: jnp.ndarray,
    neighbor_pairs: jnp.ndarray,
    n_iters: int = 100,
    lr: float = 1e-3,
    lambda_l2: float = 1e-4,
    lambda_neighbor: float = 1e-3,
    plot_callback=None,
    plot_every: int = 1,
    step_callback=None,
    loss_fn=None,
) -> OptimizationResult:
    """Run the optimisation loop.

    Parameters
    ----------
    fwd_pred : callable
        Differentiable forward prediction from ``ad_wrapper``.
    params_init : (cell_C_flat, cell_rho)
    u_ref_boundary : (n_boundary, 4) reference displacement at boundary nodes,
        pre-interpolated onto the cloak mesh boundary positions.
    boundary_indices : node indices for cloaking loss
    neighbor_pairs : (n_pairs, 2) array of adjacent cell indices
    plot_callback : optional callable(step, u)
        Called with the step index and solution array.
    plot_every : int
        Plot every N steps (1 = every step). Costs one extra forward
        solve per plotted step.
    step_callback : optional callable(step, total, cloak, l2, neighbor, params)
        Called after each step with loss components and current params.
        Useful for incremental CSV logging and checkpointing.
    loss_fn : optional callable(u_cloak, u_ref, indices) -> scalar
        JAX-traceable cloaking loss. Defaults to ``cloaking_loss`` (relative L2).
    """
    params = jax.tree.map(jnp.copy, params_init)
    opt_state = adam_init(params)
    loss_history: list[float] = []
    cloak_history: list[float] = []
    l2_history: list[float] = []
    neighbor_history: list[float] = []

    neighbor_pairs_jnp = jnp.array(neighbor_pairs)
    boundary_indices_jnp = jnp.array(boundary_indices)

    loss_and_grad = jax.value_and_grad(
        lambda p: total_loss(
            p, params_init, fwd_pred, u_ref_boundary,
            boundary_indices_jnp, neighbor_pairs_jnp,
            lambda_l2, lambda_neighbor, loss_fn,
        )
    )

    for step in range(n_iters):
        loss_val, grads = loss_and_grad(params)
        loss_val_float = float(loss_val)
        loss_history.append(loss_val_float)

        # Compute regularisation terms cheaply (no forward pass needed)
        L_l2 = float(l2_regularization(params, params_init))
        L_nb = float(neighbor_regularization(params, neighbor_pairs_jnp))
        L_cloak = loss_val_float - lambda_l2 * L_l2 - lambda_neighbor * L_nb
        cloak_history.append(L_cloak)
        l2_history.append(L_l2)
        neighbor_history.append(L_nb)
        print(
            f"  Step {step:4d} | total = {loss_val_float:.4e}"
            f"  cloak = {L_cloak:.4e}"
            f"  cloak_pct = {np.sqrt(L_cloak) * 100:.2e}"  # take sq.root and multiply by 100 for percent
            f"  L2_reg = {L_l2:.4e}"
            f"  neighbor = {L_nb:.4e}"
        )

        if step_callback is not None:
            step_callback(step, loss_val_float, L_cloak, L_l2, L_nb, params)

        # Plot current solution (extra forward pass, only every N steps)
        if plot_callback is not None and step % plot_every == 0:
            sol_list = fwd_pred(params)
            plot_callback(step, np.asarray(sol_list[0]))

        updates, opt_state = adam_update(grads, opt_state, lr=lr)
        params = jax.tree.map(lambda p, u: p + u, params, updates)

    # Always plot the final state
    if plot_callback is not None:
        sol_list = fwd_pred(params)
        plot_callback(n_iters, np.asarray(sol_list[0]))

    return OptimizationResult(
        params=params,
        loss_history=loss_history,
        cloak_history=cloak_history,
        l2_history=l2_history,
        neighbor_history=neighbor_history,
    )
