"""Cloaking loss metrics.

Provides :func:`compute_cloaking_loss` which measures how well the cloaked
field matches the reference field on all physical boundaries and across the
full physical domain outside the cloak.

Also provides the transmitted displacement ratio metric from
Chatzopoulos et al. (2023), Fig 2(k):  <|u_cloak|> / <|u_ref|>
on the free surface beyond the cloaked region.  Available both as a
NumPy evaluation metric (:func:`transmitted_displacement_ratio`) and a
JAX-traceable loss (:func:`transmission_loss`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.optimize import (
    get_outside_cloak_indices,
    get_right_boundary_indices,
    get_all_physical_boundary_indices,
    get_top_surface_beyond_cloak_indices,
)


@dataclass
class CloakingLoss:
    dist_boundary: float  # distortion % on all four physical boundaries
    dist_right: float     # distortion % on right physical boundary only
    dist_outside: float   # distortion % over all physical nodes outside cloak
    n_boundary: int       # number of nodes on all physical boundaries
    n_right: int          # number of nodes on right boundary
    n_outside: int        # number of nodes outside cloak


def _relative_l2(u_cloak: np.ndarray, u_ref: np.ndarray) -> float:
    """Relative L2 displacement difference: ||u_cloak - u_ref||^2 / ||u_ref||^2."""
    diff = u_cloak - u_ref
    ref_norm_sq = float(np.sum(u_ref ** 2)) + 1e-30
    return float(np.sum(diff ** 2) / ref_norm_sq)


def _distortion_pct(u_cloak: np.ndarray, u_ref: np.ndarray) -> float:
    """100 * ||u_cloak - u_ref|| / ||u_ref||."""
    return 100.0 * np.sqrt(_relative_l2(u_cloak, u_ref))


# ── Transmitted displacement ratio ──────────────────────────────────


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    """Total displacement magnitude per node: sqrt(|ux|^2 + |uy|^2).

    Parameters
    ----------
    u : (n_nodes, 4) with DOFs [Re(ux), Re(uy), Im(ux), Im(uy)]
    """
    return np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2 + u[:, 3]**2)


def transmitted_displacement_ratio(
    u_case: np.ndarray,
    u_ref: np.ndarray,
    case_surface_idx: np.ndarray,
    ref_surface_idx: np.ndarray,
) -> float:
    """<|u_case|> / <|u_ref|> on the free surface beyond the cloak.

    This is the metric from Chatzopoulos et al. (2023), Fig 2(k).
    A perfect cloak yields a ratio of 1.0.

    Parameters
    ----------
    u_case : (n_nodes_case, 4) solution on the case mesh (cloak/obstacle)
    u_ref : (n_nodes_ref, 4) solution on the reference mesh
    case_surface_idx : node indices into u_case for the evaluation surface
    ref_surface_idx : corresponding node indices into u_ref
    """
    mag_case = displacement_magnitude(u_case[case_surface_idx])
    mag_ref = displacement_magnitude(u_ref[ref_surface_idx])
    return float(np.mean(mag_case)) / (float(np.mean(mag_ref)) + 1e-30)


# ── Mesh-independent fixed-position surface metric ──────────────────
#
# The legacy node-based metric averages |u| at whichever surface mesh nodes
# happen to fall in the evaluation region. With unstructured triangular
# meshes, the *number* and *positions* of those nodes change with each
# mesh refinement, so the metric itself is mesh-dependent — refining the
# mesh re-samples a different observable. The functions below sidestep
# that by evaluating |u| at a set of *fixed* x-positions, interpolated
# from each mesh's surface nodes. Pass ``cfg.loss.n_eval_points > 0`` to
# opt in; the legacy mechanism is preserved for ``n_eval_points == 0``.


def make_fixed_surface_eval_points(
    geometry,
    params,
    n_points: int,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return ``M`` x-coordinates evenly spaced across the free surface beyond
    the cloak footprint, optionally jittered by Gaussian noise.

    The endpoints of ``[x_off, x_off+W]`` are excluded (avoiding domain corners),
    and points whose ``(x, y_top)`` lies inside the defect/cloak footprint are
    dropped — those positions have no free surface in the cloak mesh. The
    Gaussian noise is added *before* the in-defect filter so that two runs
    with the same seed produce identical x-arrays (deterministic).
    """
    x_left = params.x_off
    x_right = params.x_off + params.W
    # ``n_points + 2`` then trim endpoints, so we get exactly n_points interior
    # samples evenly spaced across the open interval.
    xs = np.linspace(x_left, x_right, n_points + 2)[1:-1]
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        xs = xs + rng.normal(0.0, float(noise_sigma), size=xs.shape)
        xs = np.clip(xs, x_left, x_right)

    # Drop fixed positions inside the defect footprint (no free surface there).
    y_top = params.y_top
    keep = []
    for x in xs:
        pt = jnp.array([float(x), float(y_top) - 1e-6])
        if not bool(geometry.in_defect(pt)) and not bool(geometry.in_cloak(pt)):
            keep.append(float(x))
    return np.asarray(keep, dtype=np.float64)


def _surface_mag_along_x(
    u: np.ndarray,
    mesh,
    y_top: float,
    atol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted-x, |u|-at-those-nodes) for nodes on the top surface.

    ``atol`` defaults to a small relative tolerance based on mesh spread.
    """
    pts = np.asarray(mesh.points)
    if atol is None:
        atol = 1e-6 * max(1.0, float(np.ptp(pts[:, 1])))
    is_top = np.isclose(pts[:, 1], y_top, atol=atol)
    if not np.any(is_top):
        raise RuntimeError(
            "No mesh nodes found on the top surface y == y_top "
            "(this should be impossible given gmsh edge embedding)."
        )
    nodes = np.where(is_top)[0]
    xs = pts[nodes, 0]
    perm = np.argsort(xs)
    xs_sorted = xs[perm]
    mag = displacement_magnitude(u[nodes[perm]])
    return xs_sorted, mag


def transmitted_displacement_ratio_fixed(
    u_case: np.ndarray,
    u_ref: np.ndarray,
    case_mesh,
    ref_mesh,
    x_positions: np.ndarray,
    y_top: float,
) -> float:
    """Mesh-independent variant of :func:`transmitted_displacement_ratio`.

    Linearly interpolates ``|u_case|`` and ``|u_ref|`` from each mesh's top-
    surface nodes onto ``x_positions`` (which the caller has already filtered
    to lie outside the cloak footprint), then returns the ratio of unweighted
    means. Because ``x_positions`` is shared across all sweep points, the
    metric becomes a stable functional of the mesh-converged solution.
    """
    case_xs, case_mag = _surface_mag_along_x(u_case, case_mesh, y_top)
    ref_xs, ref_mag = _surface_mag_along_x(u_ref, ref_mesh, y_top)
    case_at_x = np.interp(x_positions, case_xs, case_mag)
    ref_at_x = np.interp(x_positions, ref_xs, ref_mag)
    return float(np.mean(case_at_x)) / (float(np.mean(ref_at_x)) + 1e-30)


def transmission_loss(
    u_cloak: jnp.ndarray,
    u_ref_surface: jnp.ndarray,
    surface_indices: jnp.ndarray,
) -> jnp.ndarray:
    """JAX-traceable loss: squared deviation of transmission ratio from 1.

    Computes ``(ratio - 1)^2`` where ``ratio = <|u_cloak|> / <|u_ref|>``
    on the surface nodes beyond the cloak.  Minimising drives the
    transmitted wave amplitude toward the reference.

    Parameters
    ----------
    u_cloak : (n_nodes, 4) cloaked solution
    u_ref_surface : (n_surface, 4) reference displacement at surface nodes
        (pre-indexed from the full reference solution)
    surface_indices : node indices into u_cloak for the evaluation surface
    """
    u_s = u_cloak[surface_indices]
    mag_cloak = jnp.sqrt(
        u_s[:, 0]**2 + u_s[:, 1]**2 + u_s[:, 2]**2 + u_s[:, 3]**2
    )
    mag_ref = jnp.sqrt(
        u_ref_surface[:, 0]**2 + u_ref_surface[:, 1]**2
        + u_ref_surface[:, 2]**2 + u_ref_surface[:, 3]**2
    )
    ratio = jnp.mean(mag_cloak) / (jnp.mean(mag_ref) + 1e-30)
    return (ratio - 1.0) ** 2


# ── Loss resolution from config ────────────────────────────────────


def make_fixed_depth_eval_points(
    geometry,
    params,
    depth: float,
    n_points: int,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Return ``(xs, y_depth)``: x-positions on a horizontal line at depth
    ``depth`` below the free surface, optionally jittered by Gaussian noise.

    Positions whose ``(x, y_depth)`` falls inside the cloak or defect footprint
    are dropped — those are either absent from the cloak mesh (defect) or
    measure the cloak's interior (cloak), neither of which reflects the
    cloak's external invisibility. The kept positions are exactly those where
    a perfect cloak would yield ``u_cloak == u_ref``.

    Endpoints of ``[x_off, x_off + W]`` are excluded to avoid the right PML
    interface and the left edge of the physical domain.
    """
    if depth <= 0.0:
        raise ValueError(
            f"depth must be > 0 (got {depth!r}); use loss.type='top_surface' "
            f"for the free surface."
        )
    if depth >= float(params.H):
        raise ValueError(
            f"depth ({depth!r}) must be < physical height H ({params.H!r}); "
            f"deeper lines fall in/below the bottom PML, where the field is "
            f"attenuated and the loss is meaningless."
        )
    x_left = params.x_off
    x_right = params.x_off + params.W
    y_depth = float(params.y_top) - float(depth)
    xs = np.linspace(x_left, x_right, n_points + 2)[1:-1]
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        xs = xs + rng.normal(0.0, float(noise_sigma), size=xs.shape)
        xs = np.clip(xs, x_left, x_right)

    keep = []
    for x in xs:
        pt = jnp.array([float(x), float(y_depth)])
        if not bool(geometry.in_defect(pt)) and not bool(geometry.in_cloak(pt)):
            keep.append(float(x))
    if not keep:
        raise RuntimeError(
            f"All {n_points} depth-line eval points at depth={depth!r} fall "
            f"inside the cloak/defect footprint. Increase n_eval_points, "
            f"widen the domain, or pick a depth below the cloak (> b)."
        )
    return np.asarray(keep, dtype=np.float64), y_depth


def find_nearest_node_indices(
    mesh_points: np.ndarray,
    eval_xs: np.ndarray,
    y_target: float,
) -> np.ndarray:
    """For each ``x_eval``, return the cloak-mesh node index closest to
    ``(x_eval, y_target)`` in Euclidean distance.

    Used by the ``"depth_line"`` loss, which evaluates u on an interior
    horizontal line. Unlike the embedded top-surface path, the cloak mesh is
    not built with these positions forced — accuracy depends on local mesh
    density at the target depth (control via
    ``MeshConfig.refinement_factor_outside``).
    """
    pts = np.asarray(mesh_points)
    out = np.empty(len(eval_xs), dtype=np.int64)
    for i, x in enumerate(eval_xs):
        dx = pts[:, 0] - float(x)
        dy = pts[:, 1] - float(y_target)
        out[i] = int(np.argmin(dx * dx + dy * dy))
    return out


def find_embedded_eval_node_indices(
    mesh_points: np.ndarray,
    eval_xs: np.ndarray,
    y_top: float,
    tol: float = 1e-7,
) -> np.ndarray:
    """Map fixed-eval x-positions to mesh-node indices.

    Assumes the mesh was built with these positions embedded as forced 1-D
    nodes on the top edge (see ``mesh._embed_top_surface_eval_points``).
    Each ``(eval_x, y_top)`` should match exactly one node within ``tol``.
    Raises if any eval x has no matching node — that means the embedding
    silently dropped a point and the metric would be silently misaligned.
    """
    pts = np.asarray(mesh_points)
    on_top = np.where(np.abs(pts[:, 1] - y_top) < tol)[0]
    if len(on_top) == 0:
        raise RuntimeError(
            f"No mesh nodes on y == y_top={y_top!r}; cannot resolve fixed "
            f"top-surface eval points."
        )
    top_xs = pts[on_top, 0]
    out = np.empty(len(eval_xs), dtype=np.int64)
    for i, x in enumerate(eval_xs):
        d = np.abs(top_xs - x)
        j = int(np.argmin(d))
        if d[j] > tol:
            raise RuntimeError(
                f"Fixed eval x={x!r} has no matching mesh node within tol="
                f"{tol!r} (closest is {top_xs[j]!r}, dist={d[j]!r}). The "
                f"mesh was likely generated without embedding the eval "
                f"points — check that loss.n_eval_points was set at "
                f"mesh-build time."
            )
        out[i] = on_top[j]
    return out


def resolve_loss_target(
    loss_type: str,
    mesh_points: np.ndarray,
    geometry,
    params,
    kept_nodes: np.ndarray,
    u_ref: np.ndarray,
    loss_cfg=None,
):
    """Resolve a loss type string to node indices, reference data, and loss fn.

    Parameters
    ----------
    loss_cfg : LossConfig, optional
        When provided and ``loss_type == "top_surface"``, ``loss_cfg.
        n_eval_points > 0`` switches the loss to use the *fixed-position*
        eval nodes embedded in the mesh by ``mesh._embed_top_surface_eval_
        points``. The legacy ``get_top_surface_beyond_cloak_indices`` path
        (all downstream surface nodes) is used when ``loss_cfg`` is None or
        ``n_eval_points == 0``.

        Required when ``loss_type == "depth_line"``: the depth and number of
        eval points are read from ``loss_cfg.depth`` and ``loss_cfg.
        n_eval_points``. Nodes are selected by nearest-neighbour to the
        ``(x, y_top - depth)`` positions (no mesh embedding).

    Returns
    -------
    indices : np.ndarray
        Node indices into the cloak mesh for loss evaluation.
    u_ref_at_nodes : jnp.ndarray
        Reference displacement at those nodes (mapped via ``kept_nodes``).
    loss_fn : callable (u_cloak, u_ref_nodes, indices) -> scalar
        JAX-traceable loss function with the same signature as
        ``cloaking_loss`` / ``transmission_loss``.
    """
    from rayleigh_cloak.optimize import cloaking_loss

    pts = np.asarray(mesh_points)

    if loss_type == "right_boundary":
        x_right = params.x_off + params.W
        indices = get_right_boundary_indices(pts, x_right)
        loss_fn = cloaking_loss
    elif loss_type == "top_surface":
        n_eval = int(loss_cfg.n_eval_points) if loss_cfg is not None else 0
        if n_eval > 0:
            eval_xs = make_fixed_surface_eval_points(
                geometry, params, n_eval,
                noise_sigma=float(loss_cfg.eval_noise_sigma),
                seed=int(loss_cfg.eval_noise_seed),
            )
            indices = find_embedded_eval_node_indices(
                pts, eval_xs, params.y_top,
            )
        else:
            indices = get_top_surface_beyond_cloak_indices(
                pts, geometry, params.y_top,
                params.x_off, params.x_off + params.W,
            )
        loss_fn = transmission_loss
    elif loss_type == "outside_cloak":
        indices = get_outside_cloak_indices(
            pts, geometry,
            params.x_off, params.y_off, params.W, params.H,
        )
        loss_fn = cloaking_loss
    elif loss_type == "depth_line":
        if loss_cfg is None:
            raise ValueError("loss.type='depth_line' requires a LossConfig.")
        n_eval = int(loss_cfg.n_eval_points)
        if n_eval <= 0:
            raise ValueError(
                "loss.type='depth_line' requires loss.n_eval_points > 0."
            )
        eval_xs, y_target = make_fixed_depth_eval_points(
            geometry, params,
            depth=float(loss_cfg.depth),
            n_points=n_eval,
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed),
        )
        indices = find_nearest_node_indices(pts, eval_xs, y_target)
        loss_fn = cloaking_loss
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type!r}. Choose from "
            f"'right_boundary', 'top_surface', 'outside_cloak', 'depth_line'."
        )

    u_ref_at_nodes = jnp.array(u_ref[kept_nodes[indices]])
    return indices, u_ref_at_nodes, loss_fn


def compute_cloaking_loss(
    cloak_result,
    ref_result,
    geometry,
) -> CloakingLoss:
    """Compute cloaking distortion metrics.

    Parameters
    ----------
    cloak_result:
        ``SolutionResult`` from the cloaked simulation.  Must have
        ``.mesh``, ``.u``, ``.params``, and ``.kept_nodes`` attributes.
    ref_result:
        ``SolutionResult`` from the reference (no-cloak) simulation on the
        shared full mesh.  Must have ``.u``.
    geometry:
        Cloak geometry object exposing ``in_cloak()`` / ``in_defect()``.
    """
    params = cloak_result.params
    kept_nodes = cloak_result.kept_nodes
    pts = np.asarray(cloak_result.mesh.points)

    # All four physical boundaries
    bnd_idx = get_all_physical_boundary_indices(
        pts, params.x_off, params.y_off, params.W, params.H,
    )
    u_ref_bnd = ref_result.u[kept_nodes[bnd_idx]]
    u_cloak_bnd = cloak_result.u[bnd_idx]
    dist_boundary = _distortion_pct(u_cloak_bnd, u_ref_bnd)

    # Right physical boundary only
    x_right = params.x_off + params.W
    right_idx = get_right_boundary_indices(pts, x_right)
    u_ref_right = ref_result.u[kept_nodes[right_idx]]
    u_cloak_right = cloak_result.u[right_idx]
    dist_right = _distortion_pct(u_cloak_right, u_ref_right)

    # All physical-domain nodes outside cloak
    outside_idx = get_outside_cloak_indices(
        pts, geometry,
        params.x_off, params.y_off, params.W, params.H,
    )
    u_ref_outside = ref_result.u[kept_nodes[outside_idx]]
    u_cloak_outside = cloak_result.u[outside_idx]
    dist_outside = _distortion_pct(u_cloak_outside, u_ref_outside)

    return CloakingLoss(
        dist_boundary=dist_boundary,
        dist_right=dist_right,
        dist_outside=dist_outside,
        n_boundary=len(bnd_idx),
        n_right=len(right_idx),
        n_outside=len(outside_idx),
    )
