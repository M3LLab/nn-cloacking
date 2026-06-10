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


# ── Mesh-independent area-weighted band metric ──────────────────────
#
# ``magnitude_band_integral`` averages |u| over whichever mesh nodes fall in
# the band [y_top - depth, y_top]. That is mesh-density-dependent: the surface
# refinement field grades element size within the band, and the *number* of
# nodes (hence the implicit weighting) changes with every refinement, so the
# node-mean is not a clean functional of the converged field. The functions
# below sidestep that by sampling |u| on a *fixed* (x, y) grid over the band
# (shared across all meshes) and interpolating from each mesh's own P1
# triangulation. The grid is uniform, so an unweighted mean over the kept grid
# points is an area average — the true area-weighted integral the
# ``magnitude_band_integral`` name promises, evaluated mesh-independently.


def make_band_grid_eval_points(
    geometry,
    params,
    depth: float,
    n_x: int,
    n_y: int,
    mode: str = "downstream",
    noise_sigma: float = 0.0,
    seed: int = 0,
    depth_top: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xs, ys)``: a fixed uniform grid over the band
    ``[y_top - depth, y_top - depth_top]`` for mesh-independent fixed-grid metrics.

    ``n_x`` x-positions span ``[x_off, x_off + W]`` (endpoints trimmed) crossed
    with ``n_y`` y-positions evenly spaced in ``[y_top - depth, y_top - depth_top]``.
    ``depth_top`` (default 0 → band reaches the free surface) lets the caller
    carve a sub-surface validation band that *excludes* the trained one — e.g.
    ``depth=1.0, depth_top=0.5`` is the strip just below a depth-0.5 training
    band. ``depth == depth_top`` collapses to a single horizontal line; with the
    default that is the free-surface line. ``mode`` mirrors ``band_x_filter``:
    ``"downstream"`` keeps ``x > x_c``, ``"full"`` keeps the whole physical
    x-range. Grid points inside the cloak or defect footprint are dropped (same
    exclusion as the node-based band). The grid does not depend on any mesh, so
    the resulting metric is a stable functional of the converged field. Raises
    if every point is dropped.
    """
    if depth_top < 0.0:
        raise ValueError(f"depth_top must be >= 0 (got {depth_top!r}).")
    if depth < depth_top:
        raise ValueError(
            f"depth ({depth!r}) must be >= depth_top ({depth_top!r})."
        )
    if depth >= float(params.H):
        raise ValueError(
            f"depth ({depth!r}) must be < physical height H ({params.H!r})."
        )
    if mode not in ("downstream", "full"):
        raise ValueError(f"Unknown mode {mode!r}; choose 'downstream' or 'full'.")
    if n_x < 1 or n_y < 1:
        raise ValueError(f"n_x and n_y must be >= 1 (got {n_x!r}, {n_y!r}).")

    x_left = params.x_off
    x_right = params.x_off + params.W
    xs = np.linspace(x_left, x_right, n_x + 2)[1:-1]
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        xs = xs + rng.normal(0.0, float(noise_sigma), size=xs.shape)
        xs = np.clip(xs, x_left, x_right)

    y_top = float(params.y_top)
    if depth == depth_top:
        ys = np.array([y_top - float(depth)], dtype=np.float64)
    else:
        ys = np.linspace(y_top - float(depth), y_top - float(depth_top), int(n_y))

    keep_x, keep_y = [], []
    for x in xs:
        if mode == "downstream" and not (float(x) > float(geometry.x_c)):
            continue
        for y in ys:
            pt = jnp.array([float(x), float(y)])
            if bool(geometry.in_defect(pt)) or bool(geometry.in_cloak(pt)):
                continue
            keep_x.append(float(x))
            keep_y.append(float(y))
    if not keep_x:
        raise RuntimeError(
            f"All {n_x * len(ys)} band-grid eval points fall inside the "
            f"cloak/defect footprint (depth={depth!r}, mode={mode!r}). "
            f"Increase n_x/n_y, decrease depth, or widen the domain."
        )
    return (np.asarray(keep_x, dtype=np.float64),
            np.asarray(keep_y, dtype=np.float64))


def _interp_mag_on_mesh(mesh, mag_nodal: np.ndarray, xs, ys) -> np.ndarray:
    """Linearly interpolate a nodal scalar (|u|) onto ``(xs, ys)`` using the
    mesh's own TRI3 connectivity (exact P1 interpolation of the FEM field).

    Returns a masked array: entries outside the triangulation (in the
    defect hole or beyond the convex hull) are masked.
    """
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    pts = np.asarray(mesh.points)
    cells = np.asarray(mesh.cells)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(
            f"area band metric requires a TRI3 mesh; got cells with shape "
            f"{cells.shape} ({cells.shape[1] if cells.ndim == 2 else '?'} "
            f"nodes/cell)."
        )
    tri = Triangulation(pts[:, 0], pts[:, 1], cells)
    interp = LinearTriInterpolator(tri, np.asarray(mag_nodal))
    return interp(np.asarray(xs), np.asarray(ys))


def transmitted_band_metrics_fixed(
    u_case: np.ndarray,
    u_ref: np.ndarray,
    case_mesh,
    ref_mesh,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[float, float]:
    """Mesh-independent, area-weighted analogue of the ``magnitude_band_integral``
    metrics, evaluated on the fixed grid from :func:`make_band_grid_eval_points`.

    Interpolates ``|u_case|`` and ``|u_ref|`` onto ``(xs, ys)`` via each mesh's
    P1 triangulation, then (over grid points valid in *both* meshes) returns::

        ratio_area = <|u_case|> / <|u_ref|>
        loss_area  = < (|u_case|/|u_ref| - 1)^2 >

    Because the grid is uniform and shared across all meshes, the means are area
    averages over the band (minus the cloak/defect footprint) and converge to a
    single mesh-independent limit as the field converges. Returns ``(nan, nan)``
    if no grid point is valid in both meshes.
    """
    case_at = _interp_mag_on_mesh(case_mesh, displacement_magnitude(u_case), xs, ys)
    ref_at = _interp_mag_on_mesh(ref_mesh, displacement_magnitude(u_ref), xs, ys)
    valid = ~(np.ma.getmaskarray(case_at) | np.ma.getmaskarray(ref_at))
    if not np.any(valid):
        return float("nan"), float("nan")
    cv = np.asarray(case_at)[valid]
    rv = np.asarray(ref_at)[valid]
    ratio_area = float(np.mean(cv)) / (float(np.mean(rv)) + 1e-30)
    loss_area = float(np.mean((cv / (rv + 1e-30) - 1.0) ** 2))
    return ratio_area, loss_area


def normalized_l2_mag_error_fixed(
    u_case: np.ndarray,
    u_ref: np.ndarray,
    case_mesh,
    ref_mesh,
    xs: np.ndarray,
    ys: np.ndarray,
) -> float:
    """Mesh-independent normalized L2 magnitude error on a fixed ``(xs, ys)`` grid::

        sqrt( sum (|u_case| - |u_ref|)^2 / sum |u_ref|^2 )

    Interpolates ``|u|`` onto the grid via each mesh's P1 triangulation (over
    points valid in both). Unlike :func:`transmitted_band_metrics_fixed`'s
    ``loss_area``, this is an *energy-normalized difference* (no per-point
    division by ``|u_ref|``), so it is robust to reference near-zeros at
    standing-wave nodes. Used for the out-of-band generalization metric (pass a
    sub-surface validation grid) and any fixed-region L2 amplitude error.
    Returns ``nan`` if no grid point is valid in both meshes.
    """
    case_at = _interp_mag_on_mesh(case_mesh, displacement_magnitude(u_case), xs, ys)
    ref_at = _interp_mag_on_mesh(ref_mesh, displacement_magnitude(u_ref), xs, ys)
    valid = ~(np.ma.getmaskarray(case_at) | np.ma.getmaskarray(ref_at))
    if not np.any(valid):
        return float("nan")
    cv = np.asarray(case_at)[valid]
    rv = np.asarray(ref_at)[valid]
    num = float(np.sqrt(np.sum((cv - rv) ** 2)))
    den = float(np.sqrt(np.sum(rv ** 2))) + 1e-30
    return num / den


def profile_error_surface_fixed(
    u_case: np.ndarray,
    u_ref: np.ndarray,
    case_mesh,
    ref_mesh,
    x_positions: np.ndarray,
    y_top: float,
) -> float:
    """Mesh-independent normalized L2 error of the free-surface magnitude
    *profile* on a fixed set of x-positions::

        sqrt( sum (|u_case| - |u_ref|)^2 / sum |u_ref|^2 )   at  (x, y_top)

    Like :func:`transmitted_displacement_ratio_fixed`, ``|u|`` is interpolated
    along each mesh's top-surface nodes (1-D, boundary-safe) rather than via the
    2-D triangulation. Sensitive to profile-*shape* mismatch that the scalar
    surface ratio averages away. Perfect cloak → 0.
    """
    case_xs, case_mag = _surface_mag_along_x(u_case, case_mesh, y_top)
    ref_xs, ref_mag = _surface_mag_along_x(u_ref, ref_mesh, y_top)
    case_at = np.interp(x_positions, case_xs, case_mag)
    ref_at = np.interp(x_positions, ref_xs, ref_mag)
    num = float(np.sqrt(np.sum((case_at - ref_at) ** 2)))
    den = float(np.sqrt(np.sum(ref_at ** 2))) + 1e-30
    return num / den


def transmission_loss(
    u_cloak: jnp.ndarray,
    u_ref_surface: jnp.ndarray,
    surface_indices: jnp.ndarray,
) -> jnp.ndarray:
    """JAX-traceable loss: mean per-node squared magnitude-ratio error.

    For each eval node, computes ``(|u_cloak_i| / |u_ref_i| - 1)^2`` and
    averages over nodes, where ``|u| = sqrt(Re_x^2 + Re_y^2 + Im_x^2 +
    Im_y^2)`` is the phase-invariant complex displacement magnitude.
    Penalises amplitude mismatch at each node but ignores a global
    complex phase shift between cloak and reference.

    Parameters
    ----------
    u_cloak : (n_nodes, 4) cloaked solution
    u_ref_surface : (n_surface, 4) reference displacement at surface nodes
        (pre-indexed from the full reference solution)
    surface_indices : node indices into u_cloak for the evaluation surface
    """
    u_s = u_cloak[surface_indices]
    mag_c = jnp.sqrt(jnp.sum(u_s ** 2, axis=1))
    mag_r = jnp.sqrt(jnp.sum(u_ref_surface ** 2, axis=1))
    err = (mag_c / (mag_r + 1e-30) - 1.0) ** 2
    return jnp.mean(err)


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


def make_biased_depth_eval_points(
    geometry,
    params,
    n_points: int,
    alpha: float = 4.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xs, ys)``: ``M`` eval points with a surface-like x-spread but
    pushed to a depth power-law biased toward the free surface.

    The x-sampling matches :func:`make_fixed_surface_eval_points` exactly
    (evenly spaced across ``[x_off, x_off + W]``, endpoints trimmed, optional
    Gaussian jitter). Each point's depth below the free surface is drawn from a
    power-law inverse-CDF so points cluster near the surface::

        physical_y = y_top - H * (1 - U**(1/alpha)),   U ~ Uniform(0, 1)

    ``alpha == 1`` is uniform in depth; larger ``alpha`` concentrates points
    nearer the surface (default 4). Points whose ``(x, physical_y)`` falls
    inside the cloak or defect footprint are dropped, so the loss measures the
    field only *beyond / below* the cloak, never inside it (same footprint
    filter the surface sampler uses). Raises if every point is dropped.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0 (got {alpha!r}).")

    x_left = params.x_off
    x_right = params.x_off + params.W
    xs = np.linspace(x_left, x_right, n_points + 2)[1:-1]

    rng = np.random.default_rng(seed)
    if noise_sigma > 0:
        xs = xs + rng.normal(0.0, float(noise_sigma), size=xs.shape)
        xs = np.clip(xs, x_left, x_right)

    y_top = float(params.y_top)
    H = float(params.H)
    U = rng.uniform(0.0, 1.0, size=xs.shape)
    ys = y_top - H * (1.0 - U ** (1.0 / float(alpha)))

    keep_x, keep_y = [], []
    for x, y in zip(xs, ys):
        pt = jnp.array([float(x), float(y)])
        if not bool(geometry.in_defect(pt)) and not bool(geometry.in_cloak(pt)):
            keep_x.append(float(x))
            keep_y.append(float(y))
    if not keep_x:
        raise RuntimeError(
            f"All {n_points} surface_depth eval points fall inside the "
            f"cloak/defect footprint. Increase n_eval_points, widen the "
            f"domain, or lower alpha."
        )
    return (np.asarray(keep_x, dtype=np.float64),
            np.asarray(keep_y, dtype=np.float64))


def make_surface_column_eval_points(
    geometry,
    params,
    n_x: int,
    n_y: int,
    depth: float,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xs, ys)``: a 2-D cloud with ``n_x`` surface x-positions
    (same selection as :func:`make_fixed_surface_eval_points`) and ``n_y``
    y-positions per column, evenly spaced from ``y_top - depth`` up to
    ``y_top`` (inclusive at both ends).

    Pairs whose ``(x, y)`` falls inside the cloak or defect footprint are
    dropped. The surface (``y == y_top``) row is included so the loss
    measures the free surface plus a strip going down to ``depth``.

    Returns flattened length-``n_x * n_y`` arrays (after defect/cloak
    filtering).
    """
    if depth <= 0.0:
        raise ValueError(
            f"depth must be > 0 (got {depth!r}) for surface_column."
        )
    if depth >= float(params.H):
        raise ValueError(
            f"depth ({depth!r}) must be < physical height H ({params.H!r})."
        )
    if n_y < 1:
        raise ValueError(f"n_y must be >= 1 (got {n_y!r}).")

    x_left = params.x_off
    x_right = params.x_off + params.W
    xs_surf = np.linspace(x_left, x_right, n_x + 2)[1:-1]
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        xs_surf = xs_surf + rng.normal(0.0, float(noise_sigma), size=xs_surf.shape)
        xs_surf = np.clip(xs_surf, x_left, x_right)

    y_top = float(params.y_top)
    ys_col = np.linspace(y_top - float(depth), y_top, int(n_y))

    keep_x, keep_y = [], []
    for x in xs_surf:
        for y in ys_col:
            pt = jnp.array([float(x), float(y)])
            if bool(geometry.in_defect(pt)) or bool(geometry.in_cloak(pt)):
                continue
            keep_x.append(float(x))
            keep_y.append(float(y))
    if not keep_x:
        raise RuntimeError(
            f"All {n_x * n_y} surface_column eval points fall inside the "
            f"cloak/defect footprint. Increase n_x, decrease depth, or "
            f"widen the domain."
        )
    return (np.asarray(keep_x, dtype=np.float64),
            np.asarray(keep_y, dtype=np.float64))


def find_nearest_node_indices_xy(
    mesh_points: np.ndarray,
    eval_xs: np.ndarray,
    eval_ys: np.ndarray,
) -> np.ndarray:
    """For each ``(x, y)`` pair, return the cloak-mesh node index closest in
    Euclidean distance.

    Used by the ``"surface_depth"`` loss, which samples an interior 2-D cloud.
    Unlike the embedded top-surface path, the cloak mesh is not built with
    these positions forced, so accuracy depends on local mesh density at the
    sampled depths. Indices are static (computed once at setup), which is fine
    for autodiff — only ``u_cloak[indices]`` is differentiated.
    """
    pts = np.asarray(mesh_points)
    out = np.empty(len(eval_xs), dtype=np.int64)
    for i in range(len(eval_xs)):
        dx = pts[:, 0] - float(eval_xs[i])
        dy = pts[:, 1] - float(eval_ys[i])
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


def get_magnitude_band_indices(
    mesh_points: np.ndarray,
    geometry,
    y_top: float,
    x_left: float,
    x_right: float,
    depth: float,
    mode: str = "downstream",
    tol: float = 1e-6,
) -> np.ndarray:
    """Return cloak-mesh node indices in the top band ``[y_top - depth, y_top]``.

    ``mode == "downstream"`` further restricts to ``x > x_c + tol`` — with
    ``depth == 0`` this reproduces ``get_top_surface_beyond_cloak_indices``
    exactly. ``mode == "full"`` keeps the entire physical x-range
    ``[x_left, x_right]``. Nodes inside the cloak or defect footprint are
    dropped in both modes (the band would otherwise reach into the cloak
    interior for ``depth > 0``).
    """
    import jax
    import jax.numpy as jnp

    if depth < 0.0:
        raise ValueError(f"depth must be >= 0 (got {depth!r}).")
    if mode not in ("downstream", "full"):
        raise ValueError(
            f"Unknown band_x_filter mode: {mode!r}. "
            f"Choose 'downstream' or 'full'."
        )

    pts = np.asarray(mesh_points)
    y_lo = float(y_top) - float(depth)
    in_band_y = (pts[:, 1] >= y_lo - tol) & (pts[:, 1] <= float(y_top) + tol)
    in_phys_x = (pts[:, 0] >= float(x_left) - tol) & (
        pts[:, 0] <= float(x_right) + tol
    )
    in_x = in_phys_x
    if mode == "downstream":
        in_x = in_x & (pts[:, 0] > float(geometry.x_c) + tol)

    candidate = np.where(in_band_y & in_x)[0]
    if len(candidate) == 0:
        return candidate

    cand_pts = jnp.array(pts[candidate, :2])
    in_cloak = np.asarray(jax.vmap(geometry.in_cloak)(cand_pts))
    in_defect = np.asarray(jax.vmap(geometry.in_defect)(cand_pts))
    return candidate[~(in_cloak | in_defect)]


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
    elif loss_type == "surface_depth":
        if loss_cfg is None:
            raise ValueError("loss.type='surface_depth' requires a LossConfig.")
        n_eval = int(loss_cfg.n_eval_points)
        if n_eval <= 0:
            raise ValueError(
                "loss.type='surface_depth' requires loss.n_eval_points > 0."
            )
        eval_xs, eval_ys = make_biased_depth_eval_points(
            geometry, params, n_eval,
            alpha=float(loss_cfg.alpha),
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed),
        )
        indices = find_nearest_node_indices_xy(pts, eval_xs, eval_ys)
        loss_fn = transmission_loss
    elif loss_type == "surface_column":
        if loss_cfg is None:
            raise ValueError("loss.type='surface_column' requires a LossConfig.")
        n_x_eval = int(loss_cfg.n_eval_points)
        n_y_eval = int(loss_cfg.n_column_samples)
        if n_x_eval <= 0:
            raise ValueError(
                "loss.type='surface_column' requires loss.n_eval_points > 0."
            )
        if n_y_eval <= 0:
            raise ValueError(
                "loss.type='surface_column' requires loss.n_column_samples > 0."
            )
        eval_xs, eval_ys = make_surface_column_eval_points(
            geometry, params, n_x_eval, n_y_eval,
            depth=float(loss_cfg.depth),
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed),
        )
        raw_indices = find_nearest_node_indices_xy(pts, eval_xs, eval_ys)
        indices = np.unique(raw_indices)
        loss_fn = transmission_loss
    elif loss_type == "magnitude_band_integral":
        if loss_cfg is None:
            raise ValueError(
                "loss.type='magnitude_band_integral' requires a LossConfig."
            )
        depth = float(loss_cfg.depth)
        if depth < 0.0:
            raise ValueError(
                f"loss.type='magnitude_band_integral' requires depth >= 0 "
                f"(got {depth!r})."
            )
        if depth >= float(params.H):
            raise ValueError(
                f"depth ({depth!r}) must be < physical height H "
                f"({params.H!r}) for magnitude_band_integral; deeper bands "
                f"reach into the bottom PML."
            )
        indices = get_magnitude_band_indices(
            pts, geometry, params.y_top,
            params.x_off, params.x_off + params.W,
            depth=depth,
            mode=str(loss_cfg.band_x_filter),
        )
        if len(indices) == 0:
            raise RuntimeError(
                f"magnitude_band_integral selected zero nodes (depth="
                f"{depth!r}, band_x_filter={loss_cfg.band_x_filter!r}). "
                f"Check geometry and refinement."
            )
        loss_fn = transmission_loss
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type!r}. Choose from "
            f"'right_boundary', 'top_surface', 'outside_cloak', 'depth_line', "
            f"'surface_depth', 'surface_column', 'magnitude_band_integral'."
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
