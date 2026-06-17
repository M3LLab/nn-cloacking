"""Gmsh mesh generation for the extended domain (physical + PML)."""

from __future__ import annotations

import os

import gmsh
import meshio
import numpy as np
from jax_fem.generate_mesh import Mesh

from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry


def _resolve_mesh_sizes(cfg: SimulationConfig, p: DerivedParams) -> tuple[float, float, float, float]:
    """Compute (h_elem, h_in, h_out, h_surf) from the legacy + new MeshConfig knobs.

    - h_elem  : base element size derived from nx_total / ny_total. Kept for
                back-compat: used as corner-point characteristic length.
    - h_in    : target inside the cloak       = h_elem / refinement_factor_cloak
    - h_out   : target outside (away from surface and cloak)
                                              = h_elem / refinement_factor_outside
    - h_surf  : target near the free surface  = h_elem / refinement_factor_surface

    The two ``_cloak`` and ``_surface`` factors fall back to the legacy
    ``refinement_factor`` so configs predating these fields are unaffected.
    """
    h_elem = min(p.W_total / p.nx_total, p.H_total / p.ny_total)
    rf_cloak = cfg.mesh.refinement_factor_cloak
    if rf_cloak is None:
        rf_cloak = cfg.mesh.refinement_factor
    rf_surf = cfg.mesh.refinement_factor_surface
    if rf_surf is None:
        rf_surf = rf_cloak
    rf_out = cfg.mesh.refinement_factor_outside

    h_in = h_elem / float(rf_cloak)
    h_out = h_elem / float(rf_out)
    h_surf = h_elem / float(rf_surf)
    return h_elem, h_in, h_out, h_surf


def generate_mesh(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> Mesh:
    """Generate a triangular (or quad) mesh of the full domain.

    When ``cfg.is_reference`` is True the domain is a plain rectangle (no
    defect).  Otherwise the geometry's ``build_gmsh_geometry`` cuts out the
    defect and adds refinement fields.

    Returns a ``jax_fem.generate_mesh.Mesh`` instance.
    """
    p = params
    h_elem, h_fine, h_out, h_surf = _resolve_mesh_sizes(cfg, p)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain")
    geo = gmsh.model.geo

    # Corner points
    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(p.W_total, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(p.W_total, p.H_total, 0.0, h_elem)
    p4 = geo.addPoint(0.0, p.H_total, 0.0, h_elem)

    top_eval_xs = _resolve_top_eval_xs(cfg, p, geometry)

    if cfg.is_reference:
        # Plain rectangle — no defect. When eval xs are configured, chain
        # the top edge through them so each is a real corner vertex.
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        if top_eval_xs is not None and len(top_eval_xs) > 0:
            top_lines = _chain_top_edge(
                geo, p3, p4, top_eval_xs, p.y_top, h_surf, descending=True,
            )
        else:
            top_lines = [geo.addLine(p3, p4)]
        l_left = geo.addLine(p4, p1)

        outer_loop = geo.addCurveLoop([l_bot, l_right] + top_lines + [l_left])
        geo.addPlaneSurface([outer_loop])
        gmsh.model.geo.synchronize()
    else:
        # Delegate defect cutout + refinement to the geometry object
        top_lines = geometry.build_gmsh_geometry(
            geo, (p1, p2, p3, p4), h_fine, h_elem, h_outside=h_out,
            top_eval_xs=top_eval_xs, h_surf=h_surf,
        )

    # ── surface refinement (common to all geometries) ──
    f_dist_surf = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_surf, "CurvesList", top_lines)
    gmsh.model.mesh.field.setNumber(f_dist_surf, "Sampling", 200)

    f_thresh_surf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "InField", f_dist_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMax", h_out)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMax", p.lambda_star)

    # ── compose background field ──
    if cfg.is_reference:
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)
    else:
        cloak_field = getattr(geometry, "_cloak_field_tag", None)
        if cloak_field is not None:
            f_final = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(
                f_final, "FieldsList", [cloak_field, f_thresh_surf])
            gmsh.model.mesh.field.setAsBackgroundMesh(f_final)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    os.makedirs(cfg.output_dir, exist_ok=True)
    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points[:, :2]
    cells = msh.cells_dict["triangle"]

    return Mesh(points, cells, ele_type=cfg.mesh.ele_type)


def _embed_macro_grid_lines(
    cfg: SimulationConfig,
    geometry: CloakGeometry,
    h_size: float,
) -> int:
    """Embed the macro-grid's interior lines as 1-D constraints in the surface.

    Forces gmsh to put nodes along each macro-cell boundary, so no triangular
    element straddles a discontinuity in the piecewise-constant material
    field. Each element ends up entirely inside one macro cell and its single
    quadrature point reads an unambiguous (λ, μ, ρ).

    Returns the number of internal lines embedded (for logging).

    Notes
    -----
    Must be called *after* the geometry has built the surface (so that
    ``gmsh.model.getEntities(dim=2)`` returns it) but *before*
    ``gmsh.model.mesh.generate``.
    """
    n_x = cfg.cells.n_x
    n_y = cfg.cells.n_y
    if hasattr(geometry, "bbox"):
        x_min, x_max, y_min, y_max = geometry.bbox()
    else:
        # Triangular geometry fallback (matches CellDecomposition).
        x_min = geometry.x_c - geometry.c
        x_max = geometry.x_c + geometry.c
        y_min = geometry.y_top - geometry.b
        y_max = geometry.y_top

    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y

    geo = gmsh.model.geo

    # Strategy: embed *only* the (n_x+1)*(n_y+1) lattice points, not the
    # interior lines themselves. gmsh's geo kernel can't auto-split crossing
    # embedded curves ("Unable to recover the edge"), so trying to embed the
    # full grid of lines fails. Just pinning nodes at every lattice point —
    # combined with a target element size ≈ cell width inside the cloak —
    # forces gmsh to produce small triangles whose vertices are lattice
    # nodes; those triangles lie entirely inside one macro cell and never
    # straddle a discontinuity.
    pts = []
    for i in range(n_x + 1):
        x = x_min + i * cell_dx
        for j in range(n_y + 1):
            y = y_min + j * cell_dy
            # Skip points exactly on the bbox boundary (i==0/n_x and j==0/n_y).
            # The mesh already has nodes on the cloak bbox via the geometry's
            # outer rectangle / triangle-vertex constraints — embedding co-
            # located bbox-edge points causes gmsh duplicate-node errors.
            if i == 0 or i == n_x or j == 0 or j == n_y:
                continue
            pts.append(geo.addPoint(x, y, 0.0, h_size))

    geo.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    if not surfaces:
        raise RuntimeError("No 2-D surface found to embed macro-grid lines into.")
    surf_tag = surfaces[0][1]

    if pts:
        gmsh.model.mesh.embed(0, pts, 2, surf_tag)
    return len(pts)


def _chain_top_edge(
    geo,
    p_start: int,
    p_end: int,
    eval_xs: np.ndarray,
    y_top: float,
    h_size: float,
    descending: bool,
    x_lo: float | None = None,
    x_hi: float | None = None,
    endpoint_xs: tuple[float, float] | None = None,
) -> list[int]:
    """Build a chain of gmsh lines from ``p_start`` to ``p_end`` along ``y =
    y_top``, with one intermediate vertex at each ``eval_xs`` value that lies
    strictly between the segment endpoints.

    ``descending`` is True when traveling from ``p_start`` to ``p_end`` means
    decreasing x; this controls the sort order of the intermediate xs.

    ``x_lo`` / ``x_hi`` (or ``endpoint_xs``) optionally restrict which eval
    xs are placed on this segment, used by geometries whose top edge is
    split into multiple sub-segments around the cloak. Eval xs that
    coincide with an endpoint within tolerance are dropped (gmsh would
    otherwise produce a zero-length line).

    Returns the new line tags in traversal order.
    """
    if endpoint_xs is not None:
        if x_lo is None:
            x_lo = min(endpoint_xs)
        if x_hi is None:
            x_hi = max(endpoint_xs)

    span = abs((x_hi - x_lo)) if (x_lo is not None and x_hi is not None) else 1.0
    tol = max(1e-9, 1e-6 * (span if span else 1.0))

    xs = np.asarray(eval_xs)
    if x_lo is not None:
        xs = xs[xs > x_lo + tol]
    if x_hi is not None:
        xs = xs[xs < x_hi - tol]

    if len(xs) == 0:
        return [geo.addLine(p_start, p_end)]

    xs_sorted = np.sort(xs)
    if descending:
        xs_sorted = xs_sorted[::-1]

    pts = [p_start]
    for x in xs_sorted:
        pts.append(geo.addPoint(float(x), float(y_top), 0.0, h_size))
    pts.append(p_end)
    return [geo.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _resolve_top_eval_xs(
    cfg: SimulationConfig,
    p: DerivedParams,
    geometry: CloakGeometry,
) -> np.ndarray | None:
    """Eval x-positions to embed as forced top-edge vertices, or None.

    When ``cfg.loss.n_eval_points > 0`` the geometries chain the top edges
    through these xs so each one is a real corner-of-line vertex (and
    therefore guaranteed to be a mesh node referenced by the triangulation,
    independent of refinement). Returns ``None`` when the feature is off,
    so the legacy single-line top-edge construction is kept unchanged.
    """
    n_pts = int(cfg.loss.n_eval_points)
    if n_pts <= 0:
        return None
    from rayleigh_cloak.loss import make_fixed_surface_eval_points
    return make_fixed_surface_eval_points(
        geometry, p, n_pts,
        noise_sigma=float(cfg.loss.eval_noise_sigma),
        seed=int(cfg.loss.eval_noise_seed),
    )


def _embed_physical_boundary_points(geo, p: DerivedParams, h_elem: float) -> None:
    """Embed points along the physical-domain boundaries into the mesh.

    Places points at intervals of ``h_elem`` along the left, right, and
    bottom physical boundaries.  Only strictly interior points are
    embedded (endpoints that would coincide with existing domain edges
    are skipped to avoid duplicate-point conflicts in gmsh).
    """
    x0, y0 = p.x_off, p.y_off
    x1, y1 = p.x_off + p.W, p.y_off + p.H
    eps = 1e-12  # guard against landing on domain edges

    point_tags = []

    def _is_on_domain_edge(x, y):
        """True if (x,y) is on the outer domain rectangle."""
        on_left = abs(x) < eps
        on_right = abs(x - p.W_total) < eps
        on_bottom = abs(y) < eps
        on_top = abs(y - p.H_total) < eps
        return on_left or on_right or on_bottom or on_top

    # Right boundary: x = x1, y from y0 to y1
    n_right = max(3, int(round((y1 - y0) / h_elem)) + 1)
    for i in range(n_right):
        y = y0 + i * (y1 - y0) / (n_right - 1)
        if not _is_on_domain_edge(x1, y):
            point_tags.append(geo.addPoint(x1, y, 0.0, h_elem))

    # Left boundary: x = x0, y from y0 to y1
    n_left = max(3, int(round((y1 - y0) / h_elem)) + 1)
    for i in range(n_left):
        y = y0 + i * (y1 - y0) / (n_left - 1)
        if not _is_on_domain_edge(x0, y):
            point_tags.append(geo.addPoint(x0, y, 0.0, h_elem))

    # Bottom boundary: x from x0 to x1 (skip corners)
    n_bottom = max(3, int(round((x1 - x0) / h_elem)) + 1)
    for i in range(1, n_bottom - 1):
        x = x0 + i * (x1 - x0) / (n_bottom - 1)
        if not _is_on_domain_edge(x, y0):
            point_tags.append(geo.addPoint(x, y0, 0.0, h_elem))

    geo.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    surf_tag = surfaces[0][1]
    gmsh.model.mesh.embed(0, point_tags, 2, surf_tag)


def generate_mesh_full(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> Mesh:
    """Generate a mesh of the full domain (no defect cutout).

    The cloak geometry vertices are embedded so the mesh refines properly,
    but no hole is cut.  This mesh can be used for both the reference solve
    (homogeneous material) and, after ``extract_submesh``, for the cloak solve.
    """
    p = params
    h_elem, h_fine, h_out, h_surf = _resolve_mesh_sizes(cfg, p)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain_full")
    geo = gmsh.model.geo

    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(p.W_total, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(p.W_total, p.H_total, 0.0, h_elem)
    p4 = geo.addPoint(0.0, p.H_total, 0.0, h_elem)

    top_eval_xs = _resolve_top_eval_xs(cfg, p, geometry)

    top_lines = geometry.build_gmsh_geometry_full(
        geo, (p1, p2, p3, p4), h_fine, h_elem, h_outside=h_out,
        top_eval_xs=top_eval_xs, h_surf=h_surf,
    )

    if cfg.mesh.embed_macro_grid:
        n_emb = _embed_macro_grid_lines(cfg, geometry, h_size=h_fine)
        # Don't print at module level (called from inside the inner loop in
        # benchmarks); leave the message to the caller if it wants.

    # ── surface refinement (same as generate_mesh) ──
    f_dist_surf = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_surf, "CurvesList", top_lines)
    gmsh.model.mesh.field.setNumber(f_dist_surf, "Sampling", 200)

    f_thresh_surf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "InField", f_dist_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMax", h_out)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMax", p.lambda_star)

    cloak_field = getattr(geometry, "_cloak_field_tag", None)
    if cloak_field is not None:
        f_final = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_final, "FieldsList", [cloak_field, f_thresh_surf])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_final)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)

    # ── embed physical-domain boundary lines ──
    # The physical domain sits inside the PML region.  Embedding these
    # four lines forces gmsh to place nodes exactly on them, so that
    # boundary node selection needs no spatial tolerance.
    # Embed boundary constraint points at h_out spacing so they don't force
    # the bulk mesh finer than the requested outside size.
    _embed_physical_boundary_points(geo, p, h_out)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    os.makedirs(cfg.output_dir, exist_ok=True)
    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh_full.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points[:, :2]
    cells = msh.cells_dict["triangle"]

    return Mesh(points, cells, ele_type=cfg.mesh.ele_type)


def extract_submesh(
    mesh: Mesh,
    geometry: CloakGeometry,
) -> tuple[Mesh, np.ndarray]:
    """Remove elements whose centroid lies inside the defect region.

    Orphan nodes (not referenced by any remaining element) are removed and
    cell indices are renumbered accordingly.

    Returns
    -------
    submesh : Mesh
        Clean mesh with no orphan nodes.
    kept_nodes : (n_sub_nodes,) int array
        Original node indices that were kept.  Useful for mapping
        full-mesh solutions to the submesh:
        ``u_submesh = u_full[kept_nodes]``.
    """
    import jax.numpy as jnp

    points = np.asarray(mesh.points)
    cells = np.asarray(mesh.cells)

    # Compute element centroids
    centroids = points[cells].mean(axis=1)  # (n_elem, 2)

    # Keep elements outside the defect
    keep = np.array([
        not bool(geometry.in_defect(jnp.array(c)))
        for c in centroids
    ])
    cells_sub = cells[keep]

    # Remove orphan nodes and renumber
    kept_nodes = np.unique(cells_sub)
    new_points = points[kept_nodes]
    old_to_new = np.full(len(points), -1, dtype=int)
    old_to_new[kept_nodes] = np.arange(len(kept_nodes))
    new_cells = old_to_new[cells_sub]

    return Mesh(new_points, new_cells, ele_type=mesh.ele_type), kept_nodes
