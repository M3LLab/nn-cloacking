"""Uniform-in-cloak, TRI6-capable mesh generation (new builder).

This is a from-scratch alternative to :mod:`rayleigh_cloak.mesh` for the
*homogenised* cloak solve. Two differences from the legacy builder:

1. **Uniform refinement inside the cloak.** The legacy builder grades the cloak
   element size with a ``Distance -> Threshold`` field (fine near the inner
   triangle edges, coarse toward the outer apex). Here the entire cloak bounding
   box is meshed at a single size ``h_in`` via a gmsh ``Box`` field, so every
   macro cell is discretised identically. This matters for the piecewise-constant
   homogenised material (one ``C`` per macro cell): a uniform size that divides
   the macro-cell pitch (optionally with ``embed_macro_grid``) keeps elements from
   straddling material jumps.

2. **Second-order (TRI6) elements.** When ``cfg.mesh.ele_type == "TRI6"`` the gmsh
   mesh is promoted to quadratic with ``setOrder(2)`` and read back as
   ``triangle6``. Quadratic elements have far lower numerical dispersion per DOF,
   so the macro wave solve needs ~5-6 elem/wavelength instead of ~10-20.
   ``ele_type == "TRI3"`` is also supported (linear, uniform-in-cloak).

The surface construction, defect cutout, macro-grid embedding, PML boundary
embedding and ``extract_submesh`` are all **reused** from the legacy module — only
the cloak refinement field and the element order differ.

Selected via ``cfg.mesh.builder == "uniform_tri6"`` (see ``rayleigh_cloak.solver``
dispatch). The legacy builder remains the default.
"""

from __future__ import annotations

import os

import gmsh
import meshio
import numpy as np
from jax_fem.generate_mesh import Mesh

from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry

# Reuse the legacy helpers verbatim — these are independent of the refinement
# strategy or element order.
from rayleigh_cloak.mesh import (  # noqa: F401  (extract_submesh re-exported)
    _embed_physical_boundary_points,
    _resolve_mesh_sizes,
    _resolve_top_eval_xs,
    extract_submesh,
)


def _cloak_bbox(geometry: CloakGeometry) -> tuple[float, float, float, float]:
    """Cloak bounding box ``(x_min, x_max, y_min, y_max)``.

    Matches :class:`~rayleigh_cloak.cells.CellDecomposition` so the uniform field
    aligns with the macro-cell grid.
    """
    if hasattr(geometry, "bbox"):
        return geometry.bbox()
    # Triangular geometry fallback.
    x_min = geometry.x_c - geometry.c
    x_max = geometry.x_c + geometry.c
    y_min = geometry.y_top - geometry.b
    y_max = geometry.y_top
    return x_min, x_max, y_min, y_max


def _add_uniform_cloak_field(
    geometry: CloakGeometry,
    h_in: float,
    h_out: float,
) -> int:
    """Add a gmsh ``Box`` field that is uniform ``h_in`` over the cloak bbox.

    The box spans the cloak bounding box; ``VIn = h_in`` inside it and ``VOut =
    h_out`` outside, with a linear transition of thickness ``~ (y_max - y_min)``
    so the far field coarsens smoothly. Returns the field tag (to be composed via
    ``Min`` with the surface field).
    """
    x_min, x_max, y_min, y_max = _cloak_bbox(geometry)
    thickness = max(y_max - y_min, x_max - x_min)

    f_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_box, "VIn", float(h_in))
    gmsh.model.mesh.field.setNumber(f_box, "VOut", float(h_out))
    gmsh.model.mesh.field.setNumber(f_box, "XMin", float(x_min))
    gmsh.model.mesh.field.setNumber(f_box, "XMax", float(x_max))
    gmsh.model.mesh.field.setNumber(f_box, "YMin", float(y_min))
    gmsh.model.mesh.field.setNumber(f_box, "YMax", float(y_max))
    # 2-D model lives at z=0; span z so the box contains it.
    gmsh.model.mesh.field.setNumber(f_box, "ZMin", -1.0)
    gmsh.model.mesh.field.setNumber(f_box, "ZMax", 1.0)
    gmsh.model.mesh.field.setNumber(f_box, "Thickness", float(thickness))
    return f_box


def _generate_and_read(cfg: SimulationConfig, msh_path: str) -> Mesh:
    """Generate the 2-D mesh, promote to TRI6 if requested, read it back.

    Writes ``msh_path``, finalizes gmsh, and returns a ``jax_fem`` ``Mesh`` with
    the connectivity matching ``cfg.mesh.ele_type`` (``triangle`` for TRI3,
    ``triangle6`` for TRI6).
    """
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    ele_type = cfg.mesh.ele_type
    if ele_type == "TRI6":
        # Promote to quadratic: adds the 3 edge-midside nodes per triangle. All
        # boundaries here are straight, so midside nodes land exactly at edge
        # midpoints (gmsh ordering: 3 corners, then mids of (0,1),(1,2),(2,0)),
        # which is exactly what jax_fem's TRI6 re_order [0,1,2,5,3,4] expects.
        gmsh.model.mesh.setOrder(2)
        cell_type = "triangle6"
    elif ele_type == "TRI3":
        cell_type = "triangle"
    else:
        raise ValueError(
            f"mesh_uniform builder supports ele_type in {{'TRI3','TRI6'}}; "
            f"got {ele_type!r}."
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points[:, :2]
    if cell_type not in msh.cells_dict:
        raise RuntimeError(
            f"expected '{cell_type}' cells in the generated mesh but found "
            f"{list(msh.cells_dict)}; ele_type={ele_type}."
        )
    cells = msh.cells_dict[cell_type]
    return Mesh(points, cells, ele_type=ele_type)


def generate_mesh(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
    cloak_field_fn=None,
) -> Mesh:
    """Uniform-in-cloak / TRI6 analogue of :func:`rayleigh_cloak.mesh.generate_mesh`.

    Builds a triangular mesh of the full domain with the defect cut out (unless
    ``cfg.is_reference``). The cloak refinement is uniform ``h_in`` across the
    cloak bbox instead of the legacy graded field.

    ``cloak_field_fn(geometry, h_in, h_out) -> field_tag`` overrides how the cloak
    size field is built; it defaults to the uniform-over-the-bbox ``Box`` field.
    :mod:`rayleigh_cloak.mesh_cloak_uniform` passes a field that is uniform over
    the cloak *annulus* only.
    """
    p = params
    h_elem, h_in, h_out, h_surf = _resolve_mesh_sizes(cfg, p)
    if cloak_field_fn is None:
        cloak_field_fn = _add_uniform_cloak_field

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain")
    geo = gmsh.model.geo

    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(p.W_total, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(p.W_total, p.H_total, 0.0, h_elem)
    p4 = geo.addPoint(0.0, p.H_total, 0.0, h_elem)

    top_eval_xs = _resolve_top_eval_xs(cfg, p, geometry)

    cloak_field = None
    if cfg.is_reference:
        from rayleigh_cloak.mesh import _chain_top_edge

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
        # Reuse the legacy defect cutout + surface construction. Its graded
        # ``_cloak_field_tag`` is intentionally ignored below in favour of the
        # uniform Box field.
        top_lines = geometry.build_gmsh_geometry(
            geo, (p1, p2, p3, p4), h_in, h_elem, h_outside=h_out,
            top_eval_xs=top_eval_xs, h_surf=h_surf,
        )
        cloak_field = cloak_field_fn(geometry, h_in, h_out)

    f_thresh_surf = _add_surface_field(p, top_lines, h_surf, h_out)
    _compose_background(cfg, cloak_field, f_thresh_surf)

    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh.msh")
    return _generate_and_read(cfg, msh_path)


def generate_mesh_full(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
    cloak_field_fn=None,
) -> Mesh:
    """Uniform-in-cloak / TRI6 analogue of
    :func:`rayleigh_cloak.mesh.generate_mesh_full`.

    Full-domain mesh (no defect cutout) with the cloak vertices embedded and a
    uniform ``h_in`` over the cloak bbox. Pair with ``extract_submesh`` for the
    cloak solve, exactly as in the legacy pipeline.

    See :func:`generate_mesh` for ``cloak_field_fn``.
    """
    p = params
    h_elem, h_in, h_out, h_surf = _resolve_mesh_sizes(cfg, p)
    if cloak_field_fn is None:
        cloak_field_fn = _add_uniform_cloak_field

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain_full")
    geo = gmsh.model.geo

    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(p.W_total, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(p.W_total, p.H_total, 0.0, h_elem)
    p4 = geo.addPoint(0.0, p.H_total, 0.0, h_elem)

    top_eval_xs = _resolve_top_eval_xs(cfg, p, geometry)

    # Reuse the legacy full-domain surface construction (embeds cloak vertices +
    # inner-triangle conformity lines, sets the graded ``_cloak_field_tag`` which
    # we ignore). Pass ``h_in`` as the cloak characteristic length for parity.
    top_lines = geometry.build_gmsh_geometry_full(
        geo, (p1, p2, p3, p4), h_in, h_elem, h_outside=h_out,
        top_eval_xs=top_eval_xs, h_surf=h_surf,
    )

    if cfg.mesh.embed_macro_grid:
        # The legacy macro-grid *point* embedding is incompatible with uniform
        # sizing: the embedded lattice points land within meshing tolerance of
        # the regular h_in nodes, producing coincident nodes and zero-area
        # sliver triangles -> a singular stiffness matrix. It is unnecessary
        # here anyway: a uniform fine h_in resolves every macro cell, and the
        # piecewise-constant material is sampled per quadrature point, so the
        # only error from an element straddling a macro-cell boundary is the
        # usual O(h) unfitted-coefficient error, which vanishes under
        # refinement (confirmed by the convergence sweep). So we skip it.
        print(
            "[mesh_uniform] note: embed_macro_grid is ignored by the uniform "
            "builder (it degenerates the mesh under uniform sizing); the "
            "uniform h_in resolves each macro cell directly."
        )

    cloak_field = cloak_field_fn(geometry, h_in, h_out)
    f_thresh_surf = _add_surface_field(p, top_lines, h_surf, h_out)
    _compose_background(cfg, cloak_field, f_thresh_surf)

    # Embed physical-domain boundary points (same as legacy generate_mesh_full)
    # so boundary node selection needs no spatial tolerance.
    _embed_physical_boundary_points(geo, p, h_out)

    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh_full.msh")
    return _generate_and_read(cfg, msh_path)


def _add_surface_field(p: DerivedParams, top_lines, h_surf: float, h_out: float) -> int:
    """Free-surface ``Distance -> Threshold`` refinement field (same as legacy).

    Grades from ``h_surf`` at the surface up to ``h_out`` over a distance of one
    star wavelength. With the uniform builder ``h_surf`` defaults to ``h_in`` (the
    legacy ``refinement_factor_surface -> refinement_factor_cloak`` fallback), so
    the cloak-top strip stays uniform at ``h_in``.
    """
    f_dist_surf = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_surf, "CurvesList", top_lines)
    gmsh.model.mesh.field.setNumber(f_dist_surf, "Sampling", 200)

    f_thresh_surf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "InField", f_dist_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMax", h_out)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMax", p.lambda_star)
    return f_thresh_surf


def _compose_background(cfg, cloak_field, f_thresh_surf) -> None:
    """Set the background mesh size field = ``Min(cloak_field, surface_field)``."""
    if cloak_field is not None:
        f_final = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_final, "FieldsList", [cloak_field, f_thresh_surf])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_final)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)
