"""Gmsh tetrahedral mesh generation for the 3D extended domain (physical + PML)."""

from __future__ import annotations

import os

import gmsh
import meshio
import numpy as np
from jax_fem.generate_mesh import Mesh

from rayleigh_cloak_3d.config import DerivedParams3D, SimulationConfig3D
from rayleigh_cloak_3d.geometry.base import CloakGeometry3D


def _element_size(cfg: SimulationConfig3D, p: DerivedParams3D) -> tuple[float, float]:
    """Return ``(h_elem, h_fine)`` for the extended domain."""
    nx_total = 2 * cfg.mesh.n_pml + cfg.mesh.n_phys
    nz_total = cfg.mesh.n_pml + cfg.mesh.n_phys
    h_elem = min(p.W_total / nx_total, p.H_total / nz_total)
    h_fine = h_elem / cfg.mesh.refinement_factor
    return h_elem, h_fine


def generate_mesh_full(
    cfg: SimulationConfig3D,
    params: DerivedParams3D,
    geometry: CloakGeometry3D,
) -> Mesh:
    """Generate a tetrahedral mesh of the full 3D domain (no defect cutout).

    The cloak geometry adds a refinement field around its axis via
    ``geometry.build_gmsh_geometry_full``. No hole is cut; defect removal
    happens later through :func:`extract_submesh`.
    """
    p = params
    h_elem, h_fine = _element_size(cfg, p)
    h_surf = h_fine

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain_full_3d")
    occ = gmsh.model.occ

    box_tag = occ.addBox(0.0, 0.0, 0.0, p.W_total, p.W_total, p.H_total)
    occ.synchronize()

    # Default mesh size at all points of the box.
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_elem)

    # Delegate cloak refinement to the geometry object (adds a Threshold field,
    # stashed on geometry._cloak_field_tag).
    geometry.build_gmsh_geometry_full(occ, box_tag, h_fine, h_elem)

    # Top-surface refinement: a size field near z = z_top.
    top_surfaces = []
    for dim, tag in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2] - p.z_top) < 1e-9:
            top_surfaces.append(tag)

    f_dist_top = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_top, "SurfacesList", top_surfaces)
    gmsh.model.mesh.field.setNumber(f_dist_top, "Sampling", 100)

    f_thresh_top = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_top, "InField", f_dist_top)
    gmsh.model.mesh.field.setNumber(f_thresh_top, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_top, "SizeMax", h_elem)
    gmsh.model.mesh.field.setNumber(f_thresh_top, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_top, "DistMax", p.lambda_star)

    cloak_field = getattr(geometry, "_cloak_field_tag", None)
    fields = [f_thresh_top]
    if cloak_field is not None:
        fields.append(cloak_field)
    if len(fields) == 1:
        gmsh.model.mesh.field.setAsBackgroundMesh(fields[0])
    else:
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(3)
    if cfg.mesh.ele_type == "TET10":
        gmsh.model.mesh.setOrder(2)

    os.makedirs(cfg.output_dir, exist_ok=True)
    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh_3d_full.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points
    cells_key = "tetra10" if cfg.mesh.ele_type == "TET10" else "tetra"
    cells = msh.cells_dict[cells_key]

    return Mesh(points, cells, ele_type=cfg.mesh.ele_type)


def extract_submesh(
    mesh: Mesh,
    geometry: CloakGeometry3D,
) -> tuple[Mesh, np.ndarray]:
    """Remove tetrahedra whose centroid lies inside the defect region.

    Orphan nodes (not referenced by any remaining tet) are removed and cell
    indices renumbered. Returns ``(submesh, kept_nodes)`` where ``kept_nodes``
    maps submesh-node index → original-mesh-node index (useful for
    ``u_sub = u_full[kept_nodes]``).
    """
    import jax.numpy as jnp

    points = np.asarray(mesh.points)
    cells = np.asarray(mesh.cells)

    centroids = points[cells[:, :4]].mean(axis=1)   # 4 corner nodes (TET4 and TET10)

    keep = np.array([
        not bool(geometry.in_defect(jnp.array(c)))
        for c in centroids
    ])
    cells_sub = cells[keep]

    kept_nodes = np.unique(cells_sub)
    new_points = points[kept_nodes]
    old_to_new = np.full(len(points), -1, dtype=int)
    old_to_new[kept_nodes] = np.arange(len(kept_nodes))
    new_cells = old_to_new[cells_sub]

    return Mesh(new_points, new_cells, ele_type=mesh.ele_type), kept_nodes
