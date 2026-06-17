"""Test that index-based and position-based node correspondence agree
for top-surface nodes used by the transmission loss metric.

The cloak mesh is a submesh of the full mesh (defect elements removed,
orphan nodes dropped).  `kept_nodes[i]` maps cloak-mesh node `i` back to
the full-mesh index.  This test verifies:

1. Positional identity: cloak_mesh.points[i] == full_mesh.points[kept_nodes[i]]
   for every top-surface node (exact, by construction of extract_submesh).

2. No aliasing: no two distinct cloak-mesh surface nodes map to the same
   full-mesh node.

3. Reference surface coverage: the full-mesh positions at
   kept_nodes[cloak_surface_idx] match the cloak-mesh surface positions,
   so the displacement ratio <|u_cloak|>/<|u_ref|> compares the same
   physical locations.
"""

import numpy as np
import pytest

from rayleigh_cloak import load_config
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.solver import _create_geometry


@pytest.fixture(scope="module")
def meshes_and_geometry():
    """Build full mesh, cloak submesh, and geometry from default config."""
    config = load_config("configs/default.yaml")
    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)
    full_mesh = generate_mesh_full(config, dp, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    return full_mesh, cloak_mesh, kept_nodes, geometry, dp


def test_position_identity(meshes_and_geometry):
    """cloak_mesh.points[i] must exactly equal full_mesh.points[kept_nodes[i]]."""
    full_mesh, cloak_mesh, kept_nodes, geometry, dp = meshes_and_geometry

    cloak_surface_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top,
        dp.x_off, dp.x_off + dp.W,
    )
    assert len(cloak_surface_idx) > 0, "No surface nodes found"

    cloak_pts = np.asarray(cloak_mesh.points[cloak_surface_idx])
    full_pts = np.asarray(full_mesh.points[kept_nodes[cloak_surface_idx]])

    np.testing.assert_array_equal(
        cloak_pts, full_pts,
        err_msg="Index-based mapping disagrees with positions",
    )


def test_no_aliasing(meshes_and_geometry):
    """Distinct cloak-mesh surface nodes must map to distinct full-mesh nodes."""
    full_mesh, cloak_mesh, kept_nodes, geometry, dp = meshes_and_geometry

    cloak_surface_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top,
        dp.x_off, dp.x_off + dp.W,
    )
    ref_idx = kept_nodes[cloak_surface_idx]
    assert len(ref_idx) == len(np.unique(ref_idx)), (
        "Multiple cloak-mesh surface nodes map to the same full-mesh node"
    )


def test_full_mesh_surface_is_superset(meshes_and_geometry):
    """Every cloak-mesh surface node should also be on the full-mesh surface."""
    full_mesh, cloak_mesh, kept_nodes, geometry, dp = meshes_and_geometry

    cloak_surface_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top,
        dp.x_off, dp.x_off + dp.W,
    )
    full_surface_idx = get_top_surface_beyond_cloak_indices(
        full_mesh.points, geometry, dp.y_top,
        dp.x_off, dp.x_off + dp.W,
    )

    # The mapped cloak indices should be a subset of the full surface indices
    ref_idx = set(kept_nodes[cloak_surface_idx].tolist())
    full_idx = set(full_surface_idx.tolist())
    missing = ref_idx - full_idx
    assert len(missing) == 0, (
        f"{len(missing)} cloak surface nodes not found on full-mesh surface: "
        f"sample positions {np.asarray(full_mesh.points)[list(missing)[:5]]}"
    )


def test_position_based_lookup_agrees(meshes_and_geometry):
    """Cross-check: find full-mesh nodes by position and verify they match
    the index-based mapping.  This is the actual question — could a naive
    position-based lookup give different results?"""
    full_mesh, cloak_mesh, kept_nodes, geometry, dp = meshes_and_geometry

    cloak_surface_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top,
        dp.x_off, dp.x_off + dp.W,
    )
    assert len(cloak_surface_idx) > 0

    cloak_pts = np.asarray(cloak_mesh.points[cloak_surface_idx])
    full_pts = np.asarray(full_mesh.points)

    # For each cloak surface node, find the nearest full-mesh node by position
    from scipy.spatial import cKDTree
    tree = cKDTree(full_pts)
    dists, pos_based_idx = tree.query(cloak_pts)

    # Index-based mapping
    idx_based = kept_nodes[cloak_surface_idx]

    # They should agree exactly (dist should be 0)
    assert np.all(dists < 1e-12), (
        f"Max positional mismatch: {dists.max():.2e}"
    )
    np.testing.assert_array_equal(
        idx_based, pos_based_idx,
        err_msg=(
            "Position-based nearest-neighbor lookup disagrees with "
            "index-based kept_nodes mapping"
        ),
    )
