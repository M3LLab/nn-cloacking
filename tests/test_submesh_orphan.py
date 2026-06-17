"""Test that extract_submesh produces no orphan nodes."""
import numpy as np
import pytest

from rayleigh_cloak.config import DerivedParams, load_config
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh


@pytest.fixture
def setup():
    """Load config and create the full mesh + submesh."""
    cfg = load_config("configs/optimize.yaml")
    params = DerivedParams.from_config(cfg)
    geo = TriangularCloakGeometry.from_params(params)
    full_mesh = generate_mesh_full(cfg, params, geo)
    submesh, kept_nodes = extract_submesh(full_mesh, geo)
    return full_mesh, submesh, kept_nodes, geo, params


def test_no_orphan_nodes(setup):
    """After extract_submesh, every node must belong to at least one element."""
    _, submesh, _, _, _ = setup
    cells = np.asarray(submesh.cells)
    used_nodes = set(cells.ravel())
    n_points = len(submesh.points)
    orphan_nodes = set(range(n_points)) - used_nodes
    assert len(orphan_nodes) == 0, f"{len(orphan_nodes)} orphan nodes remain"


def test_kept_nodes_mapping(setup):
    """kept_nodes maps submesh indices to full-mesh positions."""
    full_mesh, submesh, kept_nodes, _, _ = setup
    full_pts = np.asarray(full_mesh.points)
    sub_pts = np.asarray(submesh.points)
    np.testing.assert_allclose(sub_pts, full_pts[kept_nodes], atol=1e-10)


def test_boundary_nodes_preserved(setup):
    """Right PML boundary nodes must exist in the submesh."""
    _, submesh, kept_nodes, _, params = setup
    # Use mesh-edge boundary (Dirichlet BC location), not physical-PML interface
    x_right = params.W_total
    sub_pts = np.asarray(submesh.points)
    boundary_mask = np.abs(sub_pts[:, 0] - x_right) < 1e-6
    assert boundary_mask.sum() > 0, "No right boundary nodes in submesh"
