"""Test that extract_submesh produces no orphan nodes."""
import jax.numpy as jnp
import numpy as np

from jax_fem.generate_mesh import Mesh
from rayleigh_cloak.mesh import extract_submesh
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry


def _make_simple_mesh_and_geometry():
    """Create a tiny mesh with a defect region to test submesh extraction."""
    geo = TriangularCloakGeometry(a=0.5, b=1.5, c=1.0, x_c=2.0, y_top=3.0)

    # 4x3 grid of points → triangular mesh
    xs = np.linspace(0.0, 4.0, 5)
    ys = np.linspace(0.0, 3.0, 4)
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    # Create TRI3 elements from the grid
    cells = []
    nx, ny = 5, 4
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            cells.append([n0, n1, n2])
            cells.append([n1, n3, n2])
    cells = np.array(cells)

    mesh = Mesh(points, cells, ele_type="TRI3")
    return mesh, geo


def test_no_orphan_nodes():
    """After extract_submesh, every node must belong to at least one element."""
    mesh, geo = _make_simple_mesh_and_geometry()
    submesh, kept_nodes = extract_submesh(mesh, geo)

    # Check which nodes are referenced by the remaining elements
    used_nodes = set(np.asarray(submesh.cells).ravel())
    n_points = len(submesh.points)
    orphan_nodes = set(range(n_points)) - used_nodes

    assert len(orphan_nodes) == 0, (
        f"{len(orphan_nodes)} orphan nodes (no connecting elements): "
        f"{sorted(orphan_nodes)[:10]}..."
    )


def test_shared_boundary_nodes_preserved():
    """Boundary node indices must still point to the same physical locations."""
    mesh, geo = _make_simple_mesh_and_geometry()
    submesh, kept_nodes = extract_submesh(mesh, geo)

    # Right boundary nodes
    x_right = 4.0
    tol = 1e-6
    orig_right = np.where(np.abs(np.asarray(mesh.points)[:, 0] - x_right) < tol)[0]
    sub_points = np.asarray(submesh.points)

    # Every original right-boundary node should still exist at the same position
    for idx in orig_right:
        orig_pos = np.asarray(mesh.points)[idx]
        # Find matching position in submesh
        dists = np.linalg.norm(sub_points - orig_pos, axis=1)
        assert np.min(dists) < tol, (
            f"Right boundary node {idx} at {orig_pos} lost in submesh"
        )
