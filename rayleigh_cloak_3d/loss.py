"""Loss-target selection for 3D cloak optimisation.

Given a mesh and domain parameters, produces a boolean / index array of
nodes to compare against the reference solve.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rayleigh_cloak_3d.config import DerivedParams3D, SimulationConfig3D
from rayleigh_cloak_3d.geometry.base import CloakGeometry3D


def _top_surface_mask(points: np.ndarray, params: DerivedParams3D, tol: float = 1e-8) -> np.ndarray:
    return np.abs(points[:, 2] - params.z_top) < tol


def _outside_cloak_mask(points: np.ndarray, geometry: CloakGeometry3D) -> np.ndarray:
    """True for nodes *not* inside the outer cloak cone."""
    pts_j = jnp.asarray(points)
    in_c = np.asarray(jnp.vectorize(
        geometry.in_cloak, signature="(n)->()"
    )(pts_j))
    return ~in_c


def _right_boundary_mask(points: np.ndarray, params: DerivedParams3D, tol: float = 1e-8) -> np.ndarray:
    """Physical-domain x = x_off + W face (excluding PML neighbours)."""
    x_right = params.x_off + params.W
    return np.abs(points[:, 0] - x_right) < tol


def resolve_loss_target(
    cfg: SimulationConfig3D,
    params: DerivedParams3D,
    geometry: CloakGeometry3D,
    mesh_points: np.ndarray,
) -> np.ndarray:
    """Return node indices to use as the cloaking-loss target set."""
    pts = np.asarray(mesh_points)
    t = cfg.loss.type
    if t == "top_surface":
        mask = _top_surface_mask(pts, params)
    elif t == "outside_cloak":
        mask = _outside_cloak_mask(pts, geometry)
    elif t == "right_boundary":
        mask = _right_boundary_mask(pts, params)
    else:
        raise ValueError(f"Unknown loss.type: {t}")
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError(
            f"Loss target '{t}' matched zero mesh nodes; check geometry/tolerances."
        )
    return idx
