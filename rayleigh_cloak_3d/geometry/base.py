"""Protocol for 3D cloak geometries."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


class CloakGeometry3D(Protocol):
    """Every 3D cloak geometry must satisfy this interface."""

    def in_cloak(self, x: jnp.ndarray) -> jnp.ndarray:
        """True if ``x`` (shape (3,)) is inside the annular cloak material region
        (excludes inner defect)."""
        ...

    def in_defect(self, x: jnp.ndarray) -> jnp.ndarray:
        """True if ``x`` is inside the hidden defect (inner cone)."""
        ...

    def F_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deformation gradient (3, 3) of the transformational map at ``x``.

        Must equal ``eye(3)`` everywhere outside the annular cloak region.
        """
        ...

    def bbox(self) -> tuple[float, float, float, float, float, float]:
        """Axis-aligned bounding box of the full cloak volume:
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``."""
        ...

    def build_gmsh_geometry_full(self, occ, box_tag, h_fine, h_elem) -> None:
        """Add refinement / embedded entities around the cloak so the mesh
        of the *full* domain (no defect cutout) is refined near the cloak.

        ``occ`` is ``gmsh.model.occ``; ``box_tag`` is the tag of the outer
        rectangular volume. Geometry implementations may stash a mesh-field
        tag on ``self._cloak_field_tag`` for the mesh builder to compose.
        """
        ...
