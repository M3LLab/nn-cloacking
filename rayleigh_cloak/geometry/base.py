"""Abstract interface for cloak geometries.

Every geometry must provide:
  - ``in_cloak`` / ``in_defect`` — region membership (JAX-traceable)
  - ``F_tensor`` — deformation gradient of the coordinate transformation
  - ``build_gmsh_geometry`` — adds the defect cutout + refinement fields to a
    gmsh model and returns the list of top-surface line tags.
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


class CloakGeometry(Protocol):
    """Protocol that every cloak geometry must satisfy."""

    def in_cloak(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return True (as a JAX bool) if *x* lies inside the cloak annulus."""
        ...

    def in_defect(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return True if *x* lies inside the hidden void."""
        ...

    def F_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deformation gradient of the coordinate transformation at *x*.

        Returns ``(2, 2)`` array.  Must equal ``eye(2)`` outside the cloak.
        """
        ...

    def build_gmsh_geometry(
        self,
        geo,
        rect_points: tuple[int, int, int, int],
        h_fine: float,
        h_elem: float,
        h_outside: float | None = None,
        top_eval_xs=None,
        h_surf: float | None = None,
    ) -> list[int]:
        """Add geometry-specific features to the gmsh model (with defect cutout).

        Parameters
        ----------
        geo : gmsh.model.geo handle
        rect_points : (p1, p2, p3, p4) — bottom-left, bottom-right,
                       top-right, top-left corner point tags.
        h_fine : target mesh size near the cloak
        h_elem : reference (legacy "outside") mesh size — also used as the
                 characteristic length for corner points.
        h_outside : target mesh size *outside* the cloak. ``None`` falls back
                    to ``h_elem`` for backwards compatibility.
        top_eval_xs : optional ndarray of forced top-edge x-positions. When
                    provided, the top edge is constructed as a chain of
                    lines through these positions so each is a real
                    triangle vertex (mesh-independent eval surface metric).
        h_surf : characteristic length for the chained top-edge vertices;
                 falls back to ``h_fine``.

        Returns
        -------
        top_lines : list of gmsh line tags composing the top boundary
                    (for surface-refinement and Neumann BC identification).
        """
        ...

    def build_gmsh_geometry_full(
        self,
        geo,
        rect_points: tuple[int, int, int, int],
        h_fine: float,
        h_elem: float,
        h_outside: float | None = None,
        top_eval_xs=None,
        h_surf: float | None = None,
    ) -> list[int]:
        """Build full domain (no defect cutout) with cloak vertices embedded.

        The mesh includes the defect region so that both reference and cloak
        solves can share the same node positions.  Elements inside the defect
        are later removed via ``extract_submesh``.

        Parameters / Returns same as ``build_gmsh_geometry``.
        """
        ...
