"""Circular annular cloak geometry (Nassar et al., 2018).

The circular cloak maps a point at the origin to a circle of radius ri,
creating a void surrounded by an annular cloaking region [ri, rc]. The
coordinate transformation is the standard Pendry-type blow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import jax.numpy as jnp
import numpy as np


@dataclass
class CircularCloakGeometry:
    """Circular annular cloak centred at (x_c, y_c).

    Parameters
    ----------
    ri : Inner (void) radius.
    rc : Outer (cloak boundary) radius.
    x_c : Centre x in extended-mesh coordinates.
    y_c : Centre y in extended-mesh coordinates.
    y_top : Top-of-domain y (free surface). Required only when chaining
            forced eval x-positions through the top edge; falls back to
            ``y_c + 2 * rc + 1e-30`` (a dummy that triggers an error if
            actually used) so existing callers that don't set it still
            work for the legacy single-line top edge.
    """

    ri: float
    rc: float
    x_c: float
    y_c: float
    y_top: float | None = None

    # ── region membership (JAX-traceable) ─────────────────────────────

    def in_cloak(self, x: jnp.ndarray) -> jnp.ndarray:
        dx = x[0] - self.x_c
        dy = x[1] - self.y_c
        r = jnp.sqrt(dx**2 + dy**2)
        return (r > self.ri) & (r < self.rc)

    def in_defect(self, x: jnp.ndarray) -> jnp.ndarray:
        dx = x[0] - self.x_c
        dy = x[1] - self.y_c
        r = jnp.sqrt(dx**2 + dy**2)
        return r < self.ri

    # ── coordinate transformation ─────────────────────────────────────

    def F_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deformation gradient for the circular blow-up (Cartesian frame).

        The blow-up maps reference coordinates X to physical coordinates x:
            x = [rᵢ/‖X‖ + (rᶜ − rᵢ)/rᶜ] · X

        In terms of the physical point, F = ∂x/∂X expressed as:
            F_ij = (1/R) · (r · δ_ij − rᵢ · x̂_i · x̂_j)

        where r = ‖x − centre‖, R = rᶜ(r − rᵢ)/(rᶜ − rᵢ) is the
        original radius, and x̂ = (x − centre)/r.
        """
        ri, rc = self.ri, self.rc
        dx = x[0] - self.x_c
        dy = x[1] - self.y_c
        r = jnp.sqrt(dx**2 + dy**2)
        r_safe = jnp.maximum(r, 1e-30)  # avoid division by zero

        # Original radius
        R = rc * (r_safe - ri) / (rc - ri)
        R_safe = jnp.maximum(R, 1e-30)

        # Unit direction vector
        x_hat = jnp.array([dx, dy]) / r_safe

        I2 = jnp.eye(2)
        F_cloak = (1.0 / R_safe) * (r_safe * I2 - ri * jnp.outer(x_hat, x_hat))

        return jnp.where(self.in_cloak(x), F_cloak, I2)

    # ── bounding box for CellDecomposition ────────────────────────────

    def bbox(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) of the cloak region."""
        return (
            self.x_c - self.rc,
            self.x_c + self.rc,
            self.y_c - self.rc,
            self.y_c + self.rc,
        )

    # ── gmsh geometry construction ────────────────────────────────────

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
        """Cut out the circular void and add cloak refinement fields.

        ``h_outside`` (default = ``h_elem``) is the target mesh size beyond
        the cloak's distance threshold; pass a value > h_elem to coarsen
        the bulk mesh away from cloak/surface.

        ``top_eval_xs`` / ``h_surf``: when provided, the top edge is built
        as a chain of lines through these forced x-positions so each is a
        real triangle vertex on the free surface.
        """
        if h_outside is None:
            h_outside = h_elem
        if h_surf is None:
            h_surf = h_fine
        p1, p2, p3, p4 = rect_points
        from rayleigh_cloak.mesh import _chain_top_edge

        # Centre and circle points
        pc = geo.addPoint(self.x_c, self.y_c, 0.0, h_fine)
        # Four points on the inner circle (void boundary)
        pr = geo.addPoint(self.x_c + self.ri, self.y_c, 0.0, h_fine)
        pt = geo.addPoint(self.x_c, self.y_c + self.ri, 0.0, h_fine)
        pl = geo.addPoint(self.x_c - self.ri, self.y_c, 0.0, h_fine)
        pb = geo.addPoint(self.x_c, self.y_c - self.ri, 0.0, h_fine)

        # Four quarter-circle arcs (void boundary)
        a1 = geo.addCircleArc(pr, pc, pt)
        a2 = geo.addCircleArc(pt, pc, pl)
        a3 = geo.addCircleArc(pl, pc, pb)
        a4 = geo.addCircleArc(pb, pc, pr)

        # Rectangle edges (top edge optionally chained through eval xs)
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        if top_eval_xs is not None:
            top_lines = _chain_top_edge(
                geo, p3, p4, np.asarray(top_eval_xs), self.y_top, h_surf,
                descending=True,
            )
        else:
            top_lines = [geo.addLine(p3, p4)]
        l_left = geo.addLine(p4, p1)

        # Outer boundary loop
        outer_loop = geo.addCurveLoop(
            [l_bot, l_right] + top_lines + [l_left]
        )
        # Inner (void) loop — reversed orientation
        inner_loop = geo.addCurveLoop([a1, a2, a3, a4])
        surf = geo.addPlaneSurface([outer_loop, inner_loop])

        gmsh.model.geo.synchronize()

        # Points on the outer cloak circle for refinement
        por = geo.addPoint(self.x_c + self.rc, self.y_c, 0.0, h_fine)
        pot = geo.addPoint(self.x_c, self.y_c + self.rc, 0.0, h_fine)
        pol = geo.addPoint(self.x_c - self.rc, self.y_c, 0.0, h_fine)
        pob = geo.addPoint(self.x_c, self.y_c - self.rc, 0.0, h_fine)
        geo.synchronize()
        gmsh.model.mesh.embed(0, [por, pot, pol, pob], 2, surf)

        # ── refinement fields ──
        # Distance from inner circle
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_inner, "CurvesList", [a1, a2, a3, a4])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 200)

        # Distance from outer cloak circle points
        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_outer, "PointsList", [por, pot, pol, pob])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_dist_cloak, "FieldsList", [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_outside)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", self.rc)

        self._cloak_field_tag = f_thresh_cloak

        return top_lines

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
        """Build full domain mesh (no void cutout) with circle points embedded.

        The inner circle points are embedded so the mesh refines around the
        cloak region. Elements inside the void can be removed later via
        ``extract_submesh``.

        See ``build_gmsh_geometry`` for ``top_eval_xs`` / ``h_surf`` semantics.
        """
        if h_outside is None:
            h_outside = h_elem
        if h_surf is None:
            h_surf = h_fine
        p1, p2, p3, p4 = rect_points
        from rayleigh_cloak.mesh import _chain_top_edge

        # Rectangle edges (top edge optionally chained through eval xs)
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        if top_eval_xs is not None:
            top_lines = _chain_top_edge(
                geo, p3, p4, np.asarray(top_eval_xs), self.y_top, h_surf,
                descending=True,
            )
        else:
            top_lines = [geo.addLine(p3, p4)]
        l_left = geo.addLine(p4, p1)

        outer_loop = geo.addCurveLoop(
            [l_bot, l_right] + top_lines + [l_left]
        )
        surf = geo.addPlaneSurface([outer_loop])

        gmsh.model.geo.synchronize()

        # Embed inner circle as constraint lines for mesh conformity
        pc = geo.addPoint(self.x_c, self.y_c, 0.0, h_fine)
        pr = geo.addPoint(self.x_c + self.ri, self.y_c, 0.0, h_fine)
        pt = geo.addPoint(self.x_c, self.y_c + self.ri, 0.0, h_fine)
        pl = geo.addPoint(self.x_c - self.ri, self.y_c, 0.0, h_fine)
        pb = geo.addPoint(self.x_c, self.y_c - self.ri, 0.0, h_fine)

        a1 = geo.addCircleArc(pr, pc, pt)
        a2 = geo.addCircleArc(pt, pc, pl)
        a3 = geo.addCircleArc(pl, pc, pb)
        a4 = geo.addCircleArc(pb, pc, pr)

        # Outer cloak circle points
        por = geo.addPoint(self.x_c + self.rc, self.y_c, 0.0, h_fine)
        pot = geo.addPoint(self.x_c, self.y_c + self.rc, 0.0, h_fine)
        pol = geo.addPoint(self.x_c - self.rc, self.y_c, 0.0, h_fine)
        pob = geo.addPoint(self.x_c, self.y_c - self.rc, 0.0, h_fine)

        geo.synchronize()
        gmsh.model.mesh.embed(0, [pc, por, pot, pol, pob], 2, surf)
        gmsh.model.mesh.embed(1, [a1, a2, a3, a4], 2, surf)

        # ── refinement fields ──
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_inner, "CurvesList", [a1, a2, a3, a4])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 200)

        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_outer, "PointsList", [por, pot, pol, pob])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_dist_cloak, "FieldsList", [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_outside)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", self.rc)

        self._cloak_field_tag = f_thresh_cloak

        return top_lines
