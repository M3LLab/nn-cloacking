"""Axisymmetric conical cloak geometry (3D analog of the 2D triangular cloak).

The outer cone has apex at ``(x_c, y_c, z_top - b)`` and base (radius ``c``) at
``z = z_top``. The inner cone (hidden defect) has apex at ``(x_c, y_c, z_top - a)``
and base radius ``a * c / b``... actually it shares the same surface radius
proportionality: at a given depth ``d``, the inner-boundary radius is
``c (1 - d/a)`` and the outer-boundary radius is ``c (1 - d/b)``.

Equivalently, using a normalised radius ``r_hat = r / c``:
  - in full cloak (outer cone):   ``r_hat <= 1``  and  ``0 <= depth <= b (1 - r_hat)``
  - in defect (inner cone):        ``r_hat <= 1``  and  ``0 <= depth <= a (1 - r_hat)``
  - in annular material region:    the set difference (cloak minus defect)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from rayleigh_cloak_3d.config import DerivedParams3D


@dataclass
class ConicalCloakGeometry:
    """Conical cloak parameters (in extended-mesh coords)."""

    a: float      # inner cone depth
    b: float      # outer cone depth
    c: float      # surface radius
    x_c: float    # axis x
    y_c: float    # axis y
    z_top: float  # free-surface z

    @staticmethod
    def from_params(p: DerivedParams3D) -> "ConicalCloakGeometry":
        return ConicalCloakGeometry(
            a=p.a, b=p.b, c=p.c, x_c=p.x_c, y_c=p.y_c, z_top=p.z_top,
        )

    def bbox(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.x_c - self.c, self.x_c + self.c,
            self.y_c - self.c, self.y_c + self.c,
            self.z_top - self.b, self.z_top,
        )

    # ── region membership (JAX-traceable) ────────────────────────────

    def _r_hat_and_depth(self, x: jnp.ndarray):
        dx = x[0] - self.x_c
        dy = x[1] - self.y_c
        r = jnp.sqrt(dx * dx + dy * dy)
        depth = self.z_top - x[2]
        r_hat = r / self.c
        return r, r_hat, depth

    def in_cloak(self, x: jnp.ndarray) -> jnp.ndarray:
        _, r_hat, depth = self._r_hat_and_depth(x)
        d_inner = self.a * (1.0 - r_hat)
        d_outer = self.b * (1.0 - r_hat)
        return (r_hat <= 1.0) & (depth >= d_inner) & (depth <= d_outer)

    def in_defect(self, x: jnp.ndarray) -> jnp.ndarray:
        _, r_hat, depth = self._r_hat_and_depth(x)
        d_inner = self.a * (1.0 - r_hat)
        return (r_hat <= 1.0) & (depth >= 0.0) & (depth <= d_inner)

    # ── deformation gradient ─────────────────────────────────────────

    def F_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deformation gradient F = ∂x_phys/∂X_ref in the annular region.

        In the annulus, the reference (virtual) cone is vertically stretched
        and radially unchanged. In cylindrical coordinates the non-trivial
        entries are F_zz = (b - a)/b and F_zr = a/c. Rotated to Cartesian:

            F = [[1, 0, 0],
                 [0, 1, 0],
                 [(a/c) dx/r, (a/c) dy/r, (b - a)/b]]

        on the annulus; F = I elsewhere.
        """
        dx = x[0] - self.x_c
        dy = x[1] - self.y_c
        r = jnp.sqrt(dx * dx + dy * dy)
        r_hat = r / self.c
        depth = self.z_top - x[2]

        # guard against r = 0 (on the axis); the annulus does touch the axis
        # at depths a..b, but the limit F_zr * dx/r → 0 there
        r_safe = jnp.maximum(r, 1e-30)
        F_zr = self.a / self.c

        F_cloak = jnp.array([
            [1.0,                 0.0,                 0.0],
            [0.0,                 1.0,                 0.0],
            [F_zr * dx / r_safe,  F_zr * dy / r_safe,  (self.b - self.a) / self.b],
        ])

        d_inner = self.a * (1.0 - r_hat)
        d_outer = self.b * (1.0 - r_hat)
        in_annulus = (r_hat <= 1.0) & (depth >= d_inner) & (depth <= d_outer)
        return jnp.where(in_annulus, F_cloak, jnp.eye(3))

    # ── gmsh construction ────────────────────────────────────────────

    def build_gmsh_geometry_full(self, occ, box_tag, h_fine, h_elem) -> None:
        """Refine mesh around the conical cloak (no defect cutout).

        We add a distance field centred on the cloak axis segment plus the
        base circle, with a threshold ramp so that elements near the cloak
        are of size ``h_fine``. The mesh builder composes this field with
        surface refinement to form the final size map.
        """
        import gmsh

        # Point at the apex of the outer cone (for distance-field seeding).
        apex_pt = occ.addPoint(self.x_c, self.y_c, self.z_top - self.b, h_fine)
        # Point at the inner apex as well (finer near defect edge).
        inner_apex_pt = occ.addPoint(self.x_c, self.y_c, self.z_top - self.a, h_fine)
        occ.synchronize()
        # Embed these points into the outer box so the mesh respects them.
        gmsh.model.mesh.embed(0, [apex_pt, inner_apex_pt], 3, box_tag)

        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList",
                                         [apex_pt, inner_apex_pt])

        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h_elem)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", self.b * 2.0)

        self._cloak_field_tag = f_thresh
