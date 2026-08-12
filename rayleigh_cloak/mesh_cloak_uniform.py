"""Uniform-inside-the-CLOAK mesh builder (``mesh.builder = "uniform_cloak"``).

Motivation
----------
For the *pixel-level* validation the microstructure lives inside the cloak
annulus and nowhere else, so that is the only region whose element size has to be
compared against the micro-pixel. The two existing builders both spend their
refinement in the wrong place:

* **legacy** grades the size by *distance from the cloak boundary curves*
  (``Threshold``, ``DistMax = 2b``). ``refinement_factor`` therefore sets the
  size only *at* the boundary; the cloak interior stays much coarser (measured:
  at rf=50 the median element inside the cloak is 2.0e-3 against a 1.33e-3
  pixel — still coarser than a pixel), and the node density inside the cloak is
  re-weighted at every refinement, so a refinement sweep is not a controlled
  experiment.
* **uniform_tri6** is uniform over the cloak *bounding box*, which is ~3x the
  area of the annulus itself (1.33 vs 0.44 here). Two thirds of those elements
  are spent on the defect notch and the bbox corners, which contain no
  microstructure at all.

This builder is uniform ``h_in`` over the **cloak annulus itself** and leaves
everything outside the cloak exactly as the legacy builder had it (the geometry's
own graded ``Threshold`` field still provides the transition away from the cloak,
and the free-surface field is untouched). So::

    elements-per-pixel = pixel / h_in = refinement_factor / (h_elem / pixel)

holds *everywhere in the cloak*, and refinement buys resolution only where the
microstructure actually is.

The annulus indicator is analytic (a ``MathEval`` field) rather than a meshed
sub-surface, because gmsh's expression parser (``mathex``) has no comparison or
ternary operators — it aborts the process on ``(x<0.5) ? a : b``. A smooth step
built from ``sqrt`` alone is used instead::

    S(t) = 0.5 * (1 + t / sqrt(t^2 + eps^2))     ->  0 for t << 0, 1 for t >> 0

and the region test is the product of one step per inequality. The transition
band has width ~``eps`` and is masked anyway by the ``Min`` with the graded
field outside, which equals ``h_in`` on the cloak boundary.
"""

from __future__ import annotations

import gmsh

from rayleigh_cloak import mesh_uniform as _mu
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry
from jax_fem.generate_mesh import Mesh


def _step(t: str, eps: float) -> str:
    """Smooth Heaviside ``S(t) ~ 1 if t > 0 else 0``, from arithmetic + sqrt only."""
    return f"(0.5*(1+({t})/sqrt(({t})^2+{eps!r}^2)))"


def _annulus_indicator(geometry: CloakGeometry, eps: float) -> str:
    """MathEval expression that is ~1 inside the cloak region and ~0 outside."""
    if hasattr(geometry, "a") and hasattr(geometry, "b") and hasattr(geometry, "c"):
        # Triangular cloak: r = |x - x_c| / c <= 1, and the depth below the free
        # surface lies between the inner and outer triangle: a(1-r) <= d <= b(1-r).
        x_c, y_top = geometry.x_c, geometry.y_top
        a, b, c = geometry.a, geometry.b, geometry.c
        r = f"(fabs(x-{x_c!r})/{c!r})"
        d = f"({y_top!r}-y)"
        d1 = f"({a!r}*(1-{r}))"
        d2 = f"({b!r}*(1-{r}))"
        return (f"{_step(f'1-{r}', eps)}"
                f"*{_step(f'{d}-{d1}', eps)}"
                f"*{_step(f'{d2}-{d}', eps)}")

    if hasattr(geometry, "ri") and hasattr(geometry, "rc"):
        # Circular cloak: ri <= |x - x_c| <= rc.
        x_c, y_c = geometry.x_c, geometry.y_c
        ri, rc = geometry.ri, geometry.rc
        rad = f"sqrt((x-{x_c!r})^2+(y-{y_c!r})^2)"
        return f"{_step(f'{rad}-{ri!r}', eps)}*{_step(f'{rc!r}-{rad}', eps)}"

    raise ValueError(
        f"uniform_cloak builder does not know how to describe the cloak region of "
        f"{type(geometry).__name__}; add a branch to _annulus_indicator()."
    )


def _add_uniform_annulus_field(
    geometry: CloakGeometry,
    h_in: float,
    h_out: float,
) -> int:
    """Size field: uniform ``h_in`` inside the cloak, graded as before outside.

    Returns ``Min(MathEval(annulus), geometry._cloak_field_tag)``. The MathEval
    term pins the cloak *interior* to ``h_in``; the geometry's own graded
    ``Threshold`` (built during ``build_gmsh_geometry*``) is kept in the ``Min``
    so the mesh outside the cloak coarsens exactly the way it does today — this
    builder changes the cloak interior and nothing else.
    """
    # Transition width of the smooth step. h_in keeps the boundary sharp; the Min
    # with the graded field (which is h_in at distance 0 from the cloak) means the
    # blur band is never actually seen.
    expr = _annulus_indicator(geometry, eps=float(h_in))
    f_me = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(
        f_me, "F", f"{h_out!r}+({h_in!r}-{h_out!r})*({expr})")

    tags = [f_me]
    graded = getattr(geometry, "_cloak_field_tag", None)
    if graded is not None:
        tags.append(graded)          # unchanged behaviour OUTSIDE the cloak
    if len(tags) == 1:
        return f_me

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", tags)
    return f_min


def generate_mesh(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> Mesh:
    """As :func:`rayleigh_cloak.mesh_uniform.generate_mesh`, uniform over the annulus."""
    return _mu.generate_mesh(cfg, params, geometry,
                             cloak_field_fn=_add_uniform_annulus_field)


def generate_mesh_full(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> Mesh:
    """As :func:`rayleigh_cloak.mesh_uniform.generate_mesh_full`, uniform over the annulus."""
    return _mu.generate_mesh_full(cfg, params, geometry,
                                  cloak_field_fn=_add_uniform_annulus_field)
