"""Periodicity-preserving Chaikin smoothing for CA unit cells.

The cellular-automata generator (:mod:`.generator`) emits binary pixel grids
(1 = material, 0 = void) that are *periodic*: a cell must tile seamlessly with
copies of itself. Their material/void boundaries are pixelated staircases,
which are ugly to mesh and create spurious stress concentrations under
homogenisation.

Chaikin's corner-cutting algorithm smooths a polyline by repeatedly replacing
each vertex with two points at 1/4 and 3/4 of every edge. Applied naively to a
cell's boundary contours it would move the points where material crosses the
cell edge, breaking the periodic boundary condition (the smoothed cell would no
longer tile).

The fix implemented here keeps the periodic BC *exactly*:

1. Tile the cell ``reps x reps`` (default 3x3) and surround it with a thin void
   border, so every contour of the central tile is a closed interior loop with
   its *real* periodic neighbours on all sides.
2. Extract the 0.5 iso-contours and smooth each with Chaikin.
3. Rasterise only the **central** tile (even-odd fill of the smoothed loops).

Because the tiled field is exactly periodic and both Chaikin and even-odd
rasterisation are local, translation-equivariant operations, the central
tile's left edge stays identical to its right edge (and top to bottom). The
boundary is smoothed *through* rather than pinned, so tiling stays seamless
with no kinks.

The shipped raster is a *single* tile, so tiling it (``np.tile``) is
pixel-exact by construction. Rendering two independent interior tiles and
comparing them is only a diagnostic: float point-in-polygon is not perfectly
translation-invariant, so at fine upsampling a handful of boundary pixels
(~0.06%) can disagree between two separate renders. :func:`verify_periodicity`
reports that sub-pixel seam-fidelity error; ``smooth_visualize.py`` renders
samples with a 3x3 tiling to eyeball seam continuity.

The public entry point is :func:`smooth_cell`. Everything is pure
numpy + contourpy (no skimage/cv2/shapely needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from contourpy import LineType, contour_generator
from matplotlib.path import Path
from numpy.typing import NDArray

__all__ = [
    "SmoothResult",
    "chaikin",
    "smooth_cell",
    "rasterize_periodic",
    "verify_periodicity",
]


def chaikin(
    points: NDArray[np.floating],
    iterations: int = 2,
    ratio: float = 0.25,
    closed: bool = True,
) -> NDArray[np.float64]:
    """Chaikin corner-cutting subdivision of a polyline.

    Each iteration replaces every edge ``(A, B)`` with two points
    ``(1-ratio)*A + ratio*B`` and ``ratio*A + (1-ratio)*B``, rounding off
    corners. ``iterations`` doublings of the vertex count converge to a
    quadratic B-spline.

    Args:
        points: ``(N, 2)`` polyline vertices.
        iterations: Number of corner-cutting passes.
        ratio: Cut fraction in ``(0, 0.5)``; 0.25 is the classic Chaikin value.
        closed: If True, treat the polyline as a closed loop (wrap the last
            edge back to the first vertex); a duplicated closing vertex is
            dropped first. If False, the two endpoints are kept fixed.

    Returns:
        The smoothed ``(M, 2)`` polyline (open loops keep their endpoints;
        closed loops are returned without a duplicated closing vertex).
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {P.shape}")
    if not (0.0 < ratio < 0.5):
        raise ValueError(f"ratio must be in (0, 0.5); got {ratio}")

    if closed and len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]

    for _ in range(iterations):
        if len(P) < 2:
            break
        if closed:
            A = P
            B = np.roll(P, -1, axis=0)
            Q = (1.0 - ratio) * A + ratio * B
            R = ratio * A + (1.0 - ratio) * B
            out = np.empty((2 * len(A), 2), dtype=np.float64)
            out[0::2] = Q
            out[1::2] = R
            P = out
        else:
            A = P[:-1]
            B = P[1:]
            Q = (1.0 - ratio) * A + ratio * B
            R = ratio * A + (1.0 - ratio) * B
            inner = np.empty((2 * len(A), 2), dtype=np.float64)
            inner[0::2] = Q
            inner[1::2] = R
            P = np.vstack([P[:1], inner, P[-1:]])
    return P


def _extract_loops(
    field: NDArray[np.floating],
    iterations: int,
    ratio: float,
) -> List[NDArray[np.float64]]:
    """Extract 0.5 iso-contours of ``field`` and Chaikin-smooth each loop.

    Contours are returned in ``(x=col, y=row)`` order, matching image
    plotting conventions. ``field`` is assumed to have a void border so every
    contour is a closed interior loop.
    """
    h, w = field.shape
    gen = contour_generator(
        x=np.arange(w, dtype=np.float64),
        y=np.arange(h, dtype=np.float64),
        z=np.asarray(field, dtype=np.float64),
        line_type=LineType.Separate,
    )
    lines = gen.lines(0.5)  # list of (K, 2) arrays, each closed loop, (x, y)
    loops: List[NDArray[np.float64]] = []
    for line in lines:
        if len(line) >= 3:
            loops.append(chaikin(line, iterations=iterations, ratio=ratio, closed=True))
    return loops


def _padded_loops(
    bin_cell: NDArray[np.floating],
    reps: int,
    pad: int,
    iterations: int,
    ratio: float,
    presmooth_sigma: float,
) -> List[NDArray[np.float64]]:
    """Build smoothed contour loops for a ``reps x reps`` periodic tiling.

    Optionally pre-blurs the cell with a *periodic* (wrap-mode) Gaussian before
    tiling, which rounds the geometry at a larger length scale than Chaikin
    alone. Because the blur uses ``mode='wrap'`` on the single cell it stays
    exactly periodic, so the tiled + zero-padded field is periodic in its
    interior and the extracted loops (after Chaikin) are too.
    """
    field = bin_cell
    if presmooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        field = gaussian_filter(bin_cell, sigma=presmooth_sigma, mode="wrap")
    padded = np.pad(
        np.tile(field, (reps, reps)), pad, mode="constant", constant_values=0.0
    )
    return _extract_loops(padded, iterations=iterations, ratio=ratio)


def _tile_raster(
    loops: List[NDArray[np.float64]],
    ti: int,
    tj: int,
    upsample: int,
    H: int,
    W: int,
    pad: int,
) -> NDArray[np.uint8]:
    """Rasterise one tile ``(ti, tj)`` of the padded field via even-odd fill.

    Pixel centres of an ``H*upsample x W*upsample`` grid covering tile row
    ``ti`` / col ``tj`` (in tiled, pre-pad coordinates, then shifted by ``pad``)
    are tested against every smoothed loop; a point inside an odd number of
    loops is material. Even-odd parity is orientation-independent and fills
    regions-with-holes correctly.
    """
    hu, wu = H * upsample, W * upsample
    ys = (np.arange(hu) + 0.5) / upsample + ti * H + pad - 0.5  # sorted ascending
    xs = (np.arange(wu) + 0.5) / upsample + tj * W + pad - 0.5  # sorted ascending

    parity = np.zeros((hu, wu), dtype=np.int32)
    for lp in loops:
        # bbox-cull: only pixels inside the loop's bounding box can be contained,
        # so restrict the (expensive) point-in-polygon test to that sub-block.
        (x0, y0), (x1, y1) = lp.min(axis=0), lp.max(axis=0)
        c0, c1 = int(np.searchsorted(xs, x0, "left")), int(np.searchsorted(xs, x1, "right"))
        r0, r1 = int(np.searchsorted(ys, y0, "left")), int(np.searchsorted(ys, y1, "right"))
        if c0 >= c1 or r0 >= r1:
            continue  # loop lies entirely off this tile
        sub_x, sub_y = np.meshgrid(xs[c0:c1], ys[r0:r1])
        sub_pts = np.column_stack([sub_x.ravel(), sub_y.ravel()])
        poly = np.vstack([lp, lp[:1]])
        hit = Path(poly, closed=True).contains_points(sub_pts).astype(np.int32)
        parity[r0:r1, c0:c1] += hit.reshape(r1 - r0, c1 - c0)
    return (parity % 2 == 1).astype(np.uint8)


@dataclass
class SmoothResult:
    """Output of :func:`smooth_cell`.

    Attributes:
        raster: ``(H*upsample, W*upsample)`` uint8 smoothed central tile,
            exactly periodic (1 = material, 0 = void).
        loops: Smoothed boundary loops of the central 3x3 neighbourhood, in
            central-tile-local coordinates (``x=col, y=row``; the central tile
            spans ``[0, W] x [0, H]``, loops may extend slightly past the edges
            where material crosses the boundary). Useful for meshing / overlay.
        orig_fraction: Material fraction of the input cell.
        smooth_fraction: Material fraction of ``raster``.
        upsample: Upsampling factor used for the raster.
    """

    raster: NDArray[np.uint8]
    loops: List[NDArray[np.float64]]
    orig_fraction: float
    smooth_fraction: float
    upsample: int


def smooth_cell(
    cell: NDArray[np.integer],
    upsample: int = 4,
    iterations: int = 2,
    ratio: float = 0.25,
    pad: int = 2,
    presmooth_sigma: float = 0.0,
) -> SmoothResult:
    """Chaikin-smooth a periodic CA unit cell, preserving the periodic BC.

    Args:
        cell: ``(H, W)`` binary unit cell (1 = material, 0 = void), assumed
            periodic (tiles seamlessly with itself).
        upsample: Output resolution multiplier. ``upsample=1`` returns a same-
            resolution periodic raster; larger values reveal the smooth
            boundary. Default 4.
        iterations: Chaikin corner-cutting passes (more = smoother). Default 2.
        ratio: Chaikin cut fraction in ``(0, 0.5)``. Default 0.25.
        pad: Void border thickness around the 3x3 tiling. Default 2.
        presmooth_sigma: If > 0, first blur the cell with a periodic (wrap)
            Gaussian of this pixel std-dev. This is the *aggressive* knob: it
            rounds features at a larger length scale than Chaikin can, and
            (unlike Chaikin) can pinch off thin ligaments, so watch material
            fraction and connectivity. Default 0 (Chaikin only).

    Returns:
        A :class:`SmoothResult`. The ``raster`` is guaranteed exactly periodic
        (verify with :func:`verify_periodicity`).
    """
    cell = np.asarray(cell)
    if cell.ndim != 2:
        raise ValueError(f"cell must be 2-D; got shape {cell.shape}")
    H, W = cell.shape
    bin_cell = (cell > 0).astype(np.float64)

    loops = _padded_loops(
        bin_cell, reps=3, pad=pad, iterations=iterations, ratio=ratio,
        presmooth_sigma=presmooth_sigma,
    )

    raster = _tile_raster(loops, ti=1, tj=1, upsample=upsample, H=H, W=W, pad=pad)

    # Re-express loops in central-tile-local coordinates for downstream use.
    offset = np.array([W + pad, H + pad], dtype=np.float64)  # (x, y)
    local_loops = [lp - offset for lp in loops]

    return SmoothResult(
        raster=raster,
        loops=local_loops,
        orig_fraction=float(bin_cell.mean()),
        smooth_fraction=float(raster.mean()),
        upsample=upsample,
    )


def rasterize_periodic(
    cell: NDArray[np.integer],
    upsample: int = 4,
    iterations: int = 2,
    ratio: float = 0.25,
    pad: int = 2,
    presmooth_sigma: float = 0.0,
) -> NDArray[np.uint8]:
    """Convenience wrapper returning only the smoothed periodic raster."""
    return smooth_cell(
        cell, upsample=upsample, iterations=iterations, ratio=ratio, pad=pad,
        presmooth_sigma=presmooth_sigma,
    ).raster


def verify_periodicity(
    cell: NDArray[np.integer],
    upsample: int = 4,
    iterations: int = 2,
    ratio: float = 0.25,
    pad: int = 2,
    tol: float = 5e-3,
    presmooth_sigma: float = 0.0,
) -> Tuple[bool, dict]:
    """Self-check that smoothing preserved the periodic BC (seam fidelity).

    The shipped raster from :func:`smooth_cell` is one tile, so tiling it is
    pixel-exact -- there is nothing to break at the *pixel* level. What this
    checks is *geometric* seam fidelity: does the single output tile faithfully
    reproduce the true multi-tile smooth field across the boundary?

    It tiles the cell 5x5 (so interior tiles have fully real periodic
    neighbourhoods), rasterises three deep-interior tiles from one shared set of
    smoothed contours, and measures the fraction of pixels by which the central
    tile disagrees with its right / lower neighbour (the field's intrinsic
    translation error) and with the cheap 3x3 pipeline output. All three are
    boundary-pixel disagreements that vanish as ``upsample -> inf`` in exact
    arithmetic; float point-in-polygon leaves ~0.06% at ``upsample=6``.

    Args:
        tol: Max allowed disagreement fraction for ``ok`` to be True.

    Returns:
        ``(ok, info)`` where ``ok`` is True iff every error fraction is <= tol,
        and ``info`` holds the fractions and material fractions.
    """
    cell = np.asarray(cell)
    H, W = cell.shape
    bin_cell = (cell > 0).astype(np.float64)

    loops5 = _padded_loops(
        bin_cell, reps=5, pad=pad, iterations=iterations, ratio=ratio,
        presmooth_sigma=presmooth_sigma,
    )

    center = _tile_raster(loops5, 2, 2, upsample, H, W, pad)
    right = _tile_raster(loops5, 2, 3, upsample, H, W, pad)
    lower = _tile_raster(loops5, 3, 2, upsample, H, W, pad)

    res3 = smooth_cell(
        cell, upsample=upsample, iterations=iterations, ratio=ratio, pad=pad,
        presmooth_sigma=presmooth_sigma,
    )

    x_err = float(np.mean(center != right))
    y_err = float(np.mean(center != lower))
    out_err = float(np.mean(res3.raster != center))
    info = {
        "x_seam_err": x_err,
        "y_seam_err": y_err,
        "output_fidelity_err": out_err,
        "orig_fraction": float(bin_cell.mean()),
        "smooth_fraction": float(res3.smooth_fraction),
    }
    ok = max(x_err, y_err, out_err) <= tol
    return ok, info
