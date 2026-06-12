"""Cell decomposition of the cloak region for piecewise-constant material optimisation.

A regular grid of square cells covers the bounding box of the cloak.  Each FEM
quadrature point is mapped (once, at setup time) to the cell that contains it.
Per-cell material values are then expanded to quadrature-point arrays via JAX
fancy indexing, which is differentiable w.r.t. the cell values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.geometry.base import CloakGeometry


@dataclass
class CellDecomposition:
    """Regular grid of square cells over the cloak bounding box.

    Parameters
    ----------
    geometry : CloakGeometry
        Cloak geometry (used for bounding box and ``in_cloak`` queries).
    n_x, n_y : int
        Number of cells in x and y directions.
    """

    geometry: CloakGeometry
    n_x: int
    n_y: int

    # computed in __post_init__
    x_min: float = field(init=False)
    x_max: float = field(init=False)
    y_min: float = field(init=False)
    y_max: float = field(init=False)
    cell_dx: float = field(init=False)
    cell_dy: float = field(init=False)
    cell_centers: np.ndarray = field(init=False)   # (n_cells, 2)
    cloak_mask: np.ndarray = field(init=False)      # (n_cells,) bool

    def __post_init__(self) -> None:
        geo = self.geometry
        # Bounding box of the cloak region
        if hasattr(geo, 'bbox'):
            self.x_min, self.x_max, self.y_min, self.y_max = geo.bbox()
        else:
            # Triangular geometry fallback
            self.x_min = geo.x_c - geo.c
            self.x_max = geo.x_c + geo.c
            self.y_min = geo.y_top - geo.b
            self.y_max = geo.y_top

        self.cell_dx = (self.x_max - self.x_min) / self.n_x
        self.cell_dy = (self.y_max - self.y_min) / self.n_y

        # Build cell centres
        cx = self.x_min + (np.arange(self.n_x) + 0.5) * self.cell_dx
        cy = self.y_min + (np.arange(self.n_y) + 0.5) * self.cell_dy
        gx, gy = np.meshgrid(cx, cy, indexing="ij")  # (n_x, n_y)
        # Flatten in row-major (x varies slowest): cell_idx = ix * n_y + iy
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)

        # Evaluate in_cloak at all centres in one vectorised call
        all_centers = jnp.array(self.cell_centers.T)  # (2, n_cells)
        self.cloak_mask = np.asarray(geo.in_cloak(all_centers), dtype=bool)

    # ── properties ────────────────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        return self.n_x * self.n_y

    @property
    def n_cloak_cells(self) -> int:
        return int(self.cloak_mask.sum())

    @property
    def cloak_cell_indices(self) -> np.ndarray:
        return np.where(self.cloak_mask)[0]

    # ── quadrature-point mapping ──────────────────────────────────────

    def _point_to_cell_index(self, x: np.ndarray) -> int:
        """Map a single 2-D point to its cell index.

        Returns ``n_cells`` (sentinel) if the point is outside the grid.
        """
        ix = int(np.floor((x[0] - self.x_min) / self.cell_dx))
        iy = int(np.floor((x[1] - self.y_min) / self.cell_dy))
        if 0 <= ix < self.n_x and 0 <= iy < self.n_y:
            return ix * self.n_y + iy
        return self.n_cells  # sentinel → background

    def build_qp_mapping(
        self, physical_quad_points: np.ndarray, confine_to_cloak: bool = False,
    ) -> np.ndarray:
        """Precompute FEM-quadrature-point → cell-index mapping.

        Parameters
        ----------
        physical_quad_points : (n_fem_cells, n_qp, 2)
        confine_to_cloak : bool
            If True, also send any quadrature point that is *not* inside the
            cloak annulus (per-point ``geometry.in_cloak`` test) to the
            background sentinel, even when it falls inside a cloak cell's grid
            footprint. Default False keeps the legacy cell-centre masking (a
            cloak cell's material fills its whole rectangular grid cell). Needed
            for coarse grids (esp. 1x1) where a cell footprint spills outside the
            triangular annulus into the bulk half-space.

        Returns
        -------
        qp_to_cell : (n_fem_cells, n_qp) int array.
            Sentinel value ``n_cells`` for points outside the cell grid (and,
            when ``confine_to_cloak``, outside the cloak annulus).
        """
        pts = np.asarray(physical_quad_points)
        n_fem, n_qp, _ = pts.shape
        flat = pts.reshape(-1, 2)

        # Vectorised grid arithmetic
        ix = np.floor((flat[:, 0] - self.x_min) / self.cell_dx).astype(int)
        iy = np.floor((flat[:, 1] - self.y_min) / self.cell_dy).astype(int)
        idx = ix * self.n_y + iy

        outside = (ix < 0) | (ix >= self.n_x) | (iy < 0) | (iy >= self.n_y)
        if confine_to_cloak:
            # Per-point annulus test (in_cloak expects (2, N): x[0]=xs, x[1]=ys).
            in_annulus = np.asarray(self.geometry.in_cloak(flat.T), dtype=bool)
            outside = outside | ~in_annulus
        idx[outside] = self.n_cells  # sentinel → background

        return idx.reshape(n_fem, n_qp)

    # ── expand per-cell values to quadrature points ───────────────────

    @staticmethod
    def expand_to_quadpoints(
        cell_values: jnp.ndarray,
        qp_to_cell: jnp.ndarray,
        background_value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Expand ``(n_cells, ...)`` → ``(n_fem_cells, n_qp, ...)``.

        Appends *background_value* as an extra row so that the sentinel index
        ``n_cells`` maps to the background material.  The indexing is
        differentiable w.r.t. ``cell_values``.
        """
        bg = jnp.expand_dims(background_value, axis=0)  # (1, ...)
        extended = jnp.concatenate([cell_values, bg], axis=0)  # (n_cells+1, ...)
        return extended[qp_to_cell]  # (n_fem_cells, n_qp, ...)

    # ── neighbour topology ────────────────────────────────────────────

    def get_neighbor_pairs(self) -> np.ndarray:
        """Return ``(n_pairs, 2)`` array of 4-connected neighbour pairs.

        Only pairs where **both** cells lie inside the cloak are included.
        """
        pairs = []
        for ix in range(self.n_x):
            for iy in range(self.n_y):
                idx = ix * self.n_y + iy
                if not self.cloak_mask[idx]:
                    continue
                # right neighbour
                if ix + 1 < self.n_x:
                    nb = (ix + 1) * self.n_y + iy
                    if self.cloak_mask[nb]:
                        pairs.append((idx, nb))
                # up neighbour
                if iy + 1 < self.n_y:
                    nb = ix * self.n_y + (iy + 1)
                    if self.cloak_mask[nb]:
                        pairs.append((idx, nb))
        if not pairs:
            return np.empty((0, 2), dtype=int)
        return np.array(pairs, dtype=int)
