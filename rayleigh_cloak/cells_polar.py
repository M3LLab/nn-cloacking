"""Polar cell decomposition for the circular cloak region.

A polar grid of N angular sectors × M radial layers covers the annular
cloak region [ri, rc].  This matches the physical layout of the Nassar 2018
degenerate polar lattice, unlike the Cartesian grid in ``cells.py``.

Each FEM quadrature point is mapped (once, at setup time) to the polar cell
that contains it.  Per-cell material values are then expanded to
quadrature-point arrays via JAX fancy indexing (differentiable w.r.t. cell
values).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np


@dataclass
class PolarCellDecomposition:
    """Polar grid of cells over the annular cloak region [ri, rc].

    Parameters
    ----------
    ri : Inner (void) radius.
    rc : Outer (cloak boundary) radius.
    x_c, y_c : Centre of the cloak in extended-mesh coordinates.
    N : Number of angular sectors.
    M : Number of radial layers.
    """

    ri: float
    rc: float
    x_c: float
    y_c: float
    N: int
    M: int

    # computed in __post_init__
    r_edges: np.ndarray = field(init=False)      # (M+1,)
    phi_edges: np.ndarray = field(init=False)     # (N+1,)
    r_centers: np.ndarray = field(init=False)     # (M,)
    phi_centers: np.ndarray = field(init=False)   # (N,)
    cell_centers: np.ndarray = field(init=False)  # (n_cells, 2) Cartesian
    cell_r: np.ndarray = field(init=False)        # (n_cells,) radial position
    cell_phi: np.ndarray = field(init=False)      # (n_cells,) angular position
    cell_a: np.ndarray = field(init=False)        # (n_cells,) radial cell dimension
    cell_b: np.ndarray = field(init=False)        # (n_cells,) tangential cell dimension
    cloak_mask: np.ndarray = field(init=False)    # (n_cells,) bool — all True

    def __post_init__(self) -> None:
        ri, rc = self.ri, self.rc
        N, M = self.N, self.M

        # Radial and angular edges
        self.r_edges = np.linspace(ri, rc, M + 1)
        self.phi_edges = np.linspace(0.0, 2.0 * np.pi, N + 1)

        # Cell centres in polar coordinates
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])  # (M,)
        self.phi_centers = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])  # (N,)

        # Build per-cell arrays — cell index = j * N + k  (layer j, sector k)
        r_grid, phi_grid = np.meshgrid(
            self.r_centers, self.phi_centers, indexing="ij"
        )  # (M, N)
        self.cell_r = r_grid.ravel()        # (M*N,)
        self.cell_phi = phi_grid.ravel()     # (M*N,)

        # Cartesian centres
        cx = self.x_c + self.cell_r * np.cos(self.cell_phi)
        cy = self.y_c + self.cell_r * np.sin(self.cell_phi)
        self.cell_centers = np.stack([cx, cy], axis=-1)  # (M*N, 2)

        # Physical cell dimensions (Nassar 2018 eq 2.18)
        # b = arc length = 2π r / N  (tangential)
        # a = radial extent = (rc - ri) / M  (uniform radial spacing)
        self.cell_b = 2.0 * np.pi * self.cell_r / N
        self.cell_a = np.full_like(self.cell_r, (rc - ri) / M)

        # All cells are inside the cloak by construction
        self.cloak_mask = np.ones(self.n_cells, dtype=bool)

    # ── properties ────────────────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        return self.M * self.N

    @property
    def n_cloak_cells(self) -> int:
        return self.n_cells  # all cells are in cloak

    @property
    def cloak_cell_indices(self) -> np.ndarray:
        return np.arange(self.n_cells)

    # ── quadrature-point mapping ──────────────────────────────────────

    def build_qp_mapping(self, physical_quad_points: np.ndarray) -> np.ndarray:
        """Precompute FEM-quadrature-point → cell-index mapping.

        Parameters
        ----------
        physical_quad_points : (n_fem_cells, n_qp, 2)

        Returns
        -------
        qp_to_cell : (n_fem_cells, n_qp) int array.
            Sentinel value ``n_cells`` for points outside the annulus.
        """
        pts = np.asarray(physical_quad_points)
        n_fem, n_qp, _ = pts.shape
        flat = pts.reshape(-1, 2)

        dx = flat[:, 0] - self.x_c
        dy = flat[:, 1] - self.y_c
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) % (2.0 * np.pi)  # [0, 2π)

        # Bin into radial layers and angular sectors
        dr = (self.rc - self.ri) / self.M
        dphi = 2.0 * np.pi / self.N

        j = np.floor((r - self.ri) / dr).astype(int)       # radial layer
        k = np.floor(phi / dphi).astype(int)                # angular sector

        # Clamp sector index (phi = 2π maps to sector N, wrap to 0)
        k = np.clip(k, 0, self.N - 1)

        idx = j * self.N + k

        # Mark points outside the annulus [ri, rc] as sentinel
        outside = (j < 0) | (j >= self.M)
        idx[outside] = self.n_cells

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

        Angular neighbours wrap around (sector 0 ↔ sector N-1).
        """
        M, N = self.M, self.N
        pairs = []
        for j in range(M):
            for k in range(N):
                idx = j * N + k
                # Radial neighbour (outward)
                if j + 1 < M:
                    pairs.append((idx, (j + 1) * N + k))
                # Angular neighbour (counterclockwise, with wrap)
                k_next = (k + 1) % N
                pairs.append((idx, j * N + k_next))
        if not pairs:
            return np.empty((0, 2), dtype=int)
        return np.array(pairs, dtype=int)

    # ── deformation parameter per cell ────────────────────────────────

    @property
    def cell_f(self) -> np.ndarray:
        """Deformation parameter f = (r - ri) / r for each cell."""
        return (self.cell_r - self.ri) / self.cell_r
