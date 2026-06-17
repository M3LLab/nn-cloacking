"""Material-field abstraction for 3D cloak optimisation.

A :class:`MaterialField` maps a parameter tree ``theta`` (MLP weights) to
``(C_qp, rho_qp)`` arrays at FEM quadrature points. The FEM problem is
agnostic to the choice — the two concrete implementations differ only in
where the MLP is evaluated:

- :class:`ContinuousNeuralField`: at every physical quadrature point.
- :class:`CellDecomposedNeuralField`: at the centres of a regular grid
  over the cloak bounding box; each quadrature point inherits its cell's
  value via precomputed indices.

Both share the same MLP, Fourier features, and multiplicative-residual
decoding pattern as the 2D :class:`rayleigh_cloak.neural_reparam.NeuralReparam`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak_3d.geometry.base import CloakGeometry3D
from rayleigh_cloak_3d.materials import (
    C_eff_3d,
    _get_converters,
    rho_eff_3d,
    symmetrize_stiffness_3d,
)
from rayleigh_cloak_3d.neural import fourier_features_3d, init_mlp, mlp_forward


# ── protocol ─────────────────────────────────────────────────────────


class MaterialField(Protocol):
    """Maps MLP weights → ``(C_qp, rho_qp)`` at FEM quadrature points.

    Implementations must be JAX-differentiable w.r.t. ``theta``.
    """

    def bind_mesh(self, physical_quad_points: np.ndarray) -> None: ...

    def evaluate(self, theta) -> tuple[jnp.ndarray, jnp.ndarray]: ...


# ── shared helpers ───────────────────────────────────────────────────


def _normalize_centers(centers: jnp.ndarray) -> jnp.ndarray:
    lo = centers.min(axis=0)
    hi = centers.max(axis=0)
    return (centers - lo) / (hi - lo + 1e-10)


def _init_theta(
    n_features: int,
    n_out: int,
    hidden_size: int,
    n_layers: int,
    seed: int,
) -> list[dict]:
    layer_sizes = [n_features] + [hidden_size] * (n_layers - 1) + [n_out]
    key = jax.random.PRNGKey(seed)
    theta = init_mlp(key, layer_sizes)
    # Start near the background material (MLP output ≈ 0 at init).
    theta[-1]["W"] = theta[-1]["W"] * 0.01
    theta[-1]["b"] = theta[-1]["b"] * 0.0
    return theta


# ── continuous neural field (QP-level evaluation) ────────────────────


@dataclass
class ContinuousNeuralField:
    """Neural field evaluated at every physical quadrature point.

    At :meth:`bind_mesh`, precomputes the background push-forward
    ``(C_qp_bg, rho_qp_bg)``, the in-cloak mask, and normalised QP
    coordinates + Fourier features. At :meth:`evaluate`, runs the MLP on
    the full QP set and applies a multiplicative residual masked to the
    cloak region.
    """

    geometry: CloakGeometry3D
    C0: jnp.ndarray
    rho0: float
    n_C_params: int = 2
    n_fourier: int = 32
    output_scale: float = 0.1
    symmetrize_init: bool = True

    # set by bind_mesh
    _C_bg: jnp.ndarray = field(init=False, default=None)
    _rho_bg: jnp.ndarray = field(init=False, default=None)
    _features: jnp.ndarray = field(init=False, default=None)
    _mask: jnp.ndarray = field(init=False, default=None)
    _shape: tuple = field(init=False, default=None)

    def bind_mesh(self, physical_quad_points: np.ndarray) -> None:
        pts = jnp.asarray(physical_quad_points)          # (n_fem, n_qp, 3)
        n_fem, n_qp, _ = pts.shape
        self._shape = (n_fem, n_qp)

        flat = pts.reshape(-1, 3)

        geo = self.geometry
        C0 = self.C0
        rho0 = self.rho0
        sym = self.symmetrize_init

        def _C_pt(x):
            return C_eff_3d(x, geo, C0, symmetrize=sym)

        def _rho_pt(x):
            return rho_eff_3d(x, geo, rho0)

        C_bg_flat = jax.vmap(_C_pt)(flat)                # (N, 3,3,3,3)
        rho_bg_flat = jax.vmap(_rho_pt)(flat)            # (N,)
        mask_flat = jax.vmap(geo.in_cloak)(flat)         # (N,) bool

        self._C_bg = C_bg_flat.reshape(n_fem, n_qp, 3, 3, 3, 3)
        self._rho_bg = rho_bg_flat.reshape(n_fem, n_qp)
        self._mask = mask_flat.reshape(n_fem, n_qp)

        bbox = geo.bbox()
        lo = jnp.array([bbox[0], bbox[2], bbox[4]])
        hi = jnp.array([bbox[1], bbox[3], bbox[5]])
        norm = (flat - lo) / (hi - lo + 1e-10)
        self._features = fourier_features_3d(norm, self.n_fourier)

    # ── setup helpers ────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return 6 * self.n_fourier

    def init_theta(
        self,
        hidden_size: int,
        n_layers: int,
        seed: int,
    ) -> list[dict]:
        n_out = self.n_C_params + 1
        return _init_theta(self.n_features, n_out, hidden_size, n_layers, seed)

    # ── evaluation ───────────────────────────────────────────────────

    def evaluate(self, theta) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, from_flat = _get_converters(self.n_C_params)
        n_fem, n_qp = self._shape

        raw = mlp_forward(theta, self._features)         # (N, n_C+1)
        rel_C_flat = raw[:, : self.n_C_params] * self.output_scale
        rel_rho = raw[:, self.n_C_params] * self.output_scale

        mask_flat = self._mask.reshape(-1).astype(raw.dtype)
        rel_C_flat = rel_C_flat * mask_flat[:, None]
        rel_rho = rel_rho * mask_flat

        # Apply residual directly on the flat Voigt representation so we
        # stay in the isotropic (or other low-dim) subspace throughout.
        to_flat, _ = _get_converters(self.n_C_params)
        # Project per-QP background onto the flat parameterisation.
        C_bg_flat_vec = jax.vmap(to_flat)(
            self._C_bg.reshape(-1, 3, 3, 3, 3)
        )                                                 # (N, n_C)
        new_C_flat_vec = C_bg_flat_vec * (1.0 + rel_C_flat)
        C_qp_flat = jax.vmap(from_flat)(new_C_flat_vec)   # (N, 3,3,3,3)
        rho_qp_flat = self._rho_bg.reshape(-1) * (1.0 + rel_rho)

        C_qp = C_qp_flat.reshape(n_fem, n_qp, 3, 3, 3, 3)
        rho_qp = rho_qp_flat.reshape(n_fem, n_qp)
        return C_qp, rho_qp


# ── cell decomposition (3D) + cell-decomposed neural field ───────────


@dataclass
class CellDecomposition3D:
    """Regular 3D grid of cubic cells covering the cloak bounding box."""

    geometry: CloakGeometry3D
    n_x: int
    n_y: int
    n_z: int

    x_min: float = field(init=False)
    x_max: float = field(init=False)
    y_min: float = field(init=False)
    y_max: float = field(init=False)
    z_min: float = field(init=False)
    z_max: float = field(init=False)
    cell_dx: float = field(init=False)
    cell_dy: float = field(init=False)
    cell_dz: float = field(init=False)
    cell_centers: np.ndarray = field(init=False)   # (n_cells, 3)
    cloak_mask: np.ndarray = field(init=False)     # (n_cells,) bool

    def __post_init__(self) -> None:
        bb = self.geometry.bbox()
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = bb
        self.cell_dx = (self.x_max - self.x_min) / self.n_x
        self.cell_dy = (self.y_max - self.y_min) / self.n_y
        self.cell_dz = (self.z_max - self.z_min) / self.n_z

        cx = self.x_min + (np.arange(self.n_x) + 0.5) * self.cell_dx
        cy = self.y_min + (np.arange(self.n_y) + 0.5) * self.cell_dy
        cz = self.z_min + (np.arange(self.n_z) + 0.5) * self.cell_dz
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")   # (nx, ny, nz)
        self.cell_centers = np.stack(
            [gx.ravel(), gy.ravel(), gz.ravel()], axis=-1,
        )

        mask = np.array([
            bool(self.geometry.in_cloak(jnp.array(c)))
            for c in self.cell_centers
        ])
        self.cloak_mask = mask

    @property
    def n_cells(self) -> int:
        return self.n_x * self.n_y * self.n_z

    @property
    def n_cloak_cells(self) -> int:
        return int(self.cloak_mask.sum())

    def _flat_index(self, ix: int, iy: int, iz: int) -> int:
        return (ix * self.n_y + iy) * self.n_z + iz

    def build_qp_mapping(self, physical_quad_points: np.ndarray) -> np.ndarray:
        """Return ``(n_fem_cells, n_qp)`` int array mapping QPs to cells.

        QPs outside the cloak bbox get the sentinel ``n_cells`` (background).
        """
        pts = np.asarray(physical_quad_points)
        n_fem, n_qp, _ = pts.shape
        flat = pts.reshape(-1, 3)

        ix = np.floor((flat[:, 0] - self.x_min) / self.cell_dx).astype(int)
        iy = np.floor((flat[:, 1] - self.y_min) / self.cell_dy).astype(int)
        iz = np.floor((flat[:, 2] - self.z_min) / self.cell_dz).astype(int)
        idx = (ix * self.n_y + iy) * self.n_z + iz

        outside = (
            (ix < 0) | (ix >= self.n_x)
            | (iy < 0) | (iy >= self.n_y)
            | (iz < 0) | (iz >= self.n_z)
        )
        idx[outside] = self.n_cells
        return idx.reshape(n_fem, n_qp)

    @staticmethod
    def expand_to_quadpoints(
        cell_values: jnp.ndarray,
        qp_to_cell: jnp.ndarray,
        background_value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Differentiable expansion ``(n_cells, ...)`` → ``(n_fem, n_qp, ...)``."""
        bg = jnp.expand_dims(background_value, axis=0)
        extended = jnp.concatenate([cell_values, bg], axis=0)
        return extended[qp_to_cell]


@dataclass
class CellDecomposedNeuralField:
    """MLP evaluated at cell centres; QPs inherit via precomputed indices."""

    cell_decomp: CellDecomposition3D
    C0: jnp.ndarray
    rho0: float
    n_C_params: int = 2
    n_fourier: int = 32
    output_scale: float = 0.1
    symmetrize_init: bool = True

    # set in __post_init__
    _cell_features: jnp.ndarray = field(init=False, default=None)
    _cell_C_flat_init: jnp.ndarray = field(init=False, default=None)  # (n_cells, n_C)
    _cell_rho_init: jnp.ndarray = field(init=False, default=None)     # (n_cells,)
    _cloak_mask_j: jnp.ndarray = field(init=False, default=None)
    _C0_flat: jnp.ndarray = field(init=False, default=None)

    # set by bind_mesh
    _qp_to_cell: jnp.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        cd = self.cell_decomp
        centers = jnp.array(cd.cell_centers)
        norm = _normalize_centers(centers)
        self._cell_features = fourier_features_3d(norm, self.n_fourier)

        to_flat, _ = _get_converters(self.n_C_params)

        # Initial per-cell material from the continuous push-forward,
        # evaluated at each cell centre.
        def _C_pt(x):
            return C_eff_3d(x, cd.geometry, self.C0, symmetrize=self.symmetrize_init)

        def _rho_pt(x):
            return rho_eff_3d(x, cd.geometry, self.rho0)

        C_cell = jax.vmap(_C_pt)(centers)                   # (n_cells, 3,3,3,3)
        rho_cell = jax.vmap(_rho_pt)(centers)               # (n_cells,)
        self._cell_C_flat_init = jax.vmap(to_flat)(C_cell)  # (n_cells, n_C)
        self._cell_rho_init = rho_cell
        self._cloak_mask_j = jnp.array(cd.cloak_mask)
        self._C0_flat = to_flat(self.C0)

    @property
    def n_features(self) -> int:
        return 6 * self.n_fourier

    def init_theta(
        self,
        hidden_size: int,
        n_layers: int,
        seed: int,
    ) -> list[dict]:
        n_out = self.n_C_params + 1
        return _init_theta(self.n_features, n_out, hidden_size, n_layers, seed)

    def bind_mesh(self, physical_quad_points: np.ndarray) -> None:
        self._qp_to_cell = jnp.array(
            self.cell_decomp.build_qp_mapping(np.asarray(physical_quad_points))
        )

    def evaluate(self, theta) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, from_flat = _get_converters(self.n_C_params)

        raw = mlp_forward(theta, self._cell_features)                 # (n_cells, n_C+1)
        rel_C = raw[:, : self.n_C_params] * self.output_scale
        rel_rho = raw[:, self.n_C_params] * self.output_scale

        mask = self._cloak_mask_j.astype(raw.dtype)
        cell_C_flat = self._cell_C_flat_init * (1.0 + rel_C * mask[:, None])
        cell_rho = self._cell_rho_init * (1.0 + rel_rho * mask)

        cell_C_full = jax.vmap(from_flat)(cell_C_flat)                # (n_cells, 3,3,3,3)

        # Background tensor for QPs outside the cloak bbox sentinel.
        bg_C = from_flat(self._C0_flat)
        C_qp = CellDecomposition3D.expand_to_quadpoints(
            cell_C_full, self._qp_to_cell, bg_C,
        )
        rho_qp = CellDecomposition3D.expand_to_quadpoints(
            cell_rho, self._qp_to_cell, jnp.array(self.rho0),
        )
        return C_qp, rho_qp
