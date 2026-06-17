"""FEM problem definition for frequency-domain elastodynamics.

DOF ordering per node: [Re(ux), Re(uy), Im(ux), Im(uy)]  (vec = 4).

The ``RayleighCloakProblem`` class hooks into JAX-FEM via:
  - ``custom_init``   : precompute C_eff, rho_eff, xi at quadrature points
  - ``get_tensor_map`` : stiffness + stiffness-proportional damping coupling
  - ``get_mass_map``   : inertia  + mass-proportional damping coupling
  - ``get_surface_maps`` : Rayleigh-wave point source (real part only)
"""

from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem

from rayleigh_cloak.absorbing import make_xi_profile
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry
from rayleigh_cloak.materials import C_eff, C_iso, rho_eff, _get_converters


class RayleighCloakProblem(Problem):
    """Frequency-domain elastodynamics with Rayleigh-damping absorbing layers.

    Additional attributes set before ``__init__``:
        _omega, _geometry, _C0, _rho0, _xi_fn, _x_src, _sigma_src, _F0
    """

    def custom_init(self):
        geo = self._geometry
        C0 = self._C0
        rho0 = self._rho0
        is_ref = self._is_reference
        symmetrize = getattr(type(self), '_symmetrize_cloak', False)
        xi_fn = type(self).__dict__['_xi_fn']

        if is_ref:
            def _C_eff_pt(x):
                return C0

            def _rho_eff_pt(x):
                return rho0
        else:
            def _C_eff_pt(x):
                return C_eff(x, geo, C0, symmetrize=symmetrize)

            def _rho_eff_pt(x):
                return rho_eff(x, geo, rho0)

        xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)
        self._xi_qp = xi_qp  # stored separately for set_params

        # Precompute cell mapping if cell decomposition is available
        cell_decomp = getattr(type(self), '_cell_decomp', None)
        if cell_decomp is not None:
            import numpy as np
            confine = getattr(type(self), '_confine_to_cloak', False)
            self._qp_to_cell = jnp.array(
                cell_decomp.build_qp_mapping(
                    np.asarray(self.physical_quad_points),
                    confine_to_cloak=confine,
                )
            )

        self.internal_vars = [
            jax.vmap(jax.vmap(_C_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_eff_pt))(self.physical_quad_points),
            xi_qp,
        ]

    def set_params(self, params):
        """Map cell-based material params to internal_vars at quadrature points.

        Parameters
        ----------
        params : (cell_C_flat, cell_rho) where
            cell_C_flat : (n_cells, n_C_params)
            cell_rho    : (n_cells,)
        """
        cell_C_flat, cell_rho = params
        cell_decomp = type(self)._cell_decomp

        # Check if Nassar cell material is being used
        nassar_mat = getattr(type(self), '_nassar_cell_material', None)
        if nassar_mat is not None:
            # Nassar parameterization: cell_C_flat is (n_cells, 5)
            cell_C_full, cell_rho = nassar_mat.params_to_C_full(params)
        else:
            _, from_flat = _get_converters(
                type(self)._n_C_params,
                getattr(type(self), '_aniso_cauchy', False),
            )
            cell_C_full = jax.vmap(from_flat)(cell_C_flat)  # (n_cells, 2,2,2,2)

        # Expand to quadrature points via precomputed mapping
        C_qp = cell_decomp.expand_to_quadpoints(
            cell_C_full, self._qp_to_cell, self._C0)
        rho_qp = cell_decomp.expand_to_quadpoints(
            cell_rho, self._qp_to_cell, self._rho0)

        self.internal_vars = [C_qp, rho_qp, self._xi_qp]

    def get_tensor_map(self):
        def stress(u_grad, C_q, _rho_q, xi_q):
            grad_R = u_grad[:2, :]
            grad_I = u_grad[2:, :]

            sig_R_undamped = jnp.einsum("ijkl,kl->ij", C_q, grad_R)
            sig_I_undamped = jnp.einsum("ijkl,kl->ij", C_q, grad_I)

            sig_R = sig_R_undamped - xi_q * sig_I_undamped
            sig_I = sig_I_undamped + xi_q * sig_R_undamped

            return jnp.concatenate([sig_R, sig_I], axis=0)

        return stress

    def get_mass_map(self):
        omega = self._omega
        source_type = getattr(type(self), '_source_type', 'gaussian')

        if source_type == 'plane_wave':
            # Body force source: Gaussian line source at x = x_src
            x_src = self._x_src
            sigma_src = self._sigma_src
            F0 = self._F0
            wave_type = getattr(type(self), '_wave_type', 'P')

            def inertia(u, x, _C_q, rho_q, xi_q):
                u_R, u_I = u[:2], u[2:]
                m_R = -omega ** 2 * rho_q * (u_R + xi_q * u_I)
                m_I = -omega ** 2 * rho_q * (u_I - xi_q * u_R)
                # Body force: Gaussian line source (real part only)
                f_amp = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
                if wave_type == 'P':
                    f_body = jnp.array([f_amp, 0.0])
                else:
                    f_body = jnp.array([0.0, f_amp])
                m_R = m_R + f_body
                return jnp.concatenate([m_R, m_I])

            return inertia
        else:
            def inertia(u, _x, _C_q, rho_q, xi_q):
                u_R, u_I = u[:2], u[2:]
                m_R = -omega ** 2 * rho_q * (u_R + xi_q * u_I)
                m_I = -omega ** 2 * rho_q * (u_I - xi_q * u_R)
                return jnp.concatenate([m_R, m_I])

            return inertia

    def get_surface_maps(self):
        source_type = getattr(type(self), '_source_type', 'gaussian')
        F0 = self._F0

        if source_type == 'plane_wave':
            # Plane wave uses body force in get_mass_map, not surface traction.
            # Return a zero-traction surface map (top boundary, harmless).
            def zero_traction(_u, x):
                return jnp.array([0.0, 0.0, 0.0, 0.0])
            return [zero_traction]
        else:
            # Gaussian surface source (original Rayleigh wave setup)
            x_src = self._x_src
            sigma_src = self._sigma_src

            def traction(_u, x):
                g = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
                return jnp.array([0.0, g, 0.0, 0.0])

            return [traction]


# ── boundary conditions ──────────────────────────────────────────────


def _make_dirichlet_bc(params: DerivedParams):
    """Build the ``dirichlet_bc_info`` list for zero-displacement PML edges."""
    W_total = params.W_total
    H_total = params.H_total
    pml_all = getattr(params, 'pml_all_sides', False)

    def bc_bottom(point):
        return jnp.isclose(point[1], 0.0)

    def bc_left(point):
        return jnp.isclose(point[0], 0.0)

    def bc_right(point):
        return jnp.isclose(point[0], W_total)

    def bc_top(point):
        return jnp.isclose(point[1], H_total)

    def zero(point):
        return 0.0

    loc_fns = [bc_bottom, bc_bottom, bc_bottom, bc_bottom,
               bc_left,   bc_left,   bc_left,   bc_left,
               bc_right,  bc_right,  bc_right,  bc_right]
    dof_ids = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    val_fns = [zero] * 12

    if pml_all:
        loc_fns.extend([bc_top, bc_top, bc_top, bc_top])
        dof_ids.extend([0, 1, 2, 3])
        val_fns.extend([zero, zero, zero, zero])

    return [loc_fns, dof_ids, val_fns]


def _make_top_surface(params: DerivedParams) -> Callable:
    H_total = params.H_total

    def top_surface(point):
        return jnp.isclose(point[1], H_total)

    return top_surface


def _make_left_physical_boundary(params: DerivedParams) -> Callable:
    """Left physical boundary (x = x_off) for plane-wave source."""
    x_off = params.x_off

    def left_phys(point):
        return jnp.isclose(point[0], x_off)

    return left_phys


# ── factory ──────────────────────────────────────────────────────────


def build_problem(
    mesh: Mesh,
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
    cell_decomp: CellDecomposition | None = None,
    n_C_params_override: int | None = None,
) -> RayleighCloakProblem:
    """Assemble a ``RayleighCloakProblem`` ready for solving.

    Parameters
    ----------
    cell_decomp : optional
        If provided, enables ``set_params`` for cell-based optimisation.
    n_C_params_override : optional
        Override ``cfg.cells.n_C_params`` (used by topology optimisation
        which always uses 2-param isotropic representation).
    """
    C0 = C_iso(params.lam, params.mu)

    # Per-instance subclass so state never leaks between problems (otherwise
    # later build_problem calls would overwrite earlier problems' _omega etc.
    # via the shared base class, silently mixing frequencies at solve time).
    ProblemCls = type("RayleighCloakProblemInstance", (RayleighCloakProblem,), {
        "_omega":        params.omega,
        "_geometry":     geometry,
        "_is_reference": cfg.is_reference,
        "_symmetrize_cloak": cfg.symmetrize_cloak,
        "_C0":           C0,
        "_rho0":         params.rho0,
        "_xi_fn":        make_xi_profile(params),
        "_x_src":        params.x_src,
        "_sigma_src":    params.sigma_src,
        "_F0":           params.F0,
        "_cell_decomp":  cell_decomp,
        "_confine_to_cloak": cfg.cells.confine_to_cloak,
        "_n_C_params":   n_C_params_override or cfg.cells.n_C_params,
        "_aniso_cauchy": cfg.cells.aniso_cauchy,
        "_source_type":  cfg.source.source_type,
        "_wave_type":    cfg.source.wave_type,
        "_lam_param":    params.lam,
        "_mu_param":     params.mu,
    })

    return ProblemCls(
        mesh=mesh,
        vec=4,
        dim=2,
        ele_type=cfg.mesh.ele_type,
        dirichlet_bc_info=_make_dirichlet_bc(params),
        location_fns=[_make_top_surface(params)],
    )
