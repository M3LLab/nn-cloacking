"""FEM problem definition for 3D frequency-domain elastodynamics.

DOF ordering per node: ``[Re(ux), Re(uy), Re(uz), Im(ux), Im(uy), Im(uz)]``
(``vec = 6``, ``dim = 3``).

The :class:`RayleighCloakProblem3D` class hooks into JAX-FEM via:
  - ``custom_init``      : precompute C_eff, rho_eff, xi at quadrature points
  - ``get_tensor_map``   : stiffness + stiffness-proportional damping coupling
  - ``get_mass_map``     : inertia  + mass-proportional damping coupling
  - ``get_surface_maps`` : Gaussian point-like vertical traction on top surface

Material updates during optimisation go through :meth:`set_params`, which
takes a pre-computed ``(C_qp, rho_qp)`` pair — the ``MaterialField``
abstraction evaluates the neural network (with or without a cell grid) and
hands the result to the problem. This keeps the problem agnostic to the
choice of material parameterisation.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem

from rayleigh_cloak_3d.absorbing import make_xi_profile
from rayleigh_cloak_3d.config import DerivedParams3D, SimulationConfig3D
from rayleigh_cloak_3d.geometry.base import CloakGeometry3D
from rayleigh_cloak_3d.materials import C_eff_3d, C_iso_3d, rho_eff_3d


class RayleighCloakProblem3D(Problem):
    """3D frequency-domain elastodynamics with Rayleigh-damping PMLs.

    Additional attributes set on the per-instance subclass (see
    :func:`build_problem`): ``_omega, _geometry, _is_reference, _C0, _rho0,
    _xi_fn, _x_src, _y_src, _sigma_src, _F0, _z_top``.
    """

    def custom_init(self):
        geo = self._geometry
        C0 = self._C0
        rho0 = self._rho0
        is_ref = self._is_reference
        xi_fn = type(self).__dict__['_xi_fn']

        if is_ref:
            def _C_eff_pt(x):
                return C0

            def _rho_eff_pt(x):
                return rho0
        else:
            def _C_eff_pt(x):
                return C_eff_3d(x, geo, C0)

            def _rho_eff_pt(x):
                return rho_eff_3d(x, geo, rho0)

        xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)
        self._xi_qp = xi_qp

        self.internal_vars = [
            jax.vmap(jax.vmap(_C_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_eff_pt))(self.physical_quad_points),
            xi_qp,
        ]

    def set_params(self, params):
        """Install material at quadrature points.

        Parameters
        ----------
        params : tuple ``(C_qp, rho_qp)``
            ``C_qp``  shape ``(n_cells, n_qp, 3, 3, 3, 3)``
            ``rho_qp`` shape ``(n_cells, n_qp)``
        """
        C_qp, rho_qp = params
        self.internal_vars = [C_qp, rho_qp, self._xi_qp]

    def get_tensor_map(self):
        def stress(u_grad, C_q, _rho_q, xi_q):
            # u_grad: (6, 3)
            grad_R = u_grad[:3, :]
            grad_I = u_grad[3:, :]

            sig_R_undamped = jnp.einsum("ijkl,kl->ij", C_q, grad_R)
            sig_I_undamped = jnp.einsum("ijkl,kl->ij", C_q, grad_I)

            sig_R = sig_R_undamped - xi_q * sig_I_undamped
            sig_I = sig_I_undamped + xi_q * sig_R_undamped

            return jnp.concatenate([sig_R, sig_I], axis=0)

        return stress

    def get_mass_map(self):
        omega = self._omega

        def inertia(u, _x, _C_q, rho_q, xi_q):
            u_R, u_I = u[:3], u[3:]
            m_R = -omega ** 2 * rho_q * (u_R + xi_q * u_I)
            m_I = -omega ** 2 * rho_q * (u_I - xi_q * u_R)
            return jnp.concatenate([m_R, m_I])

        return inertia

    def get_surface_maps(self):
        x_src = self._x_src
        y_src = self._y_src
        sigma_src = self._sigma_src
        F0 = self._F0

        def traction(_u, x):
            r2 = (x[0] - x_src) ** 2 + (x[1] - y_src) ** 2
            g = F0 * jnp.exp(-0.5 * r2 / sigma_src ** 2)
            # Vertical traction on top surface (Re part on uz dof).
            return jnp.array([0.0, 0.0, g, 0.0, 0.0, 0.0])

        return [traction]


# ── boundary conditions ──────────────────────────────────────────────


def _make_dirichlet_bc(params: DerivedParams3D):
    """Zero-displacement BCs on the five PML outer faces.

    Top face (``z = z_top``) is free.
    """
    W_total = params.W_total
    H_total = params.H_total

    def bc_xmin(point): return jnp.isclose(point[0], 0.0)
    def bc_xmax(point): return jnp.isclose(point[0], W_total)
    def bc_ymin(point): return jnp.isclose(point[1], 0.0)
    def bc_ymax(point): return jnp.isclose(point[1], W_total)
    def bc_zmin(point): return jnp.isclose(point[2], 0.0)

    def zero(_point): return 0.0

    loc_fns, dof_ids, val_fns = [], [], []
    for loc in (bc_xmin, bc_xmax, bc_ymin, bc_ymax, bc_zmin):
        for d in range(6):
            loc_fns.append(loc)
            dof_ids.append(d)
            val_fns.append(zero)

    return [loc_fns, dof_ids, val_fns]


def _make_top_surface(params: DerivedParams3D) -> Callable:
    H_total = params.H_total

    def top_surface(point):
        return jnp.isclose(point[2], H_total)

    return top_surface


# ── factory ──────────────────────────────────────────────────────────


def build_problem(
    mesh: Mesh,
    cfg: SimulationConfig3D,
    params: DerivedParams3D,
    geometry: CloakGeometry3D,
) -> RayleighCloakProblem3D:
    """Assemble a :class:`RayleighCloakProblem3D` ready for solving.

    The material field (continuous vs cell-decomposed) is *not* attached
    here — it is layered on top by the optimisation driver, which calls
    :meth:`set_params` with ``(C_qp, rho_qp)`` derived from the neural net.
    """
    C0 = C_iso_3d(params.lam, params.mu)

    ProblemCls = type("RayleighCloakProblem3DInstance", (RayleighCloakProblem3D,), {
        "_omega":        params.omega,
        "_geometry":     geometry,
        "_is_reference": cfg.is_reference,
        "_C0":           C0,
        "_rho0":         params.rho0,
        "_xi_fn":        make_xi_profile(params),
        "_x_src":        params.x_src,
        "_y_src":        params.y_src,
        "_sigma_src":    params.sigma_src,
        "_F0":           params.F0,
        "_z_top":        params.z_top,
        "_lam_param":    params.lam,
        "_mu_param":     params.mu,
    })

    return ProblemCls(
        mesh=mesh,
        vec=6,
        dim=3,
        ele_type=cfg.mesh.ele_type,
        dirichlet_bc_info=_make_dirichlet_bc(params),
        location_fns=[_make_top_surface(params)],
    )
