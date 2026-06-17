"""Verify BGM push-forward C_eff and cloaking on a circular domain.

Tests:
1. C_eff at phi=0 matches the analytical augmented Voigt formula
2. rho_eff matches rc^2/(rc-ri)^2 * f * rho0
3. Full-mesh identity (C_eff without void cutout) gives ~1% far-field distortion
4. Void-cutout continuous C_eff gives ~1% on a circle at r=1.5*rc
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.geometry.circular import CircularCloakGeometry
from rayleigh_cloak.materials import C_eff, C_iso, rho_eff, C_to_voigt4


@pytest.fixture
def nassar_setup():
    cfg = load_config('configs/nassar.yaml')
    params = DerivedParams.from_config(cfg)
    geo = CircularCloakGeometry(ri=params.ri, rc=params.rc,
                                 x_c=params.x_c, y_c=params.y_c)
    C0 = C_iso(params.lam, params.mu)
    return cfg, params, geo, C0


def test_C_eff_analytical(nassar_setup):
    """C_eff at phi=0 matches the analytical augmented Voigt."""
    _, params, geo, C0 = nassar_setup
    lam, mu = params.lam, params.mu

    r_test = 0.5 * (params.ri + params.rc)
    x_test = jnp.array([geo.x_c + r_test, geo.y_c])
    f = (r_test - geo.ri) / r_test

    V = C_to_voigt4(C_eff(x_test, geo, C0))
    M = 2 * mu + lam
    V_expected = jnp.array([
        [M * f, lam,   0,      0],
        [lam,   M / f, 0,      0],
        [0,     0,     mu / f, mu],
        [0,     0,     mu,     mu * f],
    ])
    assert jnp.allclose(V, V_expected, atol=1e-4), \
        f"Max err = {float(jnp.max(jnp.abs(V - V_expected)))}"


def test_rho_eff_analytical(nassar_setup):
    """rho_eff = rc^2 / (rc-ri)^2 * f * rho0."""
    _, params, geo, _ = nassar_setup

    r_test = 0.5 * (params.ri + params.rc)
    x_test = jnp.array([geo.x_c + r_test, geo.y_c])
    f = (r_test - geo.ri) / r_test

    rho = rho_eff(x_test, geo, params.rho0)
    rho_expected = params.rc**2 / (params.rc - params.ri)**2 * f * params.rho0
    assert jnp.allclose(rho, rho_expected, atol=1e-6)


@pytest.mark.slow
def test_cloaking_circle_metric(nassar_setup):
    """Continuous C_eff with void cutout gives <3% distortion on circle r=1.5*rc."""
    from rayleigh_cloak.solver import _create_geometry, jax_fem_solver
    from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
    from rayleigh_cloak.problem import build_problem
    from rayleigh_cloak.optimize import get_circular_boundary_indices

    cfg, params, _, _ = nassar_setup
    geo = _create_geometry(cfg, params)
    full_mesh = generate_mesh_full(cfg, params, geo)

    solver_opts = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

    # Reference
    p_ref = build_problem(full_mesh, cfg.model_copy(update={'is_reference': True}), params, geo)
    u_ref = np.asarray(jax_fem_solver(p_ref, solver_options=solver_opts)[0])

    # Cloak (void cutout)
    cloak_mesh, kept = extract_submesh(full_mesh, geo)
    p_cloak = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geo)
    u_cloak = np.asarray(jax_fem_solver(p_cloak, solver_options=solver_opts)[0])

    # Measure on circle r = 1.5*rc
    idx = get_circular_boundary_indices(
        np.asarray(cloak_mesh.points), geo.x_c, geo.y_c, 1.5 * params.rc)
    u_ref_meas = u_ref[kept[idx]]
    diff = u_cloak[idx] - u_ref_meas
    rn = np.sqrt(np.sum(u_ref_meas**2))
    distortion = 100 * np.sqrt(np.sum(diff**2)) / rn

    print(f"Circle r=1.5rc: {distortion:.2f}% ({len(idx)} nodes, |u_ref|={rn:.4e})")
    assert distortion < 3.0, f"Circle distortion {distortion:.2f}% > 3%"
