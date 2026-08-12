"""Verification identities for the gauge machinery (theory.md §7).

Plain script, matching the style of the repo's ``tests/`` (no pytest in the
jax-fem-env)::

    python -m gauge_opt.test_gauge
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gauge_opt import gauge as G
from gauge_opt import objectives as O
from gauge_opt.reachable import cauchy_gauge_moduli, fit_D2, voigt4

RHO0, CS = 1600.0, 300.0
MU = RHO0 * CS**2
LAM = RHO0 * (np.sqrt(3.0) * CS) ** 2 - 2 * MU   # = MU, nu = 0.25


def _setup():
    H = 4.305
    a, c = 0.0774 * H, 0.1545 * H
    b = 3.0 * a
    F = np.array([[1.0, 0.0], [a / c, (b - a) / b]])
    return F, np.asarray(G.C0_iso(LAM, MU))


def _random_gauges(n=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        A = rng.normal(size=(2, 2))
        if abs(np.linalg.det(A)) > 0.05:
            out.append(A)
    return out


# ── §7.1 ─────────────────────────────────────────────────────────────
def test_identity_gauge_matches_repo_C_eff():
    """A = I must reproduce materials.py::C_eff exactly."""
    F, C0 = _setup()
    got = np.asarray(G.C_gauge(F, np.eye(2), C0))
    want = np.einsum("jJ,lL,iJkL->ijkl", F, F, C0) / np.linalg.det(F)
    assert np.allclose(got, want, rtol=0, atol=1e-9 * np.linalg.norm(want))


# ── §7.2 ─────────────────────────────────────────────────────────────
def test_major_symmetry_all_gauges():
    """Major symmetry survives every gauge."""
    F, C0 = _setup()
    for A in _random_gauges():
        C = np.asarray(G.C_gauge(F, A, C0))
        assert np.allclose(C, np.einsum("ijkl->klij", C), atol=1e-8 * np.linalg.norm(C))


# ── §7.3 ─────────────────────────────────────────────────────────────
def test_cauchy_gauge_is_minor_symmetric():
    """A = c F^T kills the minor-symmetry violation exactly, for any c."""
    F, C0 = _setup()
    for c in (1.0, 0.37, -2.5):
        C = np.asarray(G.C_gauge(F, c * F.T, C0))
        assert O.asymmetry(C) < 1e-13, (c, O.asymmetry(C))


# ── §7.4 ─────────────────────────────────────────────────────────────
def test_cauchy_gauge_orthotropic_and_closed_form():
    """C16 = C26 = 0 in the eigenframe of B; moduli match eq (4.5)."""
    F, C0 = _setup()
    fit = fit_D2(np.asarray(G.C_gauge(F, F.T, C0)))
    theta_a, mod_a = cauchy_gauge_moduli(F, LAM, MU)
    assert fit.ortho_res < 1e-12, fit.ortho_res
    assert np.allclose(fit.moduli, mod_a, rtol=1e-10)
    assert abs(fit.theta % np.pi - theta_a % np.pi) < 1e-8


# ── §4.3 ─────────────────────────────────────────────────────────────
def test_rigid_modulus_ratios():
    """The two host-determined ratios, independent of c and of F."""
    F, C0 = _setup()
    for c in (1.0, 0.2, 5.0):
        C11, C12, C22, C66 = fit_D2(np.asarray(G.C_gauge(F, c * F.T, C0))).moduli
        assert np.isclose(C12 / C66, LAM / MU, rtol=1e-10)
        assert np.isclose(
            C11 * C22 / (C12 * C66), (LAM + 2 * MU) ** 2 / (LAM * MU), rtol=1e-10
        )


# ── §7.5 ─────────────────────────────────────────────────────────────
def test_density_spd_all_gauges():
    """rho^A is symmetric positive-definite for every invertible A."""
    F, _ = _setup()
    for A in _random_gauges():
        rho = np.asarray(G.rho_gauge(F, A, RHO0))
        assert np.allclose(rho, rho.T)
        assert np.linalg.eigvalsh(rho).min() > 0


# ── §7.6 ─────────────────────────────────────────────────────────────
def test_willis_vanishes_for_constant_gauge():
    """Piecewise-affine map + constant gauge  =>  S = W = 0 exactly."""
    F, C0 = _setup()
    dA = np.zeros((2, 2, 2))
    for A in _random_gauges(4):
        assert np.linalg.norm(np.asarray(G.willis_S(F, A, dA, C0))) == 0.0
        assert np.linalg.norm(np.asarray(G.willis_W(F, dA, C0))) == 0.0


# ── §7.7 ─────────────────────────────────────────────────────────────
def test_inertia_is_a_gauge_invariant():
    """Rank 3 with one zero eigenvalue, for EVERY gauge (Sylvester)."""
    F, C0 = _setup()
    for A in _random_gauges(16, seed=3):
        M = voigt4(np.asarray(G.C_gauge(F, A, C0)))
        w = np.linalg.eigvalsh(0.5 * (M + M.T)) / np.linalg.norm(M)
        assert abs(w[0]) < 1e-12, f"expected a zero eigenvalue, got {w[0]}"
        assert (w[1:] > 1e-6).all(), f"expected three positive, got {w}"


# ── §7.8 ─────────────────────────────────────────────────────────────
def test_symmetrizing_gauge_is_unique():
    """Every unconstrained minimizer of the asymmetry is parallel to F^T."""
    from scipy.optimize import minimize

    F, C0 = _setup()

    def f(p):
        A = p.reshape(2, 2)
        if abs(np.linalg.det(A)) < 1e-6:
            return 1e3
        return O.asymmetry(np.asarray(G.C_gauge(F, A, C0)))

    rng = np.random.default_rng(1)
    for _ in range(5):
        r = minimize(f, rng.normal(size=4), method="Nelder-Mead",
                     options=dict(maxiter=40000, maxfev=40000, xatol=1e-12, fatol=1e-14))
        A = r.x.reshape(2, 2)
        c = np.sum(A * F.T) / np.sum(F.T * F.T)
        assert np.linalg.norm(A - c * F.T) / np.linalg.norm(A) < 1e-9


def test_pd_check_uses_symmetric_strain_block():
    """The 3x3 block stays PD across the family though the 4x4 is singular."""
    F, C0 = _setup()
    for s in np.linspace(0.0, 1.0, 6):
        C = np.asarray(G.C_gauge(F, G.gauge_power(F, s), C0))
        assert O.positive_definite(C)
        assert abs(O.couple_stiffness(C)) < 1e-12


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
