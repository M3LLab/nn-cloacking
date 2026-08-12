"""Stage 1 of the pipeline: sweep the gauge family and print the trade-off.

Answers the question "is there a gauge giving minor-symmetric tensors?" with
numbers, on the actual triangular-cloak deformation gradient.

Run:
    python -m gauge_opt.sweep_gauge
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gauge_opt import gauge as G
from gauge_opt import objectives as O
from gauge_opt.reachable import cauchy_gauge_moduli, fit_D2, load_dataset_box


def triangular_F(a: float, b: float, c: float, sign: float = 1.0) -> np.ndarray:
    """Deformation gradient of the triangular cloak (geometry/triangular.py:53).

    Piecewise constant — this is what makes the Willis terms vanish identically
    in the cloak interior (theory.md §4.2).
    """
    return np.array([[1.0, 0.0], [sign * a / c, (b - a) / b]])


def main() -> None:
    # ── host + geometry, matching configs/default.yaml ────────────────
    rho0, cs = 1600.0, 300.0
    cp = np.sqrt(3.0) * cs
    mu = rho0 * cs**2
    lam = rho0 * cp**2 - 2 * mu           # = mu here, i.e. nu = 0.25

    H = 4.305 * 1.0                        # H_factor * lambda_star
    a, c_geo = 0.0774 * H, 0.1545 * H
    b = 3.0 * a

    F = triangular_F(a, b, c_geo)
    C0 = np.asarray(G.C0_iso(lam, mu))
    J = np.linalg.det(F)

    print(f"host:  lam = {lam:.4g}  mu = {mu:.4g}  rho0 = {rho0:g}  (nu = 0.25)")
    print(f"F   = [[{F[0,0]:.4f}, {F[0,1]:.4f}], [{F[1,0]:.4f}, {F[1,1]:.4f}]]   J = {J:.4f}")

    box = load_dataset_box()
    print("\ndataset box (mean +/- 1.5 sigma, PLACEHOLDER — see reachable.py):")
    for k, (lo, hi) in box.items():
        print(f"  {k}: [{lo:.4g}, {hi:.4g}]")

    # ── sweep s from identity gauge to Cauchy gauge ──────────────────
    print("\n" + "=" * 96)
    print("gauge family  A(s) = (F^T)^s     s=0 -> identity/Cosserat,  s=1 -> Cauchy")
    print("=" * 96)
    hdr = (
        f"{'s':>5} {'asym(6.1)':>11} {'ortho res':>11} {'log rho ratio':>14} "
        f"{'theta[deg]':>11} {'box dist':>9} {'couple':>10} {'PD':>3}"
    )
    print(hdr)
    print("-" * 96)

    for s in np.linspace(0.0, 1.0, 11):
        A = G.gauge_power(F, s)
        C = np.asarray(G.C_gauge(F, A, C0))
        rho = np.asarray(G.rho_gauge(F, A, rho0))
        fit = fit_D2(C)
        print(
            f"{s:5.2f} {O.asymmetry(C):11.3e} {fit.ortho_res:11.3e} "
            f"{O.density_anisotropy(rho):14.4f} {np.degrees(fit.theta):11.2f} "
            f"{O.d2_defect(C, box)[0] - fit.asym_res - fit.ortho_res:9.3f} "
            f"{O.couple_stiffness(C):10.3e} "
            f"{'y' if O.positive_definite(C) else 'N':>3}"
        )

    # ── endpoints in detail ──────────────────────────────────────────
    for name, A in (("identity  A = I", np.eye(2)), ("Cauchy    A = F^T", F.T)):
        C = np.asarray(G.C_gauge(F, A, C0))
        rho = np.asarray(G.rho_gauge(F, A, rho0))
        fit = fit_D2(C)
        print(f"\n--- {name} " + "-" * (72 - len(name)))
        print(f"  minor-symmetry violation : {O.asymmetry(C):.3e}")
        print(f"  couple stiffness (rel)   : {O.couple_stiffness(C):.3e}   stable(3x3 PD): {O.positive_definite(C)}")
        print(f"  orthotropy residual      : {fit.ortho_res:.3e}   at theta = {np.degrees(fit.theta):.3f} deg")
        print(f"  density eigenvalues      : {np.linalg.eigvalsh(rho)}  (log ratio {O.density_anisotropy(rho):.4f})")
        print(f"  moduli (C11,C12,C22,C66) : {fit.moduli}")
        print(f"  ratio C12/C66            : {fit.moduli[1]/fit.moduli[3]:.6f}   (host lam/mu = {lam/mu:.6f})")
        print(f"  box distance             : {O.d2_defect(C, box)[0] - fit.asym_res - fit.ortho_res:.4f}")

    # ── cross-check the closed form, theory.md eq (4.5) ──────────────
    theta_a, mod_a = cauchy_gauge_moduli(F, lam, mu)
    fit = fit_D2(np.asarray(G.C_gauge(F, F.T, C0)))
    print("\n--- closed form eq (4.5) vs numeric fit " + "-" * 39)
    print(f"  analytic moduli : {mod_a}")
    print(f"  numeric  moduli : {fit.moduli}")
    print(f"  max rel. error  : {np.max(np.abs(mod_a - fit.moduli) / np.abs(mod_a)):.3e}")
    print(f"  analytic theta  : {np.degrees(theta_a) % 180:.4f} deg   numeric {np.degrees(fit.theta) % 180:.4f} deg")

    # ── Willis check on the piecewise-affine map ─────────────────────
    dA = np.zeros((2, 2, 2))               # grad A = 0 for constant gauge, §4.2
    S = np.asarray(G.willis_S(F, F.T, dA, C0))
    print(f"\n  Willis |S| with constant gauge on piecewise-affine map : {np.linalg.norm(S):.3e}")


if __name__ == "__main__":
    main()
