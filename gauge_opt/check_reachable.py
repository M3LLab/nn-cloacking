"""Stage 2: is the gauge-optimized target inside what D2 blocks actually build?

Scores each gauge in the family against the fitted reachable-set model
(``dataset/gmm/gmm_flat4_squared_2m.npz``) and reports how much of that set
satisfies the rigid ratios forced by the Cauchy gauge (theory.md §4.3).

Run:
    python -m gauge_opt.check_reachable
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gauge_opt import gauge as G
from gauge_opt import objectives as O
from gauge_opt.reachable import ReachableGMM, fit_D2, load_gmm
from gauge_opt.sweep_gauge import triangular_F


def sample_gmm(gmm: ReachableGMM, n: int, seed: int = 0) -> np.ndarray:
    """Draw ``n`` samples in physical units, columns (C11, C22, C66, C12, rho)."""
    rng = np.random.default_rng(seed)
    k = rng.choice(len(gmm.weights), size=n, p=gmm.weights / gmm.weights.sum())
    S = np.zeros((n, gmm.means.shape[1]))
    for i in range(len(gmm.weights)):
        idx = np.where(k == i)[0]
        if len(idx):
            S[idx] = rng.multivariate_normal(gmm.means[i], gmm.covs[i], size=len(idx))
    return S * gmm.fstd + gmm.fmean


def main() -> None:
    rho0, cs = 1600.0, 300.0
    mu = rho0 * cs**2
    lam = rho0 * (np.sqrt(3.0) * cs) ** 2 - 2 * mu

    H = 4.305
    a, c_geo = 0.0774 * H, 0.1545 * H
    b = 3.0 * a
    F = triangular_F(a, b, c_geo)
    C0 = np.asarray(G.C0_iso(lam, mu))

    gmm = load_gmm()
    print(f"reachable-set model: {len(gmm.weights)} components, "
          f"log-density threshold {gmm.threshold:.4f}")

    # ── how much of the reachable set meets the Cauchy-gauge ratios ──
    Y = sample_gmm(gmm, 600_000)
    C11, C22, C66, C12, rho = Y.T
    keep = (C11 > 0) & (C22 > 0) & (C66 > 0) & (C12 > 0) & (rho > 0)
    C11, C22, C66, C12, rho = (v[keep] for v in (C11, C22, C66, C12, rho))
    hi, lo = np.maximum(C11, C22), np.minimum(C11, C22)

    tgt_aniso = None
    fit1 = fit_D2(np.asarray(G.C_gauge(F, F.T, C0)))
    m = fit1.moduli                                     # (C11, C12, C22, C66)
    tgt_aniso = max(m[0], m[2]) / min(m[0], m[2])
    r1_t, r2_t = m[1] / m[3], m[0] * m[2] / (m[1] * m[3])

    print(f"\nCauchy-gauge target: anisotropy {tgt_aniso:.2f}, "
          f"C12/C66 {r1_t:.3f}, C11C22/(C12C66) {r2_t:.3f}")
    print("fraction of the reachable set meeting each condition, cumulatively:")

    s1 = (hi / lo > 0.6 * tgt_aniso) & (hi / lo < 1.7 * tgt_aniso)
    s2 = s1 & (np.abs(C12 / C66 - r1_t) / r1_t < 0.25)
    s3 = s2 & (np.abs(C11 * C22 / (C12 * C66) - r2_t) / r2_t < 0.25)
    for label, sel in (("anisotropy", s1), ("+ C12/C66", s2), ("+ C11C22/(C12C66)", s3)):
        print(f"  {label:22s} {sel.mean()*100:8.4f} %   (n = {sel.sum()})")

    speeds = np.sqrt(hi / rho)
    print(f"\nscale check — sqrt(max(C11,C22)/rho) over the reachable set:")
    print(f"  pct[5,50,95] = {np.percentile(speeds, [5, 50, 95]).round(0)} m/s")
    print(f"  substrate    : cs = {cs:.0f} m/s, cp = {np.sqrt(3)*cs:.0f} m/s")
    print("  -> the catalogue was homogenized for a base material roughly")
    print("     5-8x too stiff for this substrate; renormalize before matching.")

    # ── score the gauge family ───────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'s':>5} {'asym':>10} {'log rho ratio':>14} {'log p(target)':>15} {'margin':>10} {'reach':>7}")
    print("-" * 78)
    for s in np.linspace(0.0, 1.0, 11):
        A = G.gauge_power(F, s)
        C = np.asarray(G.C_gauge(F, A, C0))
        rho_t = np.asarray(G.rho_gauge(F, A, rho0))
        fit = fit_D2(C)
        t11, t12, t22, t66 = fit.moduli
        rbar = float(np.mean(np.linalg.eigvalsh(rho_t)))   # scalar proxy; D2 rho is isotropic
        lp = float(gmm.log_prob(t11, t22, t66, t12, rbar))
        print(f"{s:5.2f} {O.asymmetry(C):10.3e} {O.density_anisotropy(rho_t):14.4f} "
              f"{lp:15.3f} {gmm.threshold - lp:10.3f} "
              f"{'y' if lp > gmm.threshold else 'N':>7}")
    print("\nmargin <= 0 means inside the achievable region.")
    print("NOTE: log p is evaluated at the RAW target moduli.  Because the")
    print("catalogue's base material is mismatched (above), these are expected")
    print("to sit outside until stage 0 renormalization is done.")


if __name__ == "__main__":
    main()
