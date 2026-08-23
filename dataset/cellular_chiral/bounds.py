"""Rigorous attainability bounds for effective (C, rho) of the cement/void cells.

Any two-phase composite of cement and void is constrained, whatever its
microstructure, by bounds that depend only on the solid volume fraction. This
module makes those bounds callable so a *target* (C, rho) can be rejected as
physically impossible before an inverse-design run is spent on it -- independent of
whether the dataset happens to contain a nearby sample.

Setup (from ``dataset/stiffness/calc_fem.py``): **plane strain**, cement
E = 30 GPa, nu = 0.2, rho = 2300 kg/m^3; void is an ersatz phase at E x 1e-6, i.e.
effectively K2 = G2 = 0. Since the dataset satisfies ``rho = rho_solid * vol``
exactly, volume fraction and density are interchangeable -- use ``vol_from_rho``.

Which bound applies to what
---------------------------
* ``hs_upper_2d`` -- Hashin-Shtrikman upper bounds on the in-plane bulk modulus
  K = (C11 + C22 + 2 C12)/4 and shear modulus. The bulk bound limits the
  hydrostatic energy of *any* two-phase composite, anisotropic ones included, so
  it is the sharp test to apply to K. The shear bound is the isotropic-composite
  bound; a strongly anisotropic cell can beat it in one direction, so it is
  reported as a diagnostic rather than enforced.
* ``voigt_upper`` -- rule of mixtures. Weaker than HS but valid componentwise and
  direction by direction for any microstructure, so it is the right ceiling for
  individual C11 / C22 / C66.
* ``positive_definite`` -- C11, C22, C66 > 0 and C11 C22 > C12^2. Not a bound but a
  hard admissibility condition on the tensor itself.

The HS lower bound is identically zero here: a disconnected solid phase carries no
load at any volume fraction, so no useful lower bound exists.
"""
from __future__ import annotations

import numpy as np

E_CEMENT = 30e9     # Pa   (dataset/stiffness/calc_fem.py)
NU = 0.2
RHO_SOLID = 2300.0  # kg/m^3

# plane-strain solid constants
LAMBDA_S = E_CEMENT * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU_S = E_CEMENT / (2.0 * (1.0 + NU))
C11_S = LAMBDA_S + 2.0 * MU_S
C12_S = LAMBDA_S
C66_S = MU_S
K_S = LAMBDA_S + MU_S      # in-plane (2-D) bulk modulus of the solid
G_S = MU_S


def vol_from_rho(rho):
    """Solid volume fraction implied by an effective density."""
    return np.asarray(rho, dtype=np.float64) / RHO_SOLID


def bulk_2d(C11, C22, C12):
    """In-plane bulk modulus of a 2-D orthotropic tensor (hydrostatic response)."""
    return (np.asarray(C11) + np.asarray(C22) + 2.0 * np.asarray(C12)) / 4.0


def hs_upper_2d(vol):
    """Hashin-Shtrikman upper bounds (K, G) for cement + void at volume fraction ``vol``.

    Two-phase 2-D HS with the void phase at K2 = G2 = 0::

        K+ = K1 + f2 / ( 1/(K2-K1) + f1/(K1+G1) )
        G+ = G1 + f2 / ( 1/(G2-G1) + f1 (K1+2G1) / (2 G1 (K1+G1)) )

    Both go to the solid values as ``vol -> 1`` and to zero as ``vol -> 0``.
    """
    f1 = np.clip(np.asarray(vol, dtype=np.float64), 0.0, 1.0)
    f2 = 1.0 - f1
    k_den = -1.0 / K_S + f1 / (K_S + G_S)
    g_den = -1.0 / G_S + f1 * (K_S + 2.0 * G_S) / (2.0 * G_S * (K_S + G_S))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = K_S + f2 / k_den
        g = G_S + f2 / g_den
    return np.where(f2 == 0, K_S, k), np.where(f2 == 0, G_S, g)


def voigt_upper(vol):
    """Rule-of-mixtures ceiling on (C11, C22, C12, C66)."""
    f1 = np.asarray(vol, dtype=np.float64)
    return f1 * C11_S, f1 * C11_S, f1 * C12_S, f1 * C66_S


def positive_definite(C11, C22, C12, C66):
    C11, C22, C12, C66 = map(np.asarray, (C11, C22, C12, C66))
    return (C11 > 0) & (C22 > 0) & (C66 > 0) & (C11 * C22 > C12**2)


def check_attainable(C11, C22, C12, C66, vol=None, rho=None, tol=1e-9):
    """Screen candidate targets. Returns ``(ok_mask, diagnostics)``.

    ``diagnostics`` holds the utilisation of each bound (value / bound); anything
    above 1 is impossible. ``hs_G_util`` is reported but **not** enforced, since
    an anisotropic cell may legitimately exceed the isotropic shear bound.
    """
    if vol is None:
        if rho is None:
            raise ValueError("pass vol or rho")
        vol = vol_from_rho(rho)
    vol = np.asarray(vol, dtype=np.float64)
    C11, C22, C12, C66 = map(lambda a: np.asarray(a, dtype=np.float64),
                             (C11, C22, C12, C66))

    k_hs, g_hs = hs_upper_2d(vol)
    v11, v22, v12, v66 = voigt_upper(vol)
    k_eff = bulk_2d(C11, C22, C12)

    d = {
        "hs_K_util": k_eff / k_hs,
        "hs_G_util": C66 / g_hs,
        "voigt_C11_util": C11 / v11,
        "voigt_C22_util": C22 / v22,
        "voigt_C66_util": C66 / v66,
        "pd_margin": 1.0 - C12**2 / (C11 * C22),
    }
    ok = (
        positive_definite(C11, C22, C12, C66)
        & (d["hs_K_util"] <= 1.0 + tol)
        & (d["voigt_C11_util"] <= 1.0 + tol)
        & (d["voigt_C22_util"] <= 1.0 + tol)
        & (d["voigt_C66_util"] <= 1.0 + tol)
        & (vol > 0) & (vol <= 1)
    )
    return ok, d


if __name__ == "__main__":  # validate the bounds against the real dataset
    import h5py

    print(f"plane-strain solid: C11={C11_S:.4g}  C12={C12_S:.4g}  C66={C66_S:.4g}  "
          f"K={K_S:.4g}  G={G_S:.4g}")
    with h5py.File("output/ca_bulk_squared/stiffness.h5", "r") as f:
        C11, C22, C12, C66, vol = (f[k][:] for k in ("C11", "C22", "C12", "C66", "vol"))
    ok, d = check_attainable(C11, C22, C12, C66, vol=vol)
    print(f"\n{len(ok)} real rows: {np.sum(ok)} satisfy every enforced bound, "
          f"{np.sum(~ok)} violate")
    for k, v in d.items():
        if k == "pd_margin":
            print(f"  {k:16s} min={v.min():.4f}")
        else:
            print(f"  {k:16s} max={v.max():.4f}  p99={np.percentile(v, 99):.4f}  "
                  f"median={np.median(v):.4f}")
