"""Scalar penalties on a gauge choice (theory.md §6).

The scattering objective is deliberately absent: it is *exactly* invariant along
the gauge orbit in the continuum limit.  Everything here measures how expensive
a given gauge is to build out of D2 blocks.
"""

from __future__ import annotations

import numpy as np

from gauge_opt.reachable import D2Fit, box_distance, fit_D2, minor_symmetrize, voigt4


def asymmetry(C: np.ndarray) -> float:
    """Relative minor-symmetry violation, eq (6.1).

    This is the 'chirality' cost: how much couple-stress / micropolar behaviour
    the cell would have to supply.  Exactly 0 for the A = c F^T gauge.
    """
    nrm = np.linalg.norm(voigt4(C))
    return float(np.linalg.norm(voigt4(C - minor_symmetrize(C))) / nrm)


def density_anisotropy(rho: np.ndarray) -> float:
    """log of the density eigenvalue ratio, eq (6.2).  0 = isotropic."""
    w = np.linalg.eigvalsh(rho)
    return float(np.log(w.max() / w.min()))


def willis_magnitude(S: np.ndarray, C: np.ndarray, l_ref: float) -> float:
    """Non-dimensional Willis coupling, eq (6.3).

    ``l_ref`` should be one unit-cell size: the comparison that matters is
    ``S`` against ``C / l_ref``, since S multiplies u while C multiplies grad u.
    """
    return float(l_ref * np.linalg.norm(S) / np.linalg.norm(voigt4(C)))


def d2_defect(C: np.ndarray, box: dict | None = None) -> tuple[float, D2Fit]:
    """Distance to the D2 reachable set, eq (6.4).

    Returns ``(total, fit)`` where ``total`` sums the orthotropy residual, the
    minor-symmetry residual and (if ``box`` given) the out-of-box distance.
    """
    fit = fit_D2(C)
    total = fit.asym_res + fit.ortho_res
    if box is not None:
        total += box_distance(fit.moduli, box)
    return float(total), fit


def positive_definite(C: np.ndarray) -> bool:
    """Hard stability constraint: PD on the *symmetric-strain* subspace.

    Tested on the 3x3 Voigt block (11, 22, sym-12), NOT on the augmented 4x4.
    The 4x4 is singular by construction for any minor-symmetric tensor — see
    :func:`couple_stiffness`.
    """
    return bool(np.linalg.eigvalsh(voigt3_sym(C)).min() > 0)


def voigt3_sym(C: np.ndarray) -> np.ndarray:
    """Symmetric-strain 3x3 Voigt block, basis (11, 22, sqrt2 * sym-12)."""
    r2 = np.sqrt(2.0)
    return np.array(
        [
            [C[0, 0, 0, 0], C[0, 0, 1, 1], r2 * 0.5 * (C[0, 0, 0, 1] + C[0, 0, 1, 0])],
            [C[1, 1, 0, 0], C[1, 1, 1, 1], r2 * 0.5 * (C[1, 1, 0, 1] + C[1, 1, 1, 0])],
            [
                r2 * 0.5 * (C[0, 1, 0, 0] + C[1, 0, 0, 0]),
                r2 * 0.5 * (C[0, 1, 1, 1] + C[1, 0, 1, 1]),
                0.5 * (C[0, 1, 0, 1] + C[0, 1, 1, 0] + C[1, 0, 0, 1] + C[1, 0, 1, 0]),
            ],
        ]
    )


def couple_stiffness(C: np.ndarray) -> float:
    """Smallest eigenvalue of the augmented 4x4, relative to ``||C||``.

    This is the energy cost of a pure local rotation — the couple-stress
    stiffness.  It is strictly positive for the Cosserat (identity) gauge and
    goes to **exactly zero** at the Cauchy gauge, because a Cauchy material
    does not resist local rotation.  That is physics, not a numerical defect.

    Pipeline consequence: the repo's full-gradient (Cosserat) FEM formulation
    inverts the augmented 4x4 and therefore goes singular as s -> 1.  Runs near
    the Cauchy end must use the standard symmetric-strain formulation instead
    (``n_C_params=6cauchy`` rather than 10/16).
    """
    M = voigt4(C)
    return float(np.linalg.eigvalsh(0.5 * (M + M.T)).min() / np.linalg.norm(M))
