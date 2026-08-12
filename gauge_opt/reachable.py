"""The D2 (orthotropic, centrosymmetric) reachable set and projection onto it.

What a D2-symmetric unit-cell block can actually deliver, and why:

* **Cauchy homogenization** of an ordinary (non-micropolar) cell always yields a
  minor-symmetric stiffness.  A D2 block cannot break minor symmetry — that
  needs a couple-transmitting lattice.
* **Centrosymmetry** of D2 forces every odd-rank effective tensor to vanish, so
  the Willis coupling ``S`` is identically zero.  Not small — exactly zero.
* **D2 point symmetry** (two orthogonal mirrors) kills C16 and C26 in the cell
  frame, leaving the orthotropic quadruple (C11, C12, C22, C66).
* The cell may be **rotated**, contributing one more parameter theta.

So the reachable set is the 5-dimensional family

    { R(theta) . orthotropic(C11, C12, C22, C66) : (C11,C12,C22,C66) in B }

with ``B`` the achievable modulus box.  That is exactly what the microstructure
generator is conditioned on (the four ``microstructure_generation_2d/scaler_*``
files) — see ``load_dataset_box``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── rotation of a 4th-order 2-D tensor ───────────────────────────────

def rotate_C(C: np.ndarray, theta: float) -> np.ndarray:
    """Rotate C_ijkl by angle theta:  C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C)


def voigt4(C: np.ndarray) -> np.ndarray:
    """Augmented 4x4 Voigt matrix, basis order (11, 22, 12, 21).

    Mirrors ``rayleigh_cloak.materials.C_to_voigt4`` so the numbers here are
    directly comparable to the rest of the repo.
    """
    pairs = [(0, 0), (1, 1), (0, 1), (1, 0)]
    return np.array([[C[i, j, k, l] for (k, l) in pairs] for (i, j) in pairs])


def minor_symmetrize(C: np.ndarray) -> np.ndarray:
    """Project onto the minor-symmetric (Cauchy) subspace.

    Same averaging recipe as ``materials.symmetrize_stiffness``:
    Csym_IJ = (C_IJ + C_ĪJ + C_IJ̄ + C_ĪJ̄)/4, bar = swap (0,1)<->(1,0).
    """
    C = 0.25 * (
        C
        + np.einsum("ijkl->jikl", C)
        + np.einsum("ijkl->ijlk", C)
        + np.einsum("ijkl->jilk", C)
    )
    return 0.5 * (C + np.einsum("ijkl->klij", C))


# ── orthotropic frame extraction ─────────────────────────────────────

@dataclass
class D2Fit:
    """Best D2 representation of a stiffness tensor."""

    theta: float          # cell orientation [rad]
    moduli: np.ndarray    # (C11, C12, C22, C66) in the cell frame
    asym_res: float       # relative minor-symmetry violation discarded
    ortho_res: float      # relative C16/C26 residual discarded


def fit_D2(C: np.ndarray, n_scan: int = 721) -> D2Fit:
    """Fit the closest rotated-orthotropic Cauchy tensor to ``C``.

    Two lossy projections happen, and both residuals are reported rather than
    silently dropped:

    1. minor-symmetrization  (a D2 cell cannot break minor symmetry),
    2. rotation to the frame minimizing C16^2 + C26^2  (D2 kills C16, C26).
    """
    nrm = np.linalg.norm(voigt4(C))
    Cs = minor_symmetrize(C)
    asym_res = np.linalg.norm(voigt4(C - Cs)) / nrm

    # Scan then refine: the objective is a trigonometric polynomial of low
    # order in theta, so a coarse scan + golden-section refinement is exact.
    def obj(t: float) -> float:
        Cr = rotate_C(Cs, t)
        return Cr[0, 0, 0, 1] ** 2 + Cr[1, 1, 0, 1] ** 2

    thetas = np.linspace(0.0, np.pi, n_scan)
    vals = np.array([obj(t) for t in thetas])
    i = int(np.argmin(vals))
    h = thetas[1] - thetas[0]
    lo, hi = thetas[i] - h, thetas[i] + h

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = obj(x1), obj(x2)
    for _ in range(80):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = obj(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = obj(x2)
    theta = float(0.5 * (lo + hi))

    Cr = rotate_C(Cs, theta)
    ortho_res = float(np.hypot(Cr[0, 0, 0, 1], Cr[1, 1, 0, 1]) / nrm)
    moduli = np.array([Cr[0, 0, 0, 0], Cr[0, 0, 1, 1], Cr[1, 1, 1, 1], Cr[0, 1, 0, 1]])
    return D2Fit(theta=theta, moduli=moduli, asym_res=float(asym_res), ortho_res=ortho_res)


# ── analytic prediction for the Cauchy gauge (theory.md eq 4.5) ──────

def cauchy_gauge_moduli(F: np.ndarray, lam: float, mu: float, c: float = 1.0):
    """Closed-form (theta, C11, C12, C22, C66) for the A = c F^T gauge.

    Returns the eigenframe of B = F F^T and the moduli of eq (4.5).  Used to
    cross-check ``fit_D2`` — the two must agree to machine precision, with
    ``ortho_res`` and ``asym_res`` both ~0.

    The angle is returned in ``fit_D2``'s convention (rotation applied *to the
    tensor*, i.e. minus the angle of the eigenframe), reduced to [0, pi).
    """
    B = np.asarray(F, dtype=float) @ np.asarray(F, dtype=float).T
    J = float(np.linalg.det(F))
    evals, evecs = np.linalg.eigh(B)
    b1, b2 = float(evals[0]), float(evals[1])
    theta = float(-np.arctan2(evecs[1, 0], evecs[0, 0]) % np.pi)
    k = c**2 / J
    return theta, np.array(
        [k * (lam + 2 * mu) * b1**2, k * lam * b1 * b2, k * (lam + 2 * mu) * b2**2, k * mu * b1 * b2]
    )


# ── achievable modulus box from the dataset ──────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
_SCALER_DIR = _ROOT / "microstructure_generation_2d"
_GMM_PATH = _ROOT / "dataset" / "gmm" / "gmm_flat4_squared_2m.npz"


@dataclass
class ReachableGMM:
    """The fitted density model of what D2 cells actually achieve.

    Wraps ``dataset/gmm/gmm_flat4_squared_2m.npz`` — a 16-component GMM over
    the standardized features ``(C1111, C2222, C1212, C1122, rho)`` fitted to
    ~906k homogenized cells, with ``threshold`` the log-density cut that the
    dataset tooling uses to call a target "achievable".

    This is the real reachable set; prefer it over :func:`load_dataset_box`.
    """

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    fmean: np.ndarray
    fstd: np.ndarray
    threshold: float

    def log_prob(self, C11, C22, C66, C12, rho) -> np.ndarray:
        """Log-density of one or many targets, in the GMM's feature order."""
        x = np.atleast_2d(np.stack(np.broadcast_arrays(C11, C22, C66, C12, rho), -1))
        z = (x - self.fmean) / self.fstd
        out = np.full((z.shape[0], len(self.weights)), -np.inf)
        for i, (w, m, S) in enumerate(zip(self.weights, self.means, self.covs)):
            L = np.linalg.cholesky(S)
            d = np.linalg.solve(L, (z - m).T).T
            out[:, i] = (
                np.log(w)
                - 0.5 * np.sum(d**2, axis=1)
                - np.sum(np.log(np.diag(L)))
                - 0.5 * z.shape[1] * np.log(2 * np.pi)
            )
        mx = out.max(axis=1, keepdims=True)
        return np.squeeze(mx[:, 0] + np.log(np.exp(out - mx).sum(axis=1)))

    def reachable(self, C11, C22, C66, C12, rho) -> np.ndarray:
        """Boolean: is this target inside the achievable region?"""
        return self.log_prob(C11, C22, C66, C12, rho) > self.threshold

    def margin(self, C11, C22, C66, C12, rho) -> np.ndarray:
        """``threshold - log_prob``: <=0 achievable, >0 = how far outside.

        This is the differentiable reachability penalty for eq (6.4) — smooth
        in the moduli, unlike a hard box.
        """
        return self.threshold - self.log_prob(C11, C22, C66, C12, rho)


def load_gmm(path: Path | None = None) -> ReachableGMM:
    """Load the fitted D2 reachable-set model."""
    d = np.load(path or _GMM_PATH, allow_pickle=True)
    order = [str(s) for s in d["feature_order"]]
    assert order == ["C1111", "C2222", "C1212", "C1122", "rho"], order
    return ReachableGMM(
        weights=d["weights"],
        means=d["means"],
        covs=d["covariances"],
        fmean=d["feature_mean"],
        fstd=d["feature_std"],
        threshold=float(d["threshold"]),
    )


def load_dataset_box(n_sigma: float = 1.5) -> dict[str, tuple[float, float]]:
    """Approximate achievable (C11, C12, C22, C66) box from the fitted scalers.

    The ``scaler_*`` files are sklearn ``StandardScaler``s carrying the training
    mean and std of each modulus, so ``mean +/- n_sigma * scale`` is a usable
    first-pass bounding box.

    CRUDE — kept only as a fast sanity bound.  It both over-claims (includes
    unrealizable corners of the box) and under-claims (clips the tails where
    the strongly anisotropic cells live).  Use :func:`load_gmm` instead for any
    number you intend to believe.
    """
    import joblib

    box = {}
    for key in ("C11", "C12", "C22", "C66"):
        sc = joblib.load(_SCALER_DIR / f"scaler_{key}")
        m, s = float(np.ravel(sc.mean_)[0]), float(np.ravel(sc.scale_)[0])
        box[key] = (max(0.0, m - n_sigma * s), m + n_sigma * s)
    return box


def box_distance(moduli: np.ndarray, box: dict[str, tuple[float, float]]) -> float:
    """Relative L2 distance from (C11, C12, C22, C66) to the achievable box.

    Zero inside the box.  Normalized per-coordinate by the box width so the
    four moduli, which differ by an order of magnitude, contribute comparably.
    """
    keys = ("C11", "C12", "C22", "C66")
    d = 0.0
    for v, k in zip(moduli, keys):
        lo, hi = box[k]
        w = max(hi - lo, 1e-30)
        d += (max(lo - v, 0.0, v - hi) / w) ** 2
    return float(np.sqrt(d))
