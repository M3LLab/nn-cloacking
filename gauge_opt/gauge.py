"""Gauge-transformed transformation-elasticity material tensors.

Implements the formulas in ``theory.md``.  Everything here is pure and
JAX-traceable except the matrix power in :func:`gauge_power`, which uses
``scipy.linalg`` (see note there).

Index convention (theory.md §1): capital = virtual domain, lowercase =
physical domain.  The gauge is the matrix field ``A`` in ``u_I = A_Ik ũ_k``.

Key entry points
----------------
``C_gauge(F, A, C0)``      stiffness  C^A_ijkl                  (theory §3, eq 3.2)
``rho_gauge(F, A, rho0)``  density    rho^A_ik  (tensorial)     (eq 3.2)
``willis_S(F, A, dA, C0)`` Willis coupling S_ijk                (eq 3.2)
``gauge_power(F, s, c)``   the interpolating family A(s,c)      (eq 5.1)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# ── host tensor ──────────────────────────────────────────────────────

def C0_iso(lam: float, mu: float) -> jnp.ndarray:
    """Isotropic 2-D host stiffness C^0_IJKL (plane strain).

    Identical to ``rayleigh_cloak.materials.C_iso``; duplicated here so this
    package has no import-time dependency on the FEM stack.
    """
    I = jnp.eye(2)
    return (
        lam * jnp.einsum("ij,kl->ijkl", I, I)
        + mu * (jnp.einsum("ik,jl->ijkl", I, I) + jnp.einsum("il,jk->ijkl", I, I))
    )


# ── gauge-transformed material (theory.md §3) ────────────────────────

def Ccal(F: jnp.ndarray, C0: jnp.ndarray) -> jnp.ndarray:
    """Mixed-index tensor C_IjKl = J^-1 F_jJ F_lL C0_IJKL   (eq 2.1).

    This is the gauge-independent core; every gauge just sandwiches it.
    """
    J = jnp.linalg.det(F)
    return jnp.einsum("jJ,lL,IJKL->IjKl", F, F, C0) / J


def C_gauge(F: jnp.ndarray, A: jnp.ndarray, C0: jnp.ndarray) -> jnp.ndarray:
    """Gauge-transformed stiffness  C^A_ijkl = A_Ii C_IjKl A_Kk   (eq 3.2).

    ``A = eye(2)`` reproduces ``rayleigh_cloak.materials.C_eff`` exactly.
    ``A = F.T`` gives a minor-symmetric (Cauchy) tensor — theory.md §4.
    """
    return jnp.einsum("Ii,IjKl,Kk->ijkl", A, Ccal(F, C0), A)


def rho_gauge(F: jnp.ndarray, A: jnp.ndarray, rho0: float) -> jnp.ndarray:
    """Gauge-transformed density  rho^A_ik = J^-1 rho0 (A^T A)_ik   (eq 3.2).

    Always symmetric positive-definite.  Scalar (isotropic) only when A is a
    multiple of a rotation, i.e. only for the identity gauge and its rescalings.
    """
    J = jnp.linalg.det(F)
    return rho0 * (A.T @ A) / J


def willis_S(
    F: jnp.ndarray, A: jnp.ndarray, dA: jnp.ndarray, C0: jnp.ndarray
) -> jnp.ndarray:
    """Willis-type coupling  S_ijk = A_Ii C_IjKl dA_Kk,l   (eq 3.2).

    Parameters
    ----------
    dA : (2, 2, 2) array, ``dA[K, k, l] = d A_Kk / d x_l``.
         Zero wherever the gauge is locally constant — which, on the
         piecewise-affine triangular map, is the whole cloak interior.
    """
    return jnp.einsum("Ii,IjKl,Kkl->ijk", A, Ccal(F, C0), dA)


def willis_W(F: jnp.ndarray, dA: jnp.ndarray, C0: jnp.ndarray) -> jnp.ndarray:
    """Zeroth-order gauge correction  W_ik = dA_Ii,j C_IjKl dA_Kk,l   (eq 3.2)."""
    return jnp.einsum("Iij,IjKl,Kkl->ik", dA, Ccal(F, C0), dA)


# ── gauge families (theory.md §5) ────────────────────────────────────

def gauge_identity(F: jnp.ndarray) -> jnp.ndarray:
    """A = I — the Cosserat / Brun-Guenneau-Movchan realization."""
    return jnp.eye(F.shape[0])


def gauge_cauchy(F: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """A = c F^T — the minor-symmetric (Cauchy) realization, theory.md §4."""
    return c * F.T


def gauge_power(F: np.ndarray, s: float, c: float = 1.0) -> np.ndarray:
    """A(s, c) = c (F^T)^s — the interpolating family, eq (5.1).

    s = 0 -> identity gauge (Cosserat);  s = 1 -> Cauchy gauge.
    Stays in GL+(2) for all s since det A = c^2 J^s > 0.

    NOTE: uses ``scipy.linalg.expm/logm``, so this is *not* JAX-traceable.
    For the gradient-based stage, substitute the closed-form 2x2 matrix power
    (eigen-decomposition of F is analytic here: spec(F) = {1, (b-a)/b}).
    """
    from scipy.linalg import expm, logm

    Ft = np.asarray(F, dtype=float).T
    return c * np.real(expm(s * logm(Ft)))
