"""Material tensor computations for 3D transformational elasticity.

Pure / JAX-traceable. Voigt convention (3D):
    0 = (0,0),  1 = (1,1),  2 = (2,2),
    3 = (1,2) = (2,1),  4 = (0,2) = (2,0),  5 = (0,1) = (1,0).
"""

from __future__ import annotations

import jax.numpy as jnp

from rayleigh_cloak_3d.geometry.base import CloakGeometry3D


# ── isotropic stiffness ──────────────────────────────────────────────


def C_iso_3d(lam: float, mu: float) -> jnp.ndarray:
    """Isotropic 3-D stiffness tensor C_{ijkl}."""
    I = jnp.eye(3)
    return (
        lam * jnp.einsum("ij,kl->ijkl", I, I)
        + mu * (jnp.einsum("ik,jl->ijkl", I, I)
                + jnp.einsum("il,jk->ijkl", I, I))
    )


# ── Voigt (6×6) conversions (classical minor-symmetric) ────────────


_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def C_to_voigt6(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (3,3,3,3) tensor (assumed minor-symmetric) to a 6×6 Voigt matrix."""
    M = jnp.zeros((6, 6))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            M = M.at[I, J].set(C[i, j, k, l])
    return M


def voigt6_to_C(M: jnp.ndarray) -> jnp.ndarray:
    """Convert 6×6 Voigt matrix to (3,3,3,3) tensor, enforcing minor symmetry."""
    C = jnp.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            v = M[I, J]
            # fill all minor-symmetric pairs (ij)(kl) == (ji)(kl) == (ij)(lk) == (ji)(lk)
            for (a, b) in ((i, j), (j, i)):
                for (c_, d_) in ((k, l), (l, k)):
                    C = C.at[a, b, c_, d_].set(v)
    return C


# ── 2-param isotropic (λ, μ) flat conversion ────────────────────────


def C_to_flat2(C: jnp.ndarray) -> jnp.ndarray:
    """Extract (λ, μ) from an isotropic (3,3,3,3) tensor."""
    mu = C[0, 1, 0, 1]
    lam = C[0, 0, 1, 1]
    return jnp.array([lam, mu])


def flat2_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    return C_iso_3d(flat[0], flat[1])


# ── converter dispatcher ────────────────────────────────────────────


def _get_converters(n_C_params: int):
    """Return ``(to_flat, from_flat)`` for the chosen material parameterisation.

    Currently only ``n_C_params = 2`` (isotropic) is implemented.
    Cubic (3), orthotropic (9), and full (21) can be added by supplying
    analogous flat↔tensor pairs; the rest of the pipeline is agnostic.
    """
    if n_C_params == 2:
        return C_to_flat2, flat2_to_C
    raise NotImplementedError(
        f"n_C_params={n_C_params} not yet implemented in rayleigh_cloak_3d. "
        "Supported: 2 (isotropic). "
    )


# ── symmetrisation ──────────────────────────────────────────────────


def symmetrize_stiffness_3d(C: jnp.ndarray) -> jnp.ndarray:
    """Project onto the minor-symmetric subspace (classical elasticity).

    The transformational push-forward generally yields a tensor that is
    *not* minor-symmetric (that's the Willis / Cosserat asymmetry). For an
    isotropic-parametrised cloak we collapse back to the minor-symmetric
    approximation via Voigt averaging.
    """
    M = C_to_voigt6(0.25 * (C + jnp.transpose(C, (1, 0, 2, 3))
                              + jnp.transpose(C, (0, 1, 3, 2))
                              + jnp.transpose(C, (1, 0, 3, 2))))
    M = 0.5 * (M + M.T)  # major symmetry as well
    return voigt6_to_C(M)


# ── position-dependent effective properties (push-forward) ──────────


def C_eff_3d(
    x: jnp.ndarray,
    geometry: CloakGeometry3D,
    C0: jnp.ndarray,
    symmetrize: bool = False,
) -> jnp.ndarray:
    """Transformational-elasticity push-forward C^eff_{ijkl}.

    Uses the minor-type pushforward (Norris 2008 / Chatzopoulos 2023
    convention), matching the 2D implementation in ``rayleigh_cloak.materials``::

        C^eff_{ijkl} = (1/J) F_{jJ} F_{lL} C0_{iJkL}

    Outside the cloak annulus, ``F = I`` and this reduces to ``C0``.
    """
    F = geometry.F_tensor(x)
    J = jnp.linalg.det(F)
    Cnew = jnp.einsum("jJ,lL,iJkL->ijkl", F, F, C0) / J
    if symmetrize:
        Cnew = symmetrize_stiffness_3d(Cnew)
    return jnp.where(geometry.in_cloak(x), Cnew, C0)


def rho_eff_3d(
    x: jnp.ndarray,
    geometry: CloakGeometry3D,
    rho0: float,
) -> jnp.ndarray:
    F = geometry.F_tensor(x)
    J = jnp.linalg.det(F)
    rho_cloak = rho0 / J
    return jnp.where(geometry.in_cloak(x), rho_cloak, rho0)
