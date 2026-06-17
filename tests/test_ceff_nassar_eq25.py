"""Verify that C_eff from the code matches eq. 2.5 of Nassar et al. (2018).

Eq. 2.5 gives the augmented Voigt matrix of the effective stiffness in the
polar basis (m, n) for a circular cloak:

    [(2μ+λ)f    λ        0      0   ]
    [  λ      (2μ+λ)/f   0      0   ]
    [  0        0        μ/f    μ    ]
    [  0        0         μ     μf   ]

where f = (‖x‖ − rᵢ) / ‖x‖  and components are in the normalised polar
basis (m, n) with m = x/‖x‖.

The code computes C_eff in Cartesian coordinates via the general push-forward
(eq. 2.3).  This test rotates the paper's polar result to Cartesian and
compares against the code.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from rayleigh_cloak.geometry.circular import CircularCloakGeometry
from rayleigh_cloak.materials import C_iso, C_eff, C_to_voigt4, voigt4_to_C


def nassar_eq25_polar_voigt(lam, mu, f):
    """Augmented 4×4 Voigt matrix from eq. 2.5, in polar basis (m, n)."""
    return jnp.array([
        [(2*mu + lam)*f,  lam,            0.0,    0.0   ],
        [lam,             (2*mu + lam)/f,  0.0,    0.0   ],
        [0.0,             0.0,             mu/f,   mu    ],
        [0.0,             0.0,             mu,     mu*f  ],
    ])


def rotate_C_tensor(C, Q):
    """Rotate a (2,2,2,2) stiffness tensor: C'_ijkl = Q_ia Q_jb Q_kc Q_ld C_abcd."""
    return jnp.einsum("ia,jb,kc,ld,abcd->ijkl", Q, Q, Q, Q, C)


def test_ceff_matches_eq25():
    """C_eff from code matches Nassar eq. 2.5 at several points in the cloak."""
    # Material parameters
    lam = 2.0
    mu = 1.0

    # Geometry
    ri = 0.3
    rc = 1.0
    geom = CircularCloakGeometry(ri=ri, rc=rc, x_c=0.0, y_c=0.0)
    C0 = C_iso(lam, mu)

    # Test at several angles and radii inside the cloak
    # Avoid r too close to ri where numerical clamping (r_safe, R_safe) causes error
    angles = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2.3, np.pi, 5.0]
    radii = [ri + 0.05, 0.5 * (ri + rc), rc - 0.01, 0.5, 0.7]

    for theta in angles:
        for r in radii:
            x = jnp.array([r * np.cos(theta), r * np.sin(theta)])

            # --- Code result (Cartesian) ---
            C_code = C_eff(x, geom, C0)

            # --- Paper eq. 2.5 (polar → Cartesian) ---
            f = (r - ri) / r

            # Polar Voigt → tensor
            M_polar = nassar_eq25_polar_voigt(lam, mu, f)
            C_polar = voigt4_to_C(M_polar)

            # Rotation matrix: polar (m, n) → Cartesian (x, y)
            # m = [cos θ, sin θ],  n = [-sin θ, cos θ]
            # Q maps polar basis vectors to Cartesian: Q = [[cos θ, -sin θ], [sin θ, cos θ]]
            c, s = np.cos(theta), np.sin(theta)
            Q = jnp.array([[c, -s],
                           [s,  c]])

            C_expected = rotate_C_tensor(C_polar, Q)

            # Compare augmented Voigt for readable output on failure
            M_code = C_to_voigt4(C_code)
            M_expected = C_to_voigt4(C_expected)

            npt.assert_allclose(
                np.array(M_code), np.array(M_expected), atol=1e-10,
                err_msg=f"Mismatch at θ={theta:.2f}, r={r:.3f} (f={f:.4f})",
            )


def test_ceff_properties():
    """Sanity checks on C_eff properties implied by eq. 2.5."""
    lam = 2.0
    mu = 1.0
    ri = 0.3
    rc = 1.0
    geom = CircularCloakGeometry(ri=ri, rc=rc, x_c=0.0, y_c=0.0)
    C0 = C_iso(lam, mu)

    # At θ=0, polar = Cartesian, so we can directly check the Voigt entries
    r = 0.6
    x = jnp.array([r, 0.0])
    C_code = C_eff(x, geom, C0)
    M = C_to_voigt4(C_code)
    f = (r - ri) / r

    # At θ=0, polar=Cartesian so Voigt should match eq. 2.5 directly
    npt.assert_allclose(float(M[0, 0]), (2*mu + lam) * f, atol=1e-12)
    npt.assert_allclose(float(M[1, 1]), (2*mu + lam) / f, atol=1e-12)
    npt.assert_allclose(float(M[0, 1]), lam, atol=1e-12)
    npt.assert_allclose(float(M[1, 0]), lam, atol=1e-12)
    npt.assert_allclose(float(M[2, 2]), mu / f, atol=1e-12)
    npt.assert_allclose(float(M[3, 3]), mu * f, atol=1e-12)
    npt.assert_allclose(float(M[2, 3]), mu, atol=1e-12)
    npt.assert_allclose(float(M[3, 2]), mu, atol=1e-12)

    # Major symmetry: C_ijkl = C_klij  (Voigt: M = M^T)
    # Note: eq 2.5 does NOT have M = M^T (rank-3), but major symmetry
    # holds at the tensor level
    npt.assert_allclose(np.array(C_code), np.array(jnp.einsum("ijkl->klij", C_code)),
                        atol=1e-12, err_msg="Major symmetry violated")

    # Rank-3: the paper states that for μ ≤ λ the effective stiffness is rank 3
    # (degenerate lattice), meaning det(M) = 0
    rank = int(jnp.linalg.matrix_rank(M))
    assert rank == 3, f"Expected rank 3 for μ≤λ, got {rank}"


if __name__ == "__main__":
    test_ceff_matches_eq25()
    print("PASS: C_eff matches Nassar eq. 2.5 at all test points")
    test_ceff_properties()
    print("PASS: C_eff properties (major symmetry, rank) verified")
