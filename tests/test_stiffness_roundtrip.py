"""Test stiffness tensor round-trip through flat parameterisations.

Diagnoses the NaN issue in the optimisation solver by checking that the
flat representations reconstruct valid stiffness tensors.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from rayleigh_cloak.materials import (
    C_iso,
    C_eff,
    C_to_flatC,
    C_to_flat6,
    C_to_voigt4,
    flatC_to_C,
    flat6_to_C,
    voigt4_to_C,
    _get_converters,
)
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry


def _make_geometry():
    """Minimal geometry matching the optimize.yaml config."""
    return TriangularCloakGeometry(a=0.333, b=1.0, c=0.665, x_c=7.25, y_top=5.305)


def _make_C0():
    rho0, cs = 1600.0, 300.0
    mu = rho0 * cs ** 2
    lam = rho0 * (cs * np.sqrt(3)) ** 2 - 2 * mu
    return C_iso(lam, mu), lam, mu


def _cloak_point(geo):
    return jnp.array([geo.x_c + 0.1, geo.y_top - 0.5 * (geo.a + geo.b)])


# ── n_C_params=4 (known limitation) ──────────────────────────────────


class TestFlatC4:
    """Tests for the n_C_params=4 representation (minor-symmetric, rank-3)."""

    def test_isotropic_roundtrip_exact(self):
        """C0 → flat4 → C should be exact for isotropic material."""
        C0, _, _ = _make_C0()
        C_rt = flatC_to_C(C_to_flatC(C0))
        np.testing.assert_allclose(C_rt, C0, atol=1e-6)

    def test_voigt4_rank_3(self):
        """4-param always produces rank-3 Voigt (known limitation)."""
        C0, _, _ = _make_C0()
        C_rt = flatC_to_C(C_to_flatC(C0))
        V = C_to_voigt4(C_rt)
        rank = np.linalg.matrix_rank(np.array(V), tol=1e-6)
        assert rank == 3, f"Expected rank 3 (singular shear block), got {rank}"

    def test_warns_on_use(self):
        with pytest.warns(UserWarning, match="rank-3"):
            _get_converters(4)


# ── n_C_params=6 (block-diagonal Cosserat) ───────────────────────────


class TestFlat6:
    """Tests for the n_C_params=6 representation."""

    def test_isotropic_roundtrip_exact(self):
        C0, _, _ = _make_C0()
        C_rt = flat6_to_C(C_to_flat6(C0))
        np.testing.assert_allclose(C_rt, C0, atol=1e-6)

    def test_cosserat_roundtrip_block_diagonal(self):
        """6-param captures the block-diagonal part of the augmented Voigt."""
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        x = _cloak_point(geo)
        C = C_eff(x, geo, C0)
        C_rt = flat6_to_C(C_to_flat6(C))

        # Block-diagonal Voigt components should match exactly
        V_orig = C_to_voigt4(C)
        V_rt = C_to_voigt4(C_rt)
        # Normal block (top-left 2×2)
        np.testing.assert_allclose(V_rt[:2, :2], V_orig[:2, :2], atol=1e-3)
        # Shear block (bottom-right 2×2)
        np.testing.assert_allclose(V_rt[2:, 2:], V_orig[2:, 2:], atol=1e-3)

    def test_voigt4_rank_isotropic(self):
        """Isotropic C0 has minor symmetry → rank 3 is expected.
        The mass term omega^2*M regularises this in the FEM system.
        """
        C0, _, _ = _make_C0()
        C_rt = flat6_to_C(C_to_flat6(C0))
        V = C_to_voigt4(C_rt)
        rank = np.linalg.matrix_rank(np.array(V), tol=1e-6)
        assert rank == 3, f"Isotropic Voigt4 expected rank 3, got {rank}"

    def test_voigt4_full_rank_cloak(self):
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        C = C_eff(_cloak_point(geo), geo, C0)
        C_rt = flat6_to_C(C_to_flat6(C))
        V = C_to_voigt4(C_rt)
        rank = np.linalg.matrix_rank(np.array(V), tol=1e-6)
        assert rank == 4, (
            f"Cloak Voigt4 rank={rank}, expected 4.\n"
            f"Voigt4:\n{np.array(V)}\n"
            f"eigenvalues: {np.linalg.eigvalsh(np.array(V))}"
        )

    def test_no_nan(self):
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        for dx in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            x = jnp.array([geo.x_c + dx, geo.y_top - 0.5 * (geo.a + geo.b)])
            C = C_eff(x, geo, C0)
            flat = C_to_flat6(C)
            C_rt = flat6_to_C(flat)
            assert not jnp.any(jnp.isnan(C_rt)), f"NaN at dx={dx}"

    def test_major_symmetry_preserved(self):
        """Reconstructed tensor must have C[i,j,k,l] = C[k,l,i,j]."""
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        C = C_eff(_cloak_point(geo), geo, C0)
        C_rt = flat6_to_C(C_to_flat6(C))
        C_T = jnp.einsum("ijkl->klij", C_rt)
        np.testing.assert_allclose(C_rt, C_T, atol=1e-6,
                                   err_msg="Major symmetry violated")

    def test_all_stress_components_nonzero(self):
        """For a general strain, all 4 stress components must be nonzero."""
        C0, _, _ = _make_C0()
        C_rt = flat6_to_C(C_to_flat6(C0))
        grad = jnp.array([[1.0, 0.3], [0.2, 0.5]])
        sig = jnp.einsum("ijkl,kl->ij", C_rt, grad)
        for i in range(2):
            for j in range(2):
                assert sig[i, j] != 0.0, f"sig[{i},{j}]=0"

    def test_cosserat_asymmetry_preserved(self):
        """6-param must preserve C[0,1,0,1] != C[1,0,1,0] from push-forward."""
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        C = C_eff(_cloak_point(geo), geo, C0)
        C_rt = flat6_to_C(C_to_flat6(C))
        assert not jnp.allclose(C_rt[0, 1, 0, 1], C_rt[1, 0, 1, 0]), \
            "Minor symmetry not broken — Cosserat asymmetry lost"

    def test_get_converters_6(self):
        to_flat, from_flat = _get_converters(6)
        C0, _, _ = _make_C0()
        C_rt = from_flat(to_flat(C0))
        np.testing.assert_allclose(C_rt, C0, atol=1e-6)


# ── n_C_params=16 (full) ─────────────────────────────────────────────


class TestVoigt16:
    def test_isotropic_roundtrip_exact(self):
        C0, _, _ = _make_C0()
        to_flat, from_flat = _get_converters(16)
        C_rt = from_flat(to_flat(C0))
        np.testing.assert_allclose(C_rt, C0, atol=1e-6)

    def test_cosserat_roundtrip_exact(self):
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        to_flat, from_flat = _get_converters(16)
        C = C_eff(_cloak_point(geo), geo, C0)
        C_rt = from_flat(to_flat(C))
        np.testing.assert_allclose(C_rt, C, atol=1e-6)

    def test_voigt4_full_rank(self):
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        to_flat, from_flat = _get_converters(16)
        C = C_eff(_cloak_point(geo), geo, C0)
        C_rt = from_flat(to_flat(C))
        V = C_to_voigt4(C_rt)
        assert np.linalg.matrix_rank(np.array(V), tol=1e-6) == 4


# ── C_eff sanity checks ──────────────────────────────────────────────


class TestCEffProperties:
    def test_major_symmetry(self):
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        C = C_eff(_cloak_point(geo), geo, C0)
        C_T = jnp.einsum("ijkl->klij", C)
        np.testing.assert_allclose(C, C_T, atol=1e-6)

    def test_minor_symmetry_broken(self):
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        C = C_eff(_cloak_point(geo), geo, C0)
        diff = jnp.max(jnp.abs(C[0, 1, :, :] - C[1, 0, :, :]))
        assert diff > 1e-6, "Minor symmetry not broken"

    def test_positive_definite_voigt(self):
        """Symmetric part of the augmented Voigt should be positive semi-definite."""
        C0, _, _ = _make_C0()
        geo = _make_geometry()
        C = C_eff(_cloak_point(geo), geo, C0)
        V = np.array(C_to_voigt4(C), dtype=np.float64)
        eigs = np.linalg.eigvalsh(0.5 * (V + V.T))
        # Allow small negative eigenvalues from float32 precision
        assert np.all(eigs > -1.0), f"Significantly negative eigenvalue: {eigs}"
