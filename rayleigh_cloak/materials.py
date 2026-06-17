"""Material tensor computations for transformational elasticity.

All functions are pure (no global state) and JAX-traceable where needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.geometry.base import CloakGeometry


# ── isotropic stiffness ──────────────────────────────────────────────

def C_iso(lam: float, mu: float) -> jnp.ndarray:
    """Isotropic 2-D stiffness tensor C_{ijkl} (plane-strain)."""
    I = jnp.eye(2)
    return (
        lam * jnp.einsum("ij,kl->ijkl", I, I)
        + mu * (jnp.einsum("ik,jl->ijkl", I, I)
                + jnp.einsum("il,jk->ijkl", I, I))
    )

# ── 2-param isotropic (λ, μ) ────────────────────────────────────────

def C_to_flat2(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) isotropic tensor to 2-param flat array [λ, μ]."""
    mu = C[0, 1, 0, 1]
    lam = C[0, 0, 1, 1]
    return jnp.array([lam, mu])


def flat2_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert 2-param flat array [λ, μ] to (2,2,2,2) isotropic tensor."""
    return C_iso(flat[0], flat[1])


# ── Voigt (3×3) conversions ────────────────────────────────
def C_to_voigt3(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to 3×3 Voigt matrix."""
    M = jnp.zeros((3, 3))
    M = M.at[0, 0].set(C[0, 0, 0, 0])
    M = M.at[1, 1].set(C[1, 1, 1, 1])
    M = M.at[2, 2].set(C[0, 1, 0, 1])
    M = M.at[0, 1].set(C[0, 0, 1, 1])
    M = M.at[1, 0].set(C[1, 1, 0, 0])
    return M

def voigt3_to_C(M: jnp.ndarray) -> jnp.ndarray:
    """Convert 3×3 Voigt matrix to (2,2,2,2) tensor.

    Sets all four shear components to M[2,2] (minor symmetry).
    """
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(M[0, 0])
    C = C.at[1, 1, 1, 1].set(M[1, 1])
    C = C.at[0, 1, 0, 1].set(M[2, 2])
    C = C.at[1, 0, 1, 0].set(M[2, 2])
    C = C.at[0, 1, 1, 0].set(M[2, 2])
    C = C.at[1, 0, 0, 1].set(M[2, 2])
    C = C.at[0, 0, 1, 1].set(M[0, 1])
    C = C.at[1, 1, 0, 0].set(M[1, 0])
    return C

def voight3_to_flatC(M: jnp.ndarray) -> jnp.ndarray:
    """Convert 3×3 Voigt matrix to flattened tensor (3,3) → (4,)."""
    return jnp.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1]])

def flatC_to_voigt3(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert flattened tensor (4,) to 3×3 Voigt matrix."""
    M = jnp.zeros((3, 3))
    M = M.at[0, 0].set(flat[0])
    M = M.at[1, 1].set(flat[1])
    M = M.at[2, 2].set(flat[2])
    M = M.at[0, 1].set(flat[3])
    M = M.at[1, 0].set(flat[3])
    return M

def C_to_flatC(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to flattened (4,) array.

    .. warning:: The 4-param representation enforces minor symmetry, producing
       a rank-3 augmented Voigt matrix.  This is **incompatible** with the
       full-gradient (Cosserat) FEM formulation.  Use ``n_C_params=6`` or
       ``n_C_params=16`` instead.
    """
    flat = jnp.zeros(4)
    flat = flat.at[0].set(C[0, 0, 0, 0])
    flat = flat.at[1].set(C[1, 1, 1, 1])
    flat = flat.at[2].set(C[0, 1, 0, 1])
    flat = flat.at[3].set(C[0, 0, 1, 1])
    return flat

def flatC_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert flattened (4,) array to (2,2,2,2) tensor.

    Enforces minor symmetry on the shear block: all four components
    C[0,1,0,1], C[1,0,1,0], C[0,1,1,0], C[1,0,0,1] are set to flat[2].

    .. warning:: Produces a rank-3 augmented Voigt matrix — singular for the
       full-gradient FEM formulation.
    """
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(flat[0])
    C = C.at[1, 1, 1, 1].set(flat[1])
    C = C.at[0, 1, 0, 1].set(flat[2])
    C = C.at[1, 0, 1, 0].set(flat[2])
    C = C.at[0, 1, 1, 0].set(flat[2])
    C = C.at[1, 0, 0, 1].set(flat[2])
    C = C.at[0, 0, 1, 1].set(flat[3])
    C = C.at[1, 1, 0, 0].set(flat[3])
    return C


# ── 6-param (block-diagonal Cosserat) ────────────────────────────────

def C_to_flat6(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to 6-param flat array.

    Captures the block-diagonal structure of the augmented 4×4 Voigt matrix,
    preserving the Cosserat asymmetry in the shear block::

        flat[0] = C[0,0,0,0]   (normal-xx)
        flat[1] = C[1,1,1,1]   (normal-yy)
        flat[2] = C[0,0,1,1]   (normal coupling, = C[1,1,0,0] by major sym.)
        flat[3] = C[0,1,0,1]   (shear diagonal)
        flat[4] = C[1,0,1,0]   (shear diagonal)
        flat[5] = C[0,1,1,0]   (shear off-diag, = C[1,0,0,1] by major sym.)
    """
    return jnp.array([
        C[0, 0, 0, 0],
        C[1, 1, 1, 1],
        C[0, 0, 1, 1],
        C[0, 1, 0, 1],
        C[1, 0, 1, 0],
        C[0, 1, 1, 0],
    ])


def flat6_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert 6-param flat array to (2,2,2,2) tensor.

    Produces the block-diagonal augmented Voigt::

        [[flat[0]  flat[2]  0        0       ]
         [flat[2]  flat[1]  0        0       ]
         [0        0        flat[3]  flat[5] ]
         [0        0        flat[5]  flat[4] ]]

    Major symmetry is preserved; the shear block has full rank when
    ``flat[3]*flat[4] != flat[5]**2``.
    """
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(flat[0])
    C = C.at[1, 1, 1, 1].set(flat[1])
    C = C.at[0, 0, 1, 1].set(flat[2])
    C = C.at[1, 1, 0, 0].set(flat[2])
    C = C.at[0, 1, 0, 1].set(flat[3])
    C = C.at[1, 0, 1, 0].set(flat[4])
    C = C.at[0, 1, 1, 0].set(flat[5])
    C = C.at[1, 0, 0, 1].set(flat[5])
    return C

# ── 6-param anisotropic Cauchy (EXACTLY the symmetrized-triangle class) ──
#
# Minimal lossless parametrisation of a 2-D minor+major-symmetric (Cauchy)
# stiffness — the symmetric Voigt 3×3
#       [[C11, C12, C16],
#        [C12, C22, C26],
#        [C16, C26, C66]]
# This is the material class produced by ``symmetrize_stiffness`` (the
# symmetrized triangular cloak): it keeps the normal–shear COUPLING C16, C26
# and enforces a SYMMETRIC shear block (C66 = C6̄6̄ = C66̄). It differs from
#   * flat4  — orthotropic Cauchy, forces C16 = C26 = 0;
#   * flat6  — block-diagonal Cosserat, also drops C16/C26 but allows shear
#              asymmetry (C66 ≠ C6̄6̄).
# Unlike flat10 (which also admits non-symmetric Cosserat tensors), flat6cauchy
# is EXACTLY the Cauchy class — so optimising in it stays in the same material
# family as symmetrization. Layout extends flat4: the first four entries are
# identical to ``C_to_flatC`` [C11, C22, C66, C12], with the two coupling
# terms appended.

# minor+major-symmetric index partners for each of C16 (11↔12) and C26 (22↔12).
_C16_PARTNERS = [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]
_C26_PARTNERS = [(1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)]
_C66_PARTNERS = [(0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)]


def C_to_flat6cauchy(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to the 6-param anisotropic-Cauchy flat array.

        flat = [C11, C22, C66, C12, C16, C26]
             = [C0000, C1111, C0101, C0011, C0001, C1101]

    The first four entries match ``C_to_flatC`` (flat4); the last two are the
    normal–shear coupling terms that flat4 drops.
    """
    return jnp.array([
        C[0, 0, 0, 0],  # C11
        C[1, 1, 1, 1],  # C22
        C[0, 1, 0, 1],  # C66
        C[0, 0, 1, 1],  # C12
        C[0, 0, 0, 1],  # C16  (11–12 normal–shear coupling)
        C[1, 1, 0, 1],  # C26  (22–12 normal–shear coupling)
    ])


def flat6cauchy_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert the 6-param anisotropic-Cauchy flat array to (2,2,2,2) tensor.

    Reconstructs the full minor+major-symmetric tensor: the shear block is
    symmetric (single C66 on all four shear slots) and the coupling terms
    C16, C26 populate all their minor/major partners. Round-trips
    ``symmetrize_stiffness`` output exactly.
    """
    C11, C22, C66, C12, C16, C26 = (flat[0], flat[1], flat[2],
                                    flat[3], flat[4], flat[5])
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(C11)
    C = C.at[1, 1, 1, 1].set(C22)
    C = C.at[0, 0, 1, 1].set(C12)
    C = C.at[1, 1, 0, 0].set(C12)
    for (a, b, c, d) in _C66_PARTNERS:
        C = C.at[a, b, c, d].set(C66)
    for (a, b, c, d) in _C16_PARTNERS:
        C = C.at[a, b, c, d].set(C16)
    for (a, b, c, d) in _C26_PARTNERS:
        C = C.at[a, b, c, d].set(C26)
    return C


# ── 10-param (symmetric augmented Voigt, full Cosserat) ──────────────

# Upper-triangle indices of the symmetric 4×4 augmented Voigt matrix.
# Voigt index map: 0→(0,0), 1→(1,1), 2→(0,1), 3→(1,0)
# 10 entries: (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
_UT_IJ = [(0, 0), (0, 1), (0, 2), (0, 3),
          (1, 1), (1, 2), (1, 3),
          (2, 2), (2, 3),
          (3, 3)]


def C_to_flat10(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to 10-param flat array.

    Stores the upper triangle of the symmetric 4×4 augmented Voigt matrix,
    preserving all Cosserat components including normal-shear coupling::

        flat[0]  = M[0,0] = C[0,0,0,0]
        flat[1]  = M[0,1] = C[0,0,1,1]
        flat[2]  = M[0,2] = C[0,0,0,1]
        flat[3]  = M[0,3] = C[0,0,1,0]
        flat[4]  = M[1,1] = C[1,1,1,1]
        flat[5]  = M[1,2] = C[1,1,0,1]
        flat[6]  = M[1,3] = C[1,1,1,0]
        flat[7]  = M[2,2] = C[0,1,0,1]
        flat[8]  = M[2,3] = C[0,1,1,0]
        flat[9]  = M[3,3] = C[1,0,1,0]

    This is the minimal lossless parameterisation for stiffness tensors with
    major symmetry (C_ijkl = C_klij ↔ M = M^T).
    """
    M = C_to_voigt4(C)
    return jnp.array([M[I, J] for I, J in _UT_IJ])


def flat10_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert 10-param flat array to (2,2,2,2) tensor.

    Reconstructs the full symmetric 4×4 augmented Voigt matrix from its
    upper triangle, then converts to tensor form.
    """
    M = jnp.zeros((4, 4))
    for k, (I, J) in enumerate(_UT_IJ):
        M = M.at[I, J].set(flat[k])
        M = M.at[J, I].set(flat[k])
    return voigt4_to_C(M)


# ── augmented Voigt (4×4) conversions ────────────────────────────────

# Augmented Voigt index map: 0→(0,0), 1→(1,1), 2→(0,1), 3→(1,0)
_PAIRS = [(0, 0), (1, 1), (0, 1), (1, 0)]


def C_to_voigt4(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to augmented 4×4 Voigt matrix."""
    M = jnp.zeros((4, 4))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            M = M.at[I, J].set(C[i, j, k, l])
    return M


def voigt4_to_C(M: jnp.ndarray) -> jnp.ndarray:
    """Convert augmented 4×4 Voigt matrix to (2,2,2,2) tensor."""
    C = jnp.zeros((2, 2, 2, 2))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            C = C.at[i, j, k, l].set(M[I, J])
    return C


# ── symmetrisation (Chatzopoulos et al.) ────────────────────────────


def symmetrize_stiffness(Ceff: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise an effective stiffness using the augmented-Voigt recipe.

    Csym_IJ = (C_IJ + C_{Ī J} + C_{I J̄} + C_{Ī J̄}) / 4

    where bar swaps indices 2↔3 (i.e. (0,1)↔(1,0)).
    """
    M = C_to_voigt4(Ceff)
    P = jnp.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])

    Msym = (M + P @ M + M @ P + P @ M @ P) / 4.0
    Msym = 0.5 * (Msym + Msym.T)  # enforce Cauchy symmetry
    return voigt4_to_C(Msym)


# ── position-dependent effective properties ──────────────────────────


def C_eff(
    x: jnp.ndarray,
    geometry: CloakGeometry,
    C0: jnp.ndarray,
    symmetrize: bool = False,
) -> jnp.ndarray:
    """Effective stiffness tensor at position *x*."""
    F = geometry.F_tensor(x)
    J = jnp.linalg.det(F)
    Cnew = jnp.einsum("jJ,lL,iJkL->ijkl", F, F, C0) / J
    if symmetrize:
        Cnew = symmetrize_stiffness(Cnew)
    return jnp.where(geometry.in_cloak(x), Cnew, C0)


def rho_eff(
    x: jnp.ndarray,
    geometry: CloakGeometry,
    rho0: float,
) -> jnp.ndarray:
    """Effective density at position *x*."""
    F = geometry.F_tensor(x)
    J = jnp.linalg.det(F)
    rho_cloak = rho0 / J
    return jnp.where(geometry.in_cloak(x), rho_cloak, rho0)


from rayleigh_cloak.cells import CellDecomposition


# ── flat↔tensor converters keyed by n_C_params ───────────────────────

def _get_converters(n_C_params: int, aniso_cauchy: bool = False):
    """Return ``(to_flat, from_flat)`` functions for the chosen parameterisation.

    Supported values:

    * **10** — symmetric augmented Voigt (recommended).  Minimal lossless
      parameterisation for tensors with major symmetry.  Preserves all
      Cosserat components including normal-shear coupling.
    * **6** — block-diagonal Cosserat.  Captures the normal block and the
      full 2×2 shear block but drops normal-shear coupling.  When
      ``aniso_cauchy=True`` this instead selects the 6-param anisotropic
      **Cauchy** form (``flat6cauchy``: [C11,C22,C66,C12,C16,C26]) — exactly
      the symmetrized-triangle material class (symmetric shear + C16/C26
      coupling).
    * **16** — full augmented 4×4 Voigt (most general, redundant with
      major symmetry).
    * **4** — minor-symmetric shear (deprecated, produces rank-3 Voigt →
      singular stiffness in the full-gradient FEM formulation).
    """
    if n_C_params == 2:
        return C_to_flat2, flat2_to_C
    if n_C_params == 4:
        import warnings
        warnings.warn(
            "n_C_params=4 enforces minor symmetry, producing a rank-3 "
            "augmented Voigt matrix.  This leads to a singular stiffness "
            "matrix in the Cosserat FEM formulation.  Use n_C_params=6 or 16.",
            stacklevel=2,
        )
        return C_to_flatC, flatC_to_C
    if n_C_params == 6:
        if aniso_cauchy:
            return C_to_flat6cauchy, flat6cauchy_to_C
        return C_to_flat6, flat6_to_C
    if n_C_params == 10:
        return C_to_flat10, flat10_to_C
    if n_C_params == 16:
        def _to16(C):
            return C_to_voigt4(C).ravel()
        def _from16(flat):
            return voigt4_to_C(flat.reshape(4, 4))
        return _to16, _from16
    raise ValueError(f"Unsupported n_C_params={n_C_params} (use 2, 6, 10, or 16)")


# ── cell-based material model ────────────────────────────────────────


class CellMaterial:
    """Piecewise-constant material over a :class:`CellDecomposition`.

    Parameters
    ----------
    geometry : CloakGeometry
    C0 : (2,2,2,2) background stiffness
    rho0 : background density
    cell_decomp : CellDecomposition
    n_C_params : number of flat parameters per cell (6 or 16)
    """

    def __init__(
        self,
        geometry: CloakGeometry,
        C0: jnp.ndarray,
        rho0: float,
        cell_decomp: CellDecomposition,
        n_C_params: int = 4,
        symmetrize_init: bool = False,
        init: Literal["pushforward", "homogeneous", "dataset_centroid"] = "pushforward",
        init_path: str | Path | None = None,
        aniso_cauchy: bool = False,
    ):
        self.geometry = geometry
        self.C0 = C0
        self.rho0 = rho0
        self.cell_decomp = cell_decomp
        self.n_C_params = n_C_params
        self.symmetrize_init = symmetrize_init
        self.init = init
        self.init_path = init_path
        self.aniso_cauchy = aniso_cauchy
        self.to_flat, self.from_flat = _get_converters(n_C_params, aniso_cauchy)

        self.cell_C_flat, self.cell_rho = self._initialize()

    def _initialize(self):
        """Compute initial per-cell C and rho.

        Modes:
          * ``pushforward``       — continuous transformation-elasticity seed at
            each cloak cell's centre. Anisotropic, sits outside the cement
            microstructure manifold (high GMM penalty).
          * ``homogeneous``       — every cell starts at the background
            ``(C0, rho0)`` (cement). Pure cement is also outside the
            microstructure manifold (much stiffer than the porous CA cells).
          * ``dataset_centroid``  — cloak cells start at the dataset's
            ``(λ, μ, ρ)`` centroid pulled from ``init_path`` (a GMM .npz
            produced by ``fit_gmm.py``). Inside the manifold by construction,
            so the GMM prior is near-zero at step 0.
        Background cells always stay at ``(C0, rho0)``.
        """
        centers = self.cell_decomp.cell_centers  # (n_cells, 2)
        mask = self.cell_decomp.cloak_mask        # (n_cells,)

        C0_flat = self.to_flat(self.C0)

        if self.init == "dataset_centroid":
            if self.init_path is None:
                raise ValueError(
                    "init='dataset_centroid' requires init_path pointing at a "
                    "GMM .npz (e.g. output/ca_bulk_squared/gmm_lambda_mu_rho.npz)"
                )
            data = np.load(str(self.init_path), allow_pickle=True)
            order = [str(s) for s in np.asarray(data["feature_order"])]
            feat_mean = np.asarray(data["feature_mean"], dtype=float)
            if order == ["lambda", "mu", "rho"]:
                if self.n_C_params != 2:
                    raise ValueError(
                        f"init_path {self.init_path} is a lame (λ, μ, ρ) prior "
                        f"but cells use n_C_params={self.n_C_params}; expected 2."
                    )
                lam_c, mu_c, rho_c = (float(v) for v in feat_mean)
                cloak_C_flat = self.to_flat(C_iso(lam_c, mu_c))
                cloak_rho = rho_c
            elif order == ["C1111", "C2222", "C1212", "C1122", "rho"]:
                if self.n_C_params != 4:
                    raise ValueError(
                        f"init_path {self.init_path} is a flat4 prior but cells "
                        f"use n_C_params={self.n_C_params}; expected 4."
                    )
                # The four stiffness means are already in flat4 column order.
                cloak_C_flat = jnp.asarray(feat_mean[:4])
                cloak_rho = float(feat_mean[4])
            else:
                raise ValueError(
                    f"unexpected feature_order in {self.init_path}: {order}; "
                    "expected ['lambda', 'mu', 'rho'] or "
                    "['C1111', 'C2222', 'C1212', 'C1122', 'rho']"
                )
        elif self.init == "homogeneous":
            cloak_C_flat = C0_flat
            cloak_rho = self.rho0
        else:
            cloak_C_flat = None  # filled per-cell below
            cloak_rho = None

        cell_C_list = []
        cell_rho_list = []
        for i, center in enumerate(centers):
            if not mask[i]:
                cell_C_list.append(C0_flat)
                cell_rho_list.append(self.rho0)
            elif self.init == "pushforward":
                x = jnp.array(center)
                C_i = C_eff(x, self.geometry, self.C0, symmetrize=self.symmetrize_init)
                cell_C_list.append(self.to_flat(C_i))
                cell_rho_list.append(float(rho_eff(x, self.geometry, self.rho0)))
            else:
                cell_C_list.append(cloak_C_flat)
                cell_rho_list.append(cloak_rho)

        return jnp.stack(cell_C_list), jnp.array(cell_rho_list)

    def get_initial_params(self):
        """Return ``(cell_C_flat, cell_rho)`` as a JAX pytree."""
        return (self.cell_C_flat, self.cell_rho)
