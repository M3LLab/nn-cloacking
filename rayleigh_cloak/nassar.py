"""Nassar 2018 lattice cell model — JAX-differentiable.

Implements the Nassar degenerate polar lattice Hooke's law (eq 2.12)
and parameter identification (eq 2.15) in JAX for differentiable
cloaking simulations.

Key design: uses ``inv_kappa = 1/κ`` as parameter instead of ``κ``
to handle the κ→∞ limit smoothly (inv_kappa=0 ↔ minor symmetry).

Reference:
    Nassar H, Chen YY, Huang GL. 2018.
    "A degenerate polar lattice for cloaking in full two-dimensional
     elastodynamics and statics." Proc. R. Soc. A 474: 20180523.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.cells_polar import PolarCellDecomposition
from rayleigh_cloak.materials import C_iso, C_to_voigt4, voigt4_to_C


# ── Nassar forward (eq 2.12) ─────────────────────────────────────────


def nassar_forward_jax(
    theta: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    inv_kappa: jnp.ndarray,
    aspect: jnp.ndarray,
) -> jnp.ndarray:
    """Build augmented 4×4 Voigt from Nassar cell parameters (eq 2.12).

    Uses ``inv_kappa = 1/κ`` so that the κ→∞ limit (minor symmetry)
    is simply ``inv_kappa = 0``.  Fully JAX-traceable.

    Parameters
    ----------
    theta : Lattice angle θ [rad].
    alpha : Diagonal spring constant α.
    beta : Vertical spring constant β.
    inv_kappa : Inverse torsion spring constant 1/κ.
    aspect : Cell aspect ratio a/b.

    Returns
    -------
    C : (4, 4) augmented Voigt stiffness matrix.
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    r = aspect  # a/b

    # Normal block (independent of κ)
    C11 = r * c**2 * alpha
    C22 = (alpha * s**2 + 2.0 * beta) / r
    C12 = c * s * alpha

    # Shear block: reformulated with inv_kappa = 1/κ
    # Original: Δ = α(c - rs)² + rκ  →  Δ/κ = α(c - rs)²/κ + r
    #           C33 = c²ακ/Δ = c²α / (Δ/κ)
    denom = alpha * (c - r * s) ** 2 * inv_kappa + r
    C33 = c**2 * alpha / denom
    C44 = r**2 * s**2 * alpha / denom
    C34 = r * c * s * alpha / denom

    M = jnp.array([
        [C11, C12, 0.0, 0.0],
        [C12, C22, 0.0, 0.0],
        [0.0, 0.0, C33, C34],
        [0.0, 0.0, C34, C44],
    ])
    return M


# ── Parameter identification (eq 2.15) ───────────────────────────────


def nassar_init_from_lame(
    lam: float,
    mu: float,
    f: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Nassar cell parameters from Lamé constants and deformation f.

    This is the forward design formula (eq 2.15).

    Parameters
    ----------
    lam : Lamé constant λ.
    mu : Lamé constant μ (requires μ ≤ λ for stability).
    f : Position-dependent deformation f = (‖x‖ − rᵢ)/‖x‖ ∈ (0, 1).

    Returns
    -------
    theta, alpha, beta, inv_kappa, aspect
    """
    theta = jnp.arctan(jnp.sqrt(lam / (2.0 * mu + lam)))
    alpha = 2.0 * (mu + lam) * jnp.sqrt(lam / (2.0 * mu + lam))
    beta = ((2.0 * mu + lam) ** 2 - lam**2) / (
        2.0 * jnp.sqrt(lam) * jnp.sqrt(2.0 * mu + lam)
    )
    aspect = f * jnp.sqrt((2.0 * mu + lam) / lam)

    # inv_kappa = 1/κ = (λ−μ)/(λμ) · f/(1−f)²
    # When λ = μ: inv_kappa = 0 (κ → ∞)
    inv_kappa = jnp.where(
        jnp.abs(lam - mu) < 1e-12 * jnp.maximum(lam, 1e-30),
        0.0,
        (lam - mu) / (lam * mu) * f / (1.0 - f) ** 2,
    )

    return theta, alpha, beta, inv_kappa, aspect


# ── Tensor rotation ──────────────────────────────────────────────────


def rotate_C_tensor(C_local: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Rotate a (2,2,2,2) stiffness tensor by angle phi.

    Transforms from local polar frame (m=radial, n=tangential) to
    global Cartesian frame (x, y).

    The rotation matrix R maps local basis vectors to global:
        e_x = cos(φ) e_m − sin(φ) e_n
        e_y = sin(φ) e_m + cos(φ) e_n

    Parameters
    ----------
    C_local : (2,2,2,2) stiffness tensor in local (m,n) frame.
    phi : Polar angle [rad] of the point (angle from x-axis to radial dir).

    Returns
    -------
    C_global : (2,2,2,2) stiffness tensor in global (x,y) frame.
    """
    cp = jnp.cos(phi)
    sp = jnp.sin(phi)
    R = jnp.array([[cp, -sp],
                    [sp,  cp]])
    return jnp.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C_local)


# ── Combined: Nassar params → Cartesian C tensor ────────────────────


def nassar_C_cartesian(
    theta: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    inv_kappa: jnp.ndarray,
    aspect: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    """Full pipeline: Nassar params → local Voigt → (2,2,2,2) → rotate.

    Returns
    -------
    C : (2,2,2,2) stiffness tensor in Cartesian frame.
    """
    M = nassar_forward_jax(theta, alpha, beta, inv_kappa, aspect)
    C_local = voigt4_to_C(M)
    return rotate_C_tensor(C_local, phi)


def nassar_C_cartesian_with_bg(
    theta: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    inv_kappa: jnp.ndarray,
    aspect: jnp.ndarray,
    phi: jnp.ndarray,
    is_cloak: jnp.ndarray,
    C0: jnp.ndarray,
) -> jnp.ndarray:
    """Nassar C in cloak cells, background C0 outside."""
    C_nassar = nassar_C_cartesian(theta, alpha, beta, inv_kappa, aspect, phi)
    return jnp.where(is_cloak, C_nassar, C0)


# ── Effective density (eq 2.6) ───────────────────────────────────────


def nassar_rho_eff(rho0: float, ri: float, rc: float, f: float) -> float:
    """Position-dependent effective density inside the cloak (eq 2.6).

    ρ_eff = rc² / (rc − rᵢ)² · f · R

    where f = (‖x‖ − rᵢ)/‖x‖ is the local deformation parameter.
    This equals R/J where J = det(F) = (rc−rᵢ)²/(rc²·f).
    """
    return rc**2 / (rc - ri) ** 2 * f * rho0


# ── NassarCellMaterial ───────────────────────────────────────────────


class NassarCellMaterial:
    """Cell-based material model using Nassar lattice parameterization.

    Each cloak cell has 5 Nassar parameters (θ, α, β, 1/κ, a/b) plus density ρ.
    The polar angle φ per cell is fixed (geometric) and not optimized.

    Parameters
    ----------
    geometry : CircularCloakGeometry
    lam, mu : Lamé constants of background medium.
    rho0 : Background density.
    cell_decomp : CellDecomposition over the cloak bounding box.
    """

    def __init__(self, geometry, lam: float, mu: float, rho0: float,
                 cell_decomp: CellDecomposition):
        self.geometry = geometry
        self.lam = lam
        self.mu = mu
        self.rho0 = rho0
        self.cell_decomp = cell_decomp
        self.C0 = C_iso(lam, mu)

        # Compute fixed polar angles and cloak membership per cell
        centers = cell_decomp.cell_centers  # (n_cells, 2)
        cx = centers[:, 0] - geometry.x_c
        cy = centers[:, 1] - geometry.y_c
        self.phi = np.arctan2(cy, cx).astype(np.float64)  # polar angle
        self.is_cloak = cell_decomp.cloak_mask  # bool array

        # Initialize Nassar params from eq 2.15
        self._init_params()

    def _init_params(self):
        """Compute initial per-cell Nassar parameters from eq 2.15."""
        geo = self.geometry
        centers = self.cell_decomp.cell_centers
        n_cells = self.cell_decomp.n_cells

        theta_arr = np.zeros(n_cells)
        alpha_arr = np.zeros(n_cells)
        beta_arr = np.zeros(n_cells)
        inv_kappa_arr = np.zeros(n_cells)
        aspect_arr = np.zeros(n_cells)
        rho_arr = np.full(n_cells, self.rho0)

        # Global Nassar params (same for all cells)
        theta_glob = np.arctan(np.sqrt(self.lam / (2.0 * self.mu + self.lam)))
        alpha_glob = 2.0 * (self.mu + self.lam) * np.sqrt(
            self.lam / (2.0 * self.mu + self.lam))
        beta_glob = ((2.0 * self.mu + self.lam)**2 - self.lam**2) / (
            2.0 * np.sqrt(self.lam) * np.sqrt(2.0 * self.mu + self.lam))

        for i in range(n_cells):
            if not self.is_cloak[i]:
                # Background cell: use global params with unit aspect (dummy)
                theta_arr[i] = theta_glob
                alpha_arr[i] = alpha_glob
                beta_arr[i] = beta_glob
                inv_kappa_arr[i] = 0.0
                aspect_arr[i] = 1.0
                continue

            cx = centers[i, 0] - geo.x_c
            cy = centers[i, 1] - geo.y_c
            r = np.sqrt(cx**2 + cy**2)
            f = (r - geo.ri) / r  # deformation parameter

            # Eq 2.15
            theta_arr[i] = theta_glob
            alpha_arr[i] = alpha_glob
            beta_arr[i] = beta_glob
            aspect_arr[i] = f * np.sqrt((2.0 * self.mu + self.lam) / self.lam)

            if abs(self.lam - self.mu) < 1e-12 * self.lam:
                inv_kappa_arr[i] = 0.0
            else:
                inv_kappa_arr[i] = ((self.lam - self.mu) / (self.lam * self.mu)
                                    * f / (1.0 - f)**2)

            # Position-dependent density (eq 2.6): ρ = rc²/(rc-ri)² · f · R
            rho_arr[i] = nassar_rho_eff(self.rho0, geo.ri, geo.rc, f)

        self.cell_theta = jnp.array(theta_arr)
        self.cell_alpha = jnp.array(alpha_arr)
        self.cell_beta = jnp.array(beta_arr)
        self.cell_inv_kappa = jnp.array(inv_kappa_arr)
        self.cell_aspect = jnp.array(aspect_arr)
        self.cell_rho = jnp.array(rho_arr)
        self.cell_phi = jnp.array(self.phi)
        self.cell_is_cloak = jnp.array(self.is_cloak, dtype=jnp.float32)

    def get_initial_params(self):
        """Return optimizable params as a JAX pytree.

        Returns
        -------
        params : tuple of (cell_C_flat, cell_rho) where
            cell_C_flat : (n_cells, 5) array [theta, alpha, beta, inv_kappa, aspect]
            cell_rho : (n_cells,) array
        """
        cell_nassar = jnp.stack([
            self.cell_theta,
            self.cell_alpha,
            self.cell_beta,
            self.cell_inv_kappa,
            self.cell_aspect,
        ], axis=-1)  # (n_cells, 5)
        return (cell_nassar, self.cell_rho)

    def params_to_C_full(self, params):
        """Convert Nassar param arrays → per-cell C(2,2,2,2) tensors.

        This is the differentiable path: params → stiffness tensors.

        Parameters
        ----------
        params : (cell_nassar, cell_rho) where
            cell_nassar : (n_cells, 5) [theta, alpha, beta, inv_kappa, aspect]
            cell_rho : (n_cells,)

        Returns
        -------
        cell_C : (n_cells, 2, 2, 2, 2)
        cell_rho : (n_cells,)
        """
        cell_nassar, cell_rho = params
        phi = self.cell_phi       # (n_cells,) — fixed
        is_cloak = self.cell_is_cloak  # (n_cells,) — fixed

        def _single_cell(nassar_5, phi_i, is_cloak_i):
            C = nassar_C_cartesian(
                nassar_5[0], nassar_5[1], nassar_5[2],
                nassar_5[3], nassar_5[4], phi_i,
            )
            # Use background C0 for non-cloak cells
            return jnp.where(is_cloak_i, C, self.C0)

        cell_C = jax.vmap(_single_cell)(cell_nassar, phi, is_cloak)
        return cell_C, cell_rho


# ── NassarPolarMaterial ────────────────────────────────────────────


class NassarPolarMaterial:
    """Cell-based material using Nassar parameterization on a polar grid.

    Unlike :class:`NassarCellMaterial` which uses a Cartesian grid and samples
    the continuous C_eff at cell centres, this class uses a polar grid
    (N sectors × M layers) matching the physical Nassar 2018 lattice layout.

    Parameters from eq 2.15: θ, α, β are uniform across all cells; only κ
    and a/b vary radially through the deformation parameter f = (r − rᵢ)/r.

    Parameters
    ----------
    geometry : CircularCloakGeometry
    lam, mu : Lamé constants of background medium.
    rho0 : Background density.
    polar_decomp : PolarCellDecomposition
    """

    def __init__(self, geometry, lam: float, mu: float, rho0: float,
                 polar_decomp: PolarCellDecomposition):
        self.geometry = geometry
        self.lam = lam
        self.mu = mu
        self.rho0 = rho0
        self.polar_decomp = polar_decomp
        self.C0 = C_iso(lam, mu)

        self._init_params()

    def _init_params(self):
        """Compute per-cell Nassar parameters from eq 2.15."""
        pd = self.polar_decomp
        n = pd.n_cells

        # Global Nassar params (uniform across all cells, eq 2.15)
        lam, mu = self.lam, self.mu
        theta_glob = np.arctan(np.sqrt(lam / (2.0 * mu + lam)))
        alpha_glob = 2.0 * (mu + lam) * np.sqrt(lam / (2.0 * mu + lam))
        beta_glob = ((2.0 * mu + lam)**2 - lam**2) / (
            2.0 * np.sqrt(lam) * np.sqrt(2.0 * mu + lam))

        # Per-cell deformation parameter
        f = pd.cell_f  # (n_cells,) = (r - ri) / r

        # Aspect ratio a/b (eq 2.15)
        aspect = f * np.sqrt((2.0 * mu + lam) / lam)

        # Inverse torsion spring constant (eq 2.15)
        if abs(lam - mu) < 1e-12 * lam:
            inv_kappa = np.zeros(n)
        else:
            inv_kappa = (lam - mu) / (lam * mu) * f / (1.0 - f)**2

        # Density (eq 2.6): ρ_eff = rc²/(rc-ri)² · f · ρ₀
        ri, rc = self.geometry.ri, self.geometry.rc
        rho_arr = rc**2 / (rc - ri)**2 * f * self.rho0

        self.cell_theta = jnp.full(n, theta_glob)
        self.cell_alpha = jnp.full(n, alpha_glob)
        self.cell_beta = jnp.full(n, beta_glob)
        self.cell_inv_kappa = jnp.array(inv_kappa)
        self.cell_aspect = jnp.array(aspect)
        self.cell_rho = jnp.array(rho_arr)
        self.cell_phi = jnp.array(pd.cell_phi)  # polar angle per cell (fixed)

    def get_initial_params(self):
        """Return optimizable params as a JAX pytree.

        Returns
        -------
        params : tuple of (cell_nassar, cell_rho) where
            cell_nassar : (n_cells, 5) [theta, alpha, beta, inv_kappa, aspect]
            cell_rho : (n_cells,)
        """
        cell_nassar = jnp.stack([
            self.cell_theta,
            self.cell_alpha,
            self.cell_beta,
            self.cell_inv_kappa,
            self.cell_aspect,
        ], axis=-1)  # (n_cells, 5)
        return (cell_nassar, self.cell_rho)

    def params_to_C_full(self, params):
        """Convert Nassar param arrays → per-cell C(2,2,2,2) tensors.

        Parameters
        ----------
        params : (cell_nassar, cell_rho)

        Returns
        -------
        cell_C : (n_cells, 2, 2, 2, 2)
        cell_rho : (n_cells,)
        """
        cell_nassar, cell_rho = params
        phi = self.cell_phi  # (n_cells,) — fixed

        def _single_cell(nassar_5, phi_i):
            return nassar_C_cartesian(
                nassar_5[0], nassar_5[1], nassar_5[2],
                nassar_5[3], nassar_5[4], phi_i,
            )

        cell_C = jax.vmap(_single_cell)(cell_nassar, phi)
        return cell_C, cell_rho
