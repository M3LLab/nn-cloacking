"""Topology neural reparameterisation: pixel-level density prediction.

The MLP maps pixel-center coordinates to a single density value (0 = void,
1 = solid).  Material properties are assigned via SIMP penalisation:

    E(x) = E_cement * rho(x)^p
    lam(x) = lam_cement * rho(x)^p
    mu(x) = mu_cement * rho(x)^p
    rho_mass(x) = rho_cement * rho(x)

Each pixel is a cell in the existing ``CellDecomposition`` pipeline, so
``expand_to_quadpoints`` maps FEM quadrature points to pixel-level material.
No per-cell homogenisation is needed — the outer FEM solver resolves the
microstructure directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.neural_reparam import (
    fourier_features,
    init_mlp,
    mlp_forward,
    random_fourier_features,
)
from rayleigh_cloak.optimize import (
    adam_init,
    adam_update,
    cloaking_loss,
    l2_regularization,
)


# ── Reparameterisation ───────────────────────────────────────────────


@dataclass
class NeuralReparamTopo:
    """Wraps an MLP that maps pixel coordinates to material density.

    Uses a residual-in-logit-space approach: the baseline logit is computed
    from dataset-matched binary targets, and the MLP predicts a small
    correction δ so that density = sigmoid(logit_baseline + output_scale * δ).
    This starts exactly at the dataset-matched pattern without pretraining.

    Attributes
    ----------
    cell_features : (n_pixels, n_features) Fourier features
    logit_baseline : (n_pixels,) logit of dataset-matched target densities
    cloak_mask : (n_pixels,) bool — which pixels are in the cloak
    cloak_idx : (n_cloak,) int — indices of cloak pixels (static)
    C0_flat : (n_C_params,) background C_flat (solid cement)
    rho0 : float — background mass density
    lam_cement : float — Lamé parameter λ of solid phase
    mu_cement : float — Lamé parameter μ of solid phase
    rho_cement : float — mass density of solid phase
    output_scale : float — scale of MLP correction in logit space
    simp_p : float — SIMP penalisation exponent
    n_C_params : int — flat stiffness dimension (must be 2 for isotropic)
    """

    cell_features: jnp.ndarray
    logit_baseline: jnp.ndarray
    cloak_mask: jnp.ndarray
    cloak_idx: jnp.ndarray
    C0_flat: jnp.ndarray
    rho0: float
    lam_cement: float
    mu_cement: float
    rho_cement: float
    n_x: int                      # pixel grid x-dimension
    n_y: int                      # pixel grid y-dimension
    output_scale: float = 0.1
    simp_p: float = 3.0
    n_C_params: int = 2

    def _cloak_channel_densities(
        self, theta: list[dict], beta: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute per-channel densities for cloak pixels.

        Returns (d_lam, d_mu, d_rho), each shape (n_cloak,).
        The same logit baseline is shared across channels; the MLP's
        3-channel output provides independent corrections.
        """
        cloak_features = self.cell_features[self.cloak_idx]       # (n_cloak, n_features)
        cloak_baseline = self.logit_baseline[self.cloak_idx]      # (n_cloak,)
        raw = mlp_forward(theta, cloak_features)                  # (n_cloak, 3)
        logits = cloak_baseline[:, None] + self.output_scale * raw  # (n_cloak, 3)
        soft = jax.nn.sigmoid(logits)
        projected = jax.nn.sigmoid(beta * (soft - 0.5))
        return projected[:, 0], projected[:, 1], projected[:, 2]

    def decode(self, theta: list[dict], beta: float = 1.0) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Map MLP weights → (cell_C_flat, cell_rho).

        Three independent density channels decouple lam, mu, rho:
            lam = lam_cement * d_lam^p
            mu  = mu_cement  * d_mu^p
            rho = rho_cement * d_rho
        Non-cloak pixels get background values directly.
        """
        n_pixels = self.cell_features.shape[0]
        d_lam, d_mu, d_rho = self._cloak_channel_densities(theta, beta)

        lam = self.lam_cement * d_lam ** self.simp_p
        mu = self.mu_cement * d_mu ** self.simp_p

        # Scatter cloak values into full arrays with background defaults
        cell_C_flat = jnp.tile(self.C0_flat, (n_pixels, 1))         # (n_pixels, 2)
        cloak_C = jnp.stack([lam, mu], axis=-1)                     # (n_cloak, 2)
        cell_C_flat = cell_C_flat.at[self.cloak_idx].set(cloak_C)

        cell_rho = jnp.full(n_pixels, self.rho0)                    # (n_pixels,)
        cell_rho = cell_rho.at[self.cloak_idx].set(self.rho_cement * d_rho)

        return (cell_C_flat, cell_rho)

    def get_densities(self, theta: list[dict], beta: float = 1.0) -> jnp.ndarray:
        """Pixel densities (rho channel): cloak from MLP, non-cloak = -1."""
        n_pixels = self.cell_features.shape[0]
        _, _, d_rho = self._cloak_channel_densities(theta, beta)
        out = jnp.full(n_pixels, -1.0)
        return out.at[self.cloak_idx].set(d_rho)

    def get_density_grid(self, theta: list[dict], beta: float = 1.0) -> np.ndarray:
        """Return densities as a 2D (n_y, n_x) array for visualisation.

        Cloak pixels have density in [0, 1]; non-cloak pixels are NaN.
        The array is oriented so that row 0 = bottom (low y) and the last
        row = top (high y), suitable for ``imshow(origin='lower')``.
        """
        densities = np.asarray(self.get_densities(theta, beta=beta))  # (n_pixels,)
        densities = np.where(densities < 0, np.nan, densities)
        grid = densities.reshape(self.n_x, self.n_y)       # (n_x, n_y)
        return grid.T                                       # (n_y, n_x)


def make_neural_reparam_topo(
    cell_decomp: CellDecomposition,
    C0_flat: jnp.ndarray,
    rho0: float,
    lam_cement: float,
    mu_cement: float,
    rho_cement: float,
    pixel_targets: np.ndarray,
    *,
    hidden_size: int = 256,
    n_layers: int = 4,
    n_fourier: int = 32,
    seed: int = 42,
    simp_p: float = 3.0,
    output_scale: float = 0.1,
    density_eps: float = 0.01,
    fourier_sigma: float = 0.0,
) -> tuple[list[dict], NeuralReparamTopo]:
    """Create a NeuralReparamTopo and initialise MLP weights.

    Parameters
    ----------
    pixel_targets : (n_pixels,) float — dataset-matched target densities
        in [0, 1].  Used to compute the logit baseline so that the MLP
        starts exactly at the matched pattern.
    density_eps : float — clamp targets away from 0/1 to keep logits finite.
    output_scale : float — scale of MLP correction in logit space.
    fourier_sigma : float — bandwidth for random Fourier features.
        If > 0, uses random projections B ~ N(0, sigma^2) instead of
        axis-aligned linspace frequencies.  Higher values resolve finer
        spatial detail.

    Returns
    -------
    theta : MLP parameters (list of {W, b})
    reparam : NeuralReparamTopo instance with decode() method
    """
    n_out = 3  # independent density channels: lam, mu, rho

    # Normalise pixel centres to [0, 1]
    centers = jnp.array(cell_decomp.cell_centers)
    lo = centers.min(axis=0)
    hi = centers.max(axis=0)
    centers_norm = (centers - lo) / (hi - lo + 1e-10)

    if fourier_sigma > 0:
        fourier_key = jax.random.PRNGKey(seed + 1)
        features, _ = random_fourier_features(
            centers_norm, n_freq=n_fourier, sigma=fourier_sigma, key=fourier_key,
        )
    else:
        features = fourier_features(centers_norm, n_fourier)
    n_features = features.shape[1]

    mask = jnp.array(cell_decomp.cloak_mask)
    cloak_idx = jnp.where(mask)[0]

    # Compute logit baseline from dataset-matched targets
    # Clamp to [eps, 1-eps] to avoid ±inf logits
    targets_clamped = jnp.clip(
        jnp.array(pixel_targets), density_eps, 1.0 - density_eps)
    logit_baseline = jnp.log(targets_clamped / (1.0 - targets_clamped))

    # MLP: features → hidden → ... → 1
    layer_sizes = [n_features] + [hidden_size] * (n_layers - 1) + [n_out]
    key = jax.random.PRNGKey(seed)
    theta = init_mlp(key, layer_sizes)

    # Scale down last layer so initial MLP output ≈ 0 → start at baseline
    theta[-1]["W"] = theta[-1]["W"] * 0.01
    theta[-1]["b"] = theta[-1]["b"] * 0.0

    reparam = NeuralReparamTopo(
        cell_features=features,
        logit_baseline=logit_baseline,
        cloak_mask=mask,
        cloak_idx=cloak_idx,
        C0_flat=C0_flat,
        rho0=rho0,
        lam_cement=lam_cement,
        mu_cement=mu_cement,
        rho_cement=rho_cement,
        n_x=cell_decomp.n_x,
        n_y=cell_decomp.n_y,
        output_scale=output_scale,
        simp_p=simp_p,
        n_C_params=2,
    )

    return theta, reparam


# ── Binarisation penalty ────────────────────────────────────────────


def binarisation_penalty(
    theta: list[dict],
    reparam: NeuralReparamTopo,
    beta: float = 1.0,
) -> jnp.ndarray:
    """SIMP binarisation penalty: 4*d*(1-d) averaged over all channels."""
    d_lam, d_mu, d_rho = reparam._cloak_channel_densities(theta, beta=beta)
    pen = (jnp.mean(4.0 * d_lam * (1.0 - d_lam))
         + jnp.mean(4.0 * d_mu * (1.0 - d_mu))
         + jnp.mean(4.0 * d_rho * (1.0 - d_rho)))
    return pen / 3.0


def material_proximity_metrics(
    theta: list[dict],
    reparam: NeuralReparamTopo,
    beta: float = 1.0,
) -> dict[str, float]:
    """Material metrics split by solid (d_rho > 0.5) and void pixels.

    Solid pixels: average relative error from cement for each channel.
    Void pixels: average rho density (lower = closer to void).
    """
    d_lam, d_mu, d_rho = reparam._cloak_channel_densities(theta, beta=beta)

    solid = d_rho > 0.5
    n_solid = jnp.sum(solid)
    n_void = jnp.sum(~solid)

    lam_err = jnp.sum(jnp.where(solid, jnp.abs(1.0 - d_lam ** reparam.simp_p), 0.0))
    mu_err = jnp.sum(jnp.where(solid, jnp.abs(1.0 - d_mu ** reparam.simp_p), 0.0))
    rho_err = jnp.sum(jnp.where(solid, jnp.abs(1.0 - d_rho), 0.0))

    avg_rho_void = jnp.sum(jnp.where(~solid, d_rho, 0.0))

    return {
        "n_solid": int(n_solid),
        "n_void": int(n_void),
        "lam_rel_err": float(lam_err / jnp.maximum(n_solid, 1)),
        "mu_rel_err": float(mu_err / jnp.maximum(n_solid, 1)),
        "rho_rel_err": float(rho_err / jnp.maximum(n_solid, 1)),
        "avg_rho_void": float(avg_rho_void / jnp.maximum(n_void, 1)),
    }


# ── Optimisation loop ───────────────────────────────────────────────


@dataclass
class TopoOptimizationResult:
    """Result of topology neural reparameterised optimisation."""
    theta: list[dict]
    params: tuple[jnp.ndarray, jnp.ndarray]
    loss_history: list[float] = field(default_factory=list)
    cloak_history: list[float] = field(default_factory=list)
    l2_history: list[float] = field(default_factory=list)
    bin_history: list[float] = field(default_factory=list)
    lam_rel_err_history: list[float] = field(default_factory=list)
    mu_rel_err_history: list[float] = field(default_factory=list)
    rho_rel_err_history: list[float] = field(default_factory=list)
    avg_rho_void_history: list[float] = field(default_factory=list)


def plot_density_grid(
    density_grid: np.ndarray,
    step: int,
    save_path: str,
    cloak_mask_2d: np.ndarray | None = None,
) -> None:
    """Save a density-field image showing the current microstructure.

    Parameters
    ----------
    density_grid : (n_y, n_x) density values in [0, 1]
    step : optimisation step index (for the title)
    save_path : output PNG path
    cloak_mask_2d : optional (n_y, n_x) bool mask — pixels outside cloak
        are shown in a distinct colour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot density with a blue→white→black colourmap (0=void, 1=solid)
    im = ax.imshow(
        density_grid,
        origin="lower",
        cmap="gray_r",
        vmin=0, vmax=1,
        aspect="equal",
        interpolation="nearest",
    )

    # Overlay cloak boundary if mask is provided
    if cloak_mask_2d is not None:
        boundary = np.zeros_like(density_grid, dtype=float)
        boundary[~cloak_mask_2d] = np.nan
        boundary[cloak_mask_2d] = np.nan
        # Draw contour of cloak region
        ax.contour(
            cloak_mask_2d.astype(float),
            levels=[0.5], colors=["red"], linewidths=1, linestyles="--",
        )

    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Density")
    ax.set_title(f"Microstructure density — step {step}")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_optimization_neural_topo(
    fwd_pred,
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    u_ref_boundary: jnp.ndarray,
    boundary_indices: jnp.ndarray,
    reparam: NeuralReparamTopo,
    theta_init: list[dict],
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "linear",
    lambda_l2: float = 1e-4,
    lambda_bin: float = 0.01,
    beta_start: float = 1.0,
    beta_end: float = 32.0,
    plot_callback=None,
    density_callback=None,
    plot_every: int = 1,
    step_callback=None,
    loss_fn=None,
) -> TopoOptimizationResult:
    """Run optimisation over MLP weights (topology reparameterisation).

    Parameters
    ----------
    fwd_pred : callable
        Differentiable forward prediction from ``ad_wrapper``.
    params_init : (cell_C_flat, cell_rho) — initial material values
    reparam : NeuralReparamTopo with decode()
    theta_init : initial MLP weights
    lambda_bin : binarisation penalty weight
    density_callback : optional callable(step, density_grid)
        Called with the step index and (n_y, n_x) density array every
        ``plot_every`` steps.
    """
    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = adam_init(theta)
    loss_history: list[float] = []
    cloak_history: list[float] = []
    l2_history: list[float] = []
    bin_history: list[float] = []
    lam_rel_err_history: list[float] = []
    mu_rel_err_history: list[float] = []
    rho_rel_err_history: list[float] = []
    avg_rho_void_history: list[float] = []

    if loss_fn is None:
        loss_fn = cloaking_loss
    _cloak_loss_fn = loss_fn

    boundary_indices_jnp = jnp.array(boundary_indices)

    def _loss_fn(theta, beta):
        params = reparam.decode(theta, beta=beta)
        sol_list = fwd_pred(params)
        u_cloak = sol_list[0]
        L_cloak = _cloak_loss_fn(u_cloak, u_ref_boundary, boundary_indices_jnp)
        L_l2 = l2_regularization(params, params_init)
        L_bin = binarisation_penalty(theta, reparam, beta=beta)
        return L_cloak + lambda_l2 * L_l2 + lambda_bin * L_bin, L_cloak

    loss_and_grad = jax.value_and_grad(
        lambda t, b: _loss_fn(t, b)[0],
        argnums=0,
        has_aux=False,
    )

    def _get_components(theta, beta):
        params = reparam.decode(theta, beta=beta)
        L_l2 = float(l2_regularization(params, params_init))
        L_bin = float(binarisation_penalty(theta, reparam, beta=beta))
        return L_l2, L_bin

    for step in range(n_iters):
        # Linear annealing of Heaviside projection sharpness
        beta = beta_start + (beta_end - beta_start) * step / max(n_iters - 1, 1)
        # Learning rate schedule
        t_frac = step / max(n_iters - 1, 1)
        if lr_end is None:
            cur_lr = lr
        elif lr_schedule == "cosine":
            cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * t_frac))
        else:
            cur_lr = lr + (lr_end - lr) * t_frac
        loss_val, grads = loss_and_grad(theta, beta)
        loss_val_float = float(loss_val)
        loss_history.append(loss_val_float)

        L_l2, L_bin = _get_components(theta, beta)
        L_cloak = loss_val_float - lambda_l2 * L_l2 - lambda_bin * L_bin
        cloak_history.append(L_cloak)
        l2_history.append(L_l2)
        bin_history.append(L_bin)

        mat_metrics = material_proximity_metrics(theta, reparam, beta=beta)
        lam_rel_err_history.append(mat_metrics["lam_rel_err"])
        mu_rel_err_history.append(mat_metrics["mu_rel_err"])
        rho_rel_err_history.append(mat_metrics["rho_rel_err"])
        avg_rho_void_history.append(mat_metrics["avg_rho_void"])

        grad_norm = float(jnp.sqrt(sum(
            jnp.sum(l["W"]**2) + jnp.sum(l["b"]**2) for l in grads
        )))
        n_s = mat_metrics["n_solid"]
        n_v = mat_metrics["n_void"]
        print(
            f"  Step {step:4d} | total = {loss_val_float:.4e}"
            f"  cloak_pct = {np.sqrt(max(L_cloak, 0)) * 100:.2f}"
            f"  L2 = {L_l2:.4e}"
            f"  bin = {L_bin:.4e}"
            f"  solid({n_s}): lam_err={mat_metrics['lam_rel_err']:.3f}"
            f" mu_err={mat_metrics['mu_rel_err']:.3f}"
            f" rho_err={mat_metrics['rho_rel_err']:.3f}"
            f"  void({n_v}): avg_rho={mat_metrics['avg_rho_void']:.3f}"
            f"  beta={beta:.1f}"
            f"  lr={cur_lr:.2e}"
            f"  |grad|={grad_norm:.4e}"
        )

        if step_callback is not None:
            params = reparam.decode(theta, beta=beta)
            step_callback(step, loss_val_float, L_cloak, L_l2, L_bin, params,
                          mat_metrics)

        if step % plot_every == 0:
            if plot_callback is not None:
                params = reparam.decode(theta, beta=beta)
                sol_list = fwd_pred(params)
                plot_callback(step, np.asarray(sol_list[0]))
            if density_callback is not None:
                density_callback(step, reparam.get_density_grid(theta, beta=beta))

        updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
        theta = jax.tree.map(lambda p, u: p + u, theta, updates)

    # Final state (use beta_end for sharpest projection)
    final_params = reparam.decode(theta, beta=beta_end)

    if plot_callback is not None:
        sol_list = fwd_pred(final_params)
        plot_callback(n_iters, np.asarray(sol_list[0]))
    if density_callback is not None:
        density_callback(n_iters, reparam.get_density_grid(theta, beta=beta_end))

    return TopoOptimizationResult(
        theta=theta,
        params=final_params,
        loss_history=loss_history,
        cloak_history=cloak_history,
        l2_history=l2_history,
        bin_history=bin_history,
        lam_rel_err_history=lam_rel_err_history,
        mu_rel_err_history=mu_rel_err_history,
        rho_rel_err_history=rho_rel_err_history,
        avg_rho_void_history=avg_rho_void_history,
    )
