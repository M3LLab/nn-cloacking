"""Neural reparameterization of cell-based material fields.

Instead of optimizing raw per-cell (C_flat, rho) arrays directly, a small MLP
maps cell-center coordinates to material parameters.  The MLP weights become
the optimization variables; gradients flow through the FEM adjoint and then
back through the network via standard JAX autodiff.

The MLP's smoothness bias replaces explicit neighbor regularization.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.loss import transmitted_displacement_ratio
from rayleigh_cloak.material_prior import GMMPrior, gmm_flat_top_penalty
from rayleigh_cloak.optimize import (
    AdamState,
    adam_init,
    adam_update,
    cloaking_loss,
    l2_regularization,
)


# ── MLP definition (pure JAX) ───────────────────────────────────────


def _init_layer(key, n_in, n_out):
    """Xavier-uniform initialization for a single dense layer."""
    k1, k2 = jax.random.split(key)
    bound = jnp.sqrt(6.0 / (n_in + n_out))
    W = jax.random.uniform(k1, (n_in, n_out), minval=-bound, maxval=bound)
    b = jnp.zeros(n_out)
    return {"W": W, "b": b}


def init_mlp(key, layer_sizes: list[int]) -> list[dict]:
    """Initialize an MLP as a list of {W, b} dicts."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        params.append(_init_layer(subkey, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_forward(params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: Dense → tanh → ... → Dense (no final activation)."""
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def fourier_features(xy: jnp.ndarray, n_freq: int = 32) -> jnp.ndarray:
    """Map (n, 2) coordinates to (n, 4*n_freq) Fourier features.

    Frequencies are log-spaced from 1 to ``n_freq`` to capture both
    large-scale gradients and sharper spatial transitions.
    """
    freqs = jnp.linspace(1.0, float(n_freq), n_freq)  # (n_freq,)
    # (n, 2) @ (2,) → project each coord onto each freq
    proj_x = xy[:, 0:1] * freqs[None, :]  # (n, n_freq)
    proj_y = xy[:, 1:2] * freqs[None, :]  # (n, n_freq)
    return jnp.concatenate([
        jnp.sin(proj_x), jnp.cos(proj_x),
        jnp.sin(proj_y), jnp.cos(proj_y),
    ], axis=-1)  # (n, 4*n_freq)


def random_fourier_features(
    xy: jnp.ndarray,
    n_freq: int = 256,
    sigma: float = 10.0,
    key: jax.random.PRNGKey | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map (n, 2) coordinates to (n, 2*n_freq) random Fourier features.

    Draws a random projection matrix B ~ N(0, sigma^2) of shape (2, n_freq).
    The features are [sin(2π xy B), cos(2π xy B)].  Higher sigma enables
    the network to represent higher spatial frequencies.

    Returns (features, B) so that B can be stored for reproducibility.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    B = sigma * jax.random.normal(key, (2, n_freq))  # (2, n_freq)
    proj = 2.0 * jnp.pi * xy @ B                      # (n, n_freq)
    features = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
    return features, B


# ── Weight I/O ──────────────────────────────────────────────────────


def save_theta(
    theta: list[dict],
    path: str,
    opt_state: AdamState | None = None,
) -> None:
    """Save MLP weights (and optionally Adam state) to an .npz file."""
    arrays = {}
    for i, layer in enumerate(theta):
        arrays[f"W_{i}"] = np.asarray(layer["W"])
        arrays[f"b_{i}"] = np.asarray(layer["b"])
    arrays["n_layers"] = np.array(len(theta))
    if opt_state is not None:
        arrays["adam_t"] = np.array(opt_state.t)
        for i, layer in enumerate(opt_state.m):
            arrays[f"adam_m_W_{i}"] = np.asarray(layer["W"])
            arrays[f"adam_m_b_{i}"] = np.asarray(layer["b"])
        for i, layer in enumerate(opt_state.v):
            arrays[f"adam_v_W_{i}"] = np.asarray(layer["W"])
            arrays[f"adam_v_b_{i}"] = np.asarray(layer["b"])
    np.savez(path, **arrays)


def load_theta(path: str) -> tuple[list[dict], AdamState | None]:
    """Load MLP weights (and Adam state if present) from an .npz file.

    Returns (theta, opt_state) where opt_state is None if not saved.
    """
    data = np.load(path)
    n_layers = int(data["n_layers"])
    theta = []
    for i in range(n_layers):
        theta.append({
            "W": jnp.array(data[f"W_{i}"]),
            "b": jnp.array(data[f"b_{i}"]),
        })
    opt_state = None
    if "adam_t" in data:
        t = int(data["adam_t"])
        m = []
        v = []
        for i in range(n_layers):
            m.append({
                "W": jnp.array(data[f"adam_m_W_{i}"]),
                "b": jnp.array(data[f"adam_m_b_{i}"]),
            })
            v.append({
                "W": jnp.array(data[f"adam_v_W_{i}"]),
                "b": jnp.array(data[f"adam_v_b_{i}"]),
            })
        opt_state = AdamState(m=m, v=v, t=t)
    return theta, opt_state


# ── Reparameterization ───────────────────────────────────────────────


@dataclass
class NeuralReparam:
    """Wraps an MLP that maps cell coordinates to material parameters.

    Attributes
    ----------
    cell_centers_norm : (n_cells, 2) normalized to [0, 1]
    cell_features : (n_cells, n_features) Fourier features
    C_flat_init : (n_cells, n_C_params) initial stiffness
    rho_init : (n_cells,) initial density
    C_scale : (n_C_params,) per-component std of initial C across cloak cells
    rho_scale : float std of initial rho across cloak cells
    cloak_mask : (n_cells,) bool — which cells are in the cloak
    """

    cell_features: jnp.ndarray
    C_flat_init: jnp.ndarray
    rho_init: jnp.ndarray
    cloak_mask: jnp.ndarray
    output_scale: float = 0.1
    # Physically-valid flat4 decode (see decode()). When False, the legacy
    # unbounded multiplicative residual is used.
    constrained: bool = False
    kappa: float = 0.95
    cap_anisotropy: bool = True
    anisotropy_log_ratio: float = math.log(15.0)  # log(R) for the anisotropy cap
    # When True (with n_C_params=6), decode via a PD Cholesky factor of the
    # anisotropic-Cauchy Voigt 3×3 instead of the flat4 orthotropic decode —
    # keeps C PD while letting the C16/C26 coupling move off a zero init.
    aniso_cauchy: bool = False

    def decode(self, theta: list[dict]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Map MLP weights → (cell_C_flat, cell_rho).

        Two modes, selected by ``constrained``:

        * **Legacy (constrained=False):** a *relative* (multiplicative) residual
          so the correction is proportional to the local initial value:
              C(x,y) = C_init(x,y) * (1 + output_scale * Phi(x,y))
          This handles the orders-of-magnitude variation in C_eff across the
          cloak but places no bound on sign or anisotropy — the MLP can drive
          a stiffness negative or to an unrealizable anisotropy ratio.

        * **Constrained flat4 (constrained=True, n_C_params=4):** a
          physically-valid parameterization of (C11, C22, C66, C12):
              C11,C22,C66 = C_init * exp(output_scale * raw)      # > 0 always
              g = 0.5*log(C11/C22); g_b = logR*tanh(g/logR)       # |C11/C22| in [1/R, R]
              C11,C22 = geomean * exp(±g_b)                        # geomean preserved
              C12 = kappa*tanh(raw) * sqrt(C11*C22)                # det>0, sign-free
          so C11/C22/C66 > 0, the orthotropic block is positive-definite
          (det = C11*C22*(1 - rho_corr^2) > 0 since |kappa*tanh| < 1), and the
          anisotropy ratio is bounded. ``raw≈0`` at init reduces to C_init.

        Non-cloak cells keep their initial (background) values exactly.
        """
        raw = mlp_forward(theta, self.cell_features)  # (n_cells, n_C_params+1)
        n_C = self.C_flat_init.shape[1]
        s = self.output_scale
        mask = self.cloak_mask

        if self.constrained and self.aniso_cauchy and n_C == 6:
            # Anisotropic-Cauchy (flat6cauchy [C11,C22,C66,C12,C16,C26]) PD decode.
            # Factor the Voigt 3×3 M=[[C11,C12,C16],[C12,C22,C26],[C16,C26,C66]]
            # as M = L Lᵀ (Cholesky). raw=0 recovers C_init exactly; C is PD for
            # any L with positive diagonal, and the ADDITIVE off-diagonals let the
            # coupling C16/C26 move away from a zero (on-axis orthotropic) init —
            # which the multiplicative legacy decode cannot do.
            Ci = self.C_flat_init
            C11i, C22i, C66i = Ci[:, 0], Ci[:, 1], Ci[:, 2]
            C12i, C16i, C26i = Ci[:, 3], Ci[:, 4], Ci[:, 5]
            Minit = jnp.stack([
                jnp.stack([C11i, C12i, C16i], axis=-1),
                jnp.stack([C12i, C22i, C26i], axis=-1),
                jnp.stack([C16i, C26i, C66i], axis=-1),
            ], axis=-2)                                  # (n_cells, 3, 3)
            Linit = jnp.linalg.cholesky(Minit)           # lower factor
            d0, d1, d2 = Linit[:, 0, 0], Linit[:, 1, 1], Linit[:, 2, 2]
            o10, o20, o21 = Linit[:, 1, 0], Linit[:, 2, 0], Linit[:, 2, 1]

            # positive diagonal in log-space; additive off-diagonal scaled by the
            # geomean of the connected init diagonals (commensurate step size).
            L00 = d0 * jnp.exp(s * raw[:, 0])
            L11 = d1 * jnp.exp(s * raw[:, 1])
            L22 = d2 * jnp.exp(s * raw[:, 2])
            L10 = o10 + s * raw[:, 3] * jnp.sqrt(d0 * d1)
            L20 = o20 + s * raw[:, 4] * jnp.sqrt(d0 * d2)
            L21 = o21 + s * raw[:, 5] * jnp.sqrt(d1 * d2)

            C11 = L00 * L00
            C22 = L10 * L10 + L11 * L11
            C66 = L20 * L20 + L21 * L21 + L22 * L22
            C12 = L00 * L10
            C16 = L00 * L20
            C26 = L10 * L20 + L11 * L21

            cell_C_c = jnp.stack([C11, C22, C66, C12, C16, C26], axis=-1)
            cell_rho_c = self.rho_init * jnp.exp(s * raw[:, n_C])

            cell_C = jnp.where(mask[:, None], cell_C_c, self.C_flat_init)
            cell_rho = jnp.where(mask, cell_rho_c, self.rho_init)
            return (cell_C, cell_rho)

        if self.constrained and n_C == 4:
            C11i = self.C_flat_init[:, 0]
            C22i = self.C_flat_init[:, 1]
            C66i = self.C_flat_init[:, 2]
            C12i = self.C_flat_init[:, 3]

            # 1) positive diagonal via log-space multiplicative residual
            C11 = C11i * jnp.exp(s * raw[:, 0])
            C22 = C22i * jnp.exp(s * raw[:, 1])
            C66 = C66i * jnp.exp(s * raw[:, 2])

            # 2) anisotropy cap — symmetric squash on the half-log-ratio,
            #    geomean kept. g = 0.5*log(C11/C22) is bounded to [-logR/2, logR/2]
            #    so the ratio C11/C22 = exp(2*g_b) lands in [1/R, R].
            if self.cap_anisotropy:
                half_logR = 0.5 * self.anisotropy_log_ratio
                g = 0.5 * jnp.log(C11 / C22)
                g_b = half_logR * jnp.tanh(g / half_logR)
                geomean = jnp.sqrt(C11 * C22)
                C11 = geomean * jnp.exp(g_b)
                C22 = geomean * jnp.exp(-g_b)

            # 3) C12 — correlation-coefficient coupling (PD det>0, allows C12<0).
            #    Centred on the init correlation so raw=0 recovers C12_init: the
            #    cap preserves geomean=sqrt(C11*C22), so C12(raw=0) = rho0*geomean
            #    = C12_init when |rho0| < kappa. rho0 is clamped just inside the
            #    PD-safe band (an init beyond it is pulled to the boundary).
            rho0 = C12i / jnp.sqrt(C11i * C22i)
            rho0c = jnp.clip(rho0, -self.kappa * (1 - 1e-6), self.kappa * (1 - 1e-6))
            offset = jnp.arctanh(rho0c / self.kappa)
            rho_corr = self.kappa * jnp.tanh(raw[:, 3] + offset)
            C12 = rho_corr * jnp.sqrt(C11 * C22)

            cell_C_c = jnp.stack([C11, C22, C66, C12], axis=-1)
            cell_rho_c = self.rho_init * jnp.exp(s * raw[:, n_C])

            # background cells pass C_flat_init / rho_init through unchanged
            cell_C = jnp.where(mask[:, None], cell_C_c, self.C_flat_init)
            cell_rho = jnp.where(mask, cell_rho_c, self.rho_init)
            return (cell_C, cell_rho)

        # legacy unbounded path (lame n_C=2, or constrained=False)
        rel_C = raw[:, :n_C] * s            # relative correction
        rel_rho = raw[:, n_C] * s

        cell_C = self.C_flat_init * (1.0 + rel_C * mask[:, None])
        cell_rho = self.rho_init * jnp.maximum(1.0 + rel_rho * mask, 1e-6)

        return (cell_C, cell_rho)


def make_neural_reparam(
    cell_decomp: CellDecomposition,
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    hidden_size: int = 256,
    n_layers: int = 4,
    n_fourier: int = 32,
    seed: int = 42,
    output_scale: float = 0.1,
    constrained: bool = False,
    kappa: float = 0.95,
    cap_anisotropy: bool = True,
    anisotropy_ratio: float = 15.0,
    aniso_cauchy: bool = False,
) -> tuple[list[dict], NeuralReparam]:
    """Create a NeuralReparam and initialize the MLP weights.

    Returns
    -------
    theta : MLP parameters (list of {W, b})
    reparam : NeuralReparam instance with decode() method
    """
    cell_C_init, cell_rho_init = params_init
    n_cells, n_C_params = cell_C_init.shape
    n_out = n_C_params + 1  # C_flat components + 1 rho

    # Normalize cell centers to [0, 1] for better conditioning
    centers = jnp.array(cell_decomp.cell_centers)
    lo = centers.min(axis=0)
    hi = centers.max(axis=0)
    centers_norm = (centers - lo) / (hi - lo + 1e-10)

    features = fourier_features(centers_norm, n_fourier)
    n_features = features.shape[1]

    mask = jnp.array(cell_decomp.cloak_mask)

    # MLP: features → hidden → ... → hidden → n_out
    layer_sizes = [n_features] + [hidden_size] * (n_layers - 1) + [n_out]
    key = jax.random.PRNGKey(seed)
    theta = init_mlp(key, layer_sizes)

    # Scale down the last layer so initial MLP output ≈ 0 (start near init params)
    theta[-1]["W"] = theta[-1]["W"] * 0.01
    theta[-1]["b"] = theta[-1]["b"] * 0.0

    reparam = NeuralReparam(
        cell_features=features,
        C_flat_init=cell_C_init,
        rho_init=cell_rho_init,
        cloak_mask=mask,
        output_scale=output_scale,
        constrained=constrained,
        kappa=kappa,
        cap_anisotropy=cap_anisotropy,
        anisotropy_log_ratio=math.log(anisotropy_ratio),
        aniso_cauchy=aniso_cauchy,
    )

    return theta, reparam


# ── Optimization loop ────────────────────────────────────────────────


@dataclass
class NeuralOptimizationResult:
    """Result of neural-reparameterized optimization."""
    theta: list[dict]
    best_theta: list[dict]  # weights at lowest total loss
    opt_state: AdamState     # final Adam state (for warm restart)
    params: tuple[jnp.ndarray, jnp.ndarray]  # final decoded (cell_C, cell_rho)
    loss_history: list[float] = field(default_factory=list)
    cloak_history: list[float] = field(default_factory=list)
    l2_history: list[float] = field(default_factory=list)
    transmission_history: list[float] = field(default_factory=list)


def run_optimization_neural(
    fwd_pred,
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    u_ref_boundary: jnp.ndarray,
    boundary_indices: jnp.ndarray,
    reparam: NeuralReparam,
    theta_init: list[dict],
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "linear",
    lambda_l2: float = 1e-4,
    plot_callback=None,
    plot_every: int = 1,
    step_callback=None,
    opt_state_init: AdamState | None = None,
    loss_fn=None,
    gmm_prior: GMMPrior | None = None,
    lambda_gmm: float = 0.0,
    n_C_params: int = 2,
    u_ref_full: np.ndarray | None = None,
    trans_surface_case: np.ndarray | None = None,
    trans_surface_ref: np.ndarray | None = None,
) -> NeuralOptimizationResult:
    """Run optimization over MLP weights (neural reparameterization).

    Parameters
    ----------
    fwd_pred : callable
        Differentiable forward prediction from ``ad_wrapper``.
    params_init : (cell_C_flat, cell_rho) — original initial material values
    reparam : NeuralReparam with decode()
    theta_init : initial MLP weights
    opt_state_init : if provided, resume Adam from this state (warm restart)
    step_callback : optional callable(step, total, cloak, l2, neighbor, params)
        Same signature as raw optimization for compatibility, plus a
        ``transmission_ratio`` keyword when the metric is available.
    loss_fn : optional callable(u_cloak, u_ref, indices) -> scalar
        JAX-traceable cloaking loss. Defaults to ``cloaking_loss`` (relative L2).
    u_ref_full, trans_surface_case, trans_surface_ref : optional
        Reference displacement field and the (case, reference) node indices of
        the free surface beyond the cloak. When all three are given, the
        transmitted-displacement ratio (``→1`` is a perfect cloak) is evaluated
        every step — for free, from the field the gradient solve already
        produces — then printed and forwarded to ``step_callback``.
    """
    if loss_fn is None:
        loss_fn = cloaking_loss
    _cloak_loss_fn = loss_fn

    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = opt_state_init if opt_state_init is not None else adam_init(theta)
    loss_history: list[float] = []
    cloak_history: list[float] = []
    l2_history: list[float] = []
    transmission_history: list[float] = []

    best_loss = float("inf")
    best_theta = jax.tree.map(jnp.copy, theta)

    boundary_indices_jnp = jnp.array(boundary_indices)

    use_gmm = gmm_prior is not None and lambda_gmm > 0.0

    def _loss_fn(theta):
        params = reparam.decode(theta)
        sol_list = fwd_pred(params)
        u_cloak = sol_list[0]
        L_cloak = _cloak_loss_fn(u_cloak, u_ref_boundary, boundary_indices_jnp)
        L_l2 = l2_regularization(params, params_init)
        total = L_cloak + lambda_l2 * L_l2
        if use_gmm:
            cell_C, cell_rho = params
            L_gmm = gmm_flat_top_penalty(
                cell_C, cell_rho, reparam.cloak_mask, gmm_prior, n_C_params
            )
            total = total + lambda_gmm * L_gmm
        # Return the forward field as aux so the transmission-ratio metric can
        # be evaluated each step without a second solve (the gradient is taken
        # w.r.t. ``total`` only — aux outputs do not enter the derivative).
        return total, u_cloak

    loss_and_grad = jax.value_and_grad(_loss_fn, has_aux=True)

    track_transmission = (
        u_ref_full is not None
        and trans_surface_case is not None
        and trans_surface_ref is not None
    )

    # Separate function to get cloak loss (no extra cost — we decompose from total)
    def _get_components(theta):
        params = reparam.decode(theta)
        L_l2 = float(l2_regularization(params, params_init))
        L_gmm = 0.0
        if use_gmm:
            cell_C, cell_rho = params
            L_gmm = float(gmm_flat_top_penalty(
                cell_C, cell_rho, reparam.cloak_mask, gmm_prior, n_C_params
            ))
        return L_l2, L_gmm

    for step in range(n_iters):
        # Learning rate schedule
        t_frac = step / max(n_iters - 1, 1)
        if lr_end is None:
            cur_lr = lr
        elif lr_schedule == "cosine":
            cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * t_frac))
        else:
            cur_lr = lr + (lr_end - lr) * t_frac

        (loss_val, u_cloak), grads = loss_and_grad(theta)
        loss_val_float = float(loss_val)
        loss_history.append(loss_val_float)

        L_l2, L_gmm = _get_components(theta)
        L_cloak = loss_val_float - lambda_l2 * L_l2 - lambda_gmm * L_gmm
        cloak_history.append(L_cloak)
        l2_history.append(L_l2)

        # Transmission ratio (independent quality metric, →1 is perfect) from
        # the field the gradient solve already produced — no extra solve.
        transmission_ratio = None
        if track_transmission:
            transmission_ratio = transmitted_displacement_ratio(
                np.asarray(u_cloak), u_ref_full,
                trans_surface_case, trans_surface_ref,
            )
            transmission_history.append(transmission_ratio)

        grad_norm = float(jnp.sqrt(sum(
            jnp.sum(l["W"]**2) + jnp.sum(l["b"]**2) for l in grads
        )))
        gmm_str = f"  GMM = {L_gmm:.4e}" if use_gmm else ""
        trans_str = (f"  trans_ratio = {transmission_ratio:.4f}"
                     if transmission_ratio is not None else "")
        print(
            f"  Step {step:4d} | total = {loss_val_float:.4e}"
            f"  cloak_pct = {np.sqrt(max(L_cloak, 0)) * 100:.2f}"
            f"{trans_str}"
            f"  L2 = {L_l2:.4e}"
            f"{gmm_str}"
            f"  lr={cur_lr:.2e}"
            f"  |grad|={grad_norm:.4e}"
        )

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_theta = jax.tree.map(jnp.copy, theta)

        if step_callback is not None:
            params = reparam.decode(theta)
            step_callback(step, loss_val_float, L_cloak, L_l2, 0.0, params,
                          theta=theta, opt_state=opt_state,
                          transmission_ratio=transmission_ratio)

        if plot_callback is not None and step % plot_every == 0:
            params = reparam.decode(theta)
            sol_list = fwd_pred(params)
            plot_callback(step, np.asarray(sol_list[0]))

        updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
        theta = jax.tree.map(lambda p, u: p + u, theta, updates)

    # Final state
    final_params = reparam.decode(theta)

    if plot_callback is not None:
        sol_list = fwd_pred(final_params)
        plot_callback(n_iters, np.asarray(sol_list[0]))

    return NeuralOptimizationResult(
        theta=theta,
        best_theta=best_theta,
        opt_state=opt_state,
        params=final_params,
        loss_history=loss_history,
        cloak_history=cloak_history,
        l2_history=l2_history,
        transmission_history=transmission_history,
    )


# ── Multi-frequency optimization ────────────────────────────────────


@dataclass
class FreqTarget:
    """Per-frequency data needed for loss evaluation.

    Built once in the solver setup and threaded into the optimization loop.
    """
    f_star: float
    weight: float
    fwd_pred: Any            # ad_wrapper callable
    u_ref_boundary: Any      # jnp.ndarray  (n_boundary, 4)
    boundary_indices: Any    # jnp.ndarray
    loss_fn: Any             # callable(u_cloak, u_ref, indices) -> scalar


def run_optimization_neural_multifreq(
    freq_targets: list[FreqTarget],
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    reparam: NeuralReparam,
    theta_init: list[dict],
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "linear",
    lambda_l2: float = 1e-4,
    plot_callback=None,
    plot_every: int = 1,
    step_callback=None,
    opt_state_init: AdamState | None = None,
    max_workers: int = 0,
    gmm_prior: GMMPrior | None = None,
    lambda_gmm: float = 0.0,
    n_C_params: int = 2,
    max_norm: float = 1.0,
) -> NeuralOptimizationResult:
    """Run neural-reparam optimization over multiple frequencies in parallel.

    Each frequency's loss+gradient is computed independently via
    ``jax.value_and_grad``, then the weighted contributions are summed.
    The per-frequency forward and adjoint PETSc solves run concurrently
    in a thread pool (PETSc releases the GIL).

    Parameters
    ----------
    freq_targets : list of FreqTarget
        One entry per frequency, each carrying its own ``fwd_pred``,
        reference data, and loss function.
    max_workers : int
        Thread-pool size.  0 (default) → ``len(freq_targets)``.
    max_norm : float
        Global-norm clipping threshold for the combined gradient.  The
        gradient is rescaled so its global L2 norm never exceeds this value.
    """
    n_freq = len(freq_targets)
    if max_workers <= 0:
        max_workers = n_freq

    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = opt_state_init if opt_state_init is not None else adam_init(theta)
    loss_history: list[float] = []
    cloak_history: list[float] = []
    l2_history: list[float] = []

    best_loss = float("inf")
    best_theta = jax.tree.map(jnp.copy, theta)

    # Pre-convert indices to jnp once
    for ft in freq_targets:
        ft.boundary_indices = jnp.array(ft.boundary_indices)

    # Build a per-frequency value_and_grad closure.
    # Each closure captures its own fwd_pred / u_ref / indices / loss_fn,
    # but reads `reparam` and `params_init` from the enclosing scope.
    def _make_freq_loss_and_grad(ft: FreqTarget):
        _fwd = ft.fwd_pred
        _u_ref = ft.u_ref_boundary
        _idx = ft.boundary_indices
        _lfn = ft.loss_fn
        _w = ft.weight

        def _loss(theta):
            params = reparam.decode(theta)
            sol_list = _fwd(params)
            return _w * _lfn(sol_list[0], _u_ref, _idx)

        return jax.value_and_grad(_loss)

    freq_loss_and_grads = [_make_freq_loss_and_grad(ft) for ft in freq_targets]

    # Use the first frequency's fwd_pred for plotting
    primary_fwd_pred = freq_targets[0].fwd_pred

    f_star_str = ", ".join(f"{ft.f_star:.2f}(w={ft.weight:.2f})"
                           for ft in freq_targets)
    print(f"  Multi-freq optimization: {n_freq} frequencies [{f_star_str}]")
    print(f"  Thread pool: {max_workers} workers")

    pool = ThreadPoolExecutor(max_workers=max_workers)

    try:
        for step in range(n_iters):
            # Learning rate schedule
            t_frac = step / max(n_iters - 1, 1)
            if lr_end is None:
                cur_lr = lr
            elif lr_schedule == "cosine":
                cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * t_frac))
            else:
                cur_lr = lr + (lr_end - lr) * t_frac

            # Dispatch per-frequency loss+grad in parallel
            futures = [pool.submit(fn, theta) for fn in freq_loss_and_grads]
            results = [f.result() for f in futures]

            # Sum losses and gradients across frequencies
            total_cloak_loss = sum(float(r[0]) for r in results)
            total_grad = jax.tree.map(
                lambda *gs: sum(gs),
                *(r[1] for r in results),
            )

            # L2 + GMM regularisation (frequency-independent, compute once)
            params = reparam.decode(theta)
            L_l2 = float(l2_regularization(params, params_init))

            use_gmm = gmm_prior is not None and lambda_gmm > 0.0
            if use_gmm:
                cell_C, cell_rho = params
                L_gmm = float(gmm_flat_top_penalty(
                    cell_C, cell_rho, reparam.cloak_mask, gmm_prior, n_C_params
                ))
            else:
                L_gmm = 0.0

            def _reg_total(t):
                p = reparam.decode(t)
                term = lambda_l2 * l2_regularization(p, params_init)
                if use_gmm:
                    cC, cr = p
                    term = term + lambda_gmm * gmm_flat_top_penalty(
                        cC, cr, reparam.cloak_mask, gmm_prior, n_C_params
                    )
                return term

            reg_grad = jax.grad(_reg_total)(theta)
            grads = jax.tree.map(lambda a, b: a + b, total_grad, reg_grad)

            # Global-norm gradient clipping (before the Adam update).
            gn = jnp.sqrt(sum(
                jnp.sum(l["W"]**2) + jnp.sum(l["b"]**2) for l in grads
            ))
            scale = jnp.minimum(1.0, max_norm / (gn + 1e-12))
            grads = jax.tree.map(lambda g: g * scale, grads)

            loss_val_float = total_cloak_loss + lambda_l2 * L_l2 + lambda_gmm * L_gmm
            loss_history.append(loss_val_float)
            cloak_history.append(total_cloak_loss)
            l2_history.append(L_l2)

            # Report the pre-clip global norm so clipping is observable.
            grad_norm = float(gn)

            # Per-frequency breakdown
            per_freq = "  ".join(
                f"f*={ft.f_star:.1f}:{float(r[0]):.4e}"
                for ft, r in zip(freq_targets, results)
            )
            gmm_str = f"  GMM = {L_gmm:.4e}" if use_gmm else ""
            print(
                f"  Step {step:4d} | total = {loss_val_float:.4e}"
                f"  cloak = {total_cloak_loss:.4e}"
                f"  L2 = {L_l2:.4e}"
                f"{gmm_str}"
                f"  lr={cur_lr:.2e}"
                f"  |grad|={grad_norm:.4e}"
                f"\n    {per_freq}"
            )

            if loss_val_float < best_loss:
                best_loss = loss_val_float
                best_theta = jax.tree.map(jnp.copy, theta)

            if step_callback is not None:
                step_callback(step, loss_val_float, total_cloak_loss, L_l2, 0.0, params, theta=theta, opt_state=opt_state)

            if plot_callback is not None and step % plot_every == 0:
                sol_list = primary_fwd_pred(params)
                plot_callback(step, np.asarray(sol_list[0]))

            updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
            theta = jax.tree.map(lambda p, u: p + u, theta, updates)
    finally:
        pool.shutdown(wait=False)

    # Final state
    final_params = reparam.decode(theta)

    if plot_callback is not None:
        sol_list = primary_fwd_pred(final_params)
        plot_callback(n_iters, np.asarray(sol_list[0]))

    return NeuralOptimizationResult(
        theta=theta,
        best_theta=best_theta,
        opt_state=opt_state,
        params=final_params,
        loss_history=loss_history,
        cloak_history=cloak_history,
        l2_history=l2_history,
    )


# ── Minimax multi-frequency optimization ────────────────────────────


def run_optimization_neural_multifreq_minimax(
    freq_targets: list[FreqTarget],
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    reparam: NeuralReparam,
    theta_init: list[dict],
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "linear",
    lambda_l2: float = 1e-4,
    plot_callback=None,
    plot_every: int = 1,
    step_callback=None,
    opt_state_init: AdamState | None = None,
    max_workers: int = 0,
    gmm_prior: GMMPrior | None = None,
    lambda_gmm: float = 0.0,
    n_C_params: int = 2,
) -> NeuralOptimizationResult:
    """Minimax multi-frequency optimization: only the worst-case frequency
    contributes to each gradient step.

    All frequencies are evaluated in parallel at each iteration. The
    frequency with the largest loss is selected, and only its gradient
    is used for the parameter update. This drives down the worst-case
    error across the frequency band.

    The loss dictionary (per-frequency losses) is tracked and printed
    at each iteration for monitoring.
    """
    n_freq = len(freq_targets)
    if max_workers <= 0:
        max_workers = n_freq

    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = opt_state_init if opt_state_init is not None else adam_init(theta)
    loss_history: list[float] = []
    cloak_history: list[float] = []
    l2_history: list[float] = []

    best_loss = float("inf")
    best_theta = jax.tree.map(jnp.copy, theta)

    # Pre-convert indices to jnp once
    for ft in freq_targets:
        ft.boundary_indices = jnp.array(ft.boundary_indices)

    # Build per-frequency value_and_grad closures (no weight scaling for minimax)
    def _make_freq_loss_and_grad(ft: FreqTarget):
        _fwd = ft.fwd_pred
        _u_ref = ft.u_ref_boundary
        _idx = ft.boundary_indices
        _lfn = ft.loss_fn

        def _loss(theta):
            params = reparam.decode(theta)
            sol_list = _fwd(params)
            return _lfn(sol_list[0], _u_ref, _idx)

        return jax.value_and_grad(_loss)

    freq_loss_and_grads = [_make_freq_loss_and_grad(ft) for ft in freq_targets]

    # Use the first frequency's fwd_pred for plotting
    primary_fwd_pred = freq_targets[0].fwd_pred

    f_star_str = ", ".join(f"{ft.f_star:.2f}" for ft in freq_targets)
    print(f"  Minimax optimization: {n_freq} frequencies [{f_star_str}]")
    print(f"  Thread pool: {max_workers} workers")

    pool = ThreadPoolExecutor(max_workers=max_workers)

    try:
        for step in range(n_iters):
            # Learning rate schedule
            t_frac = step / max(n_iters - 1, 1)
            if lr_end is None:
                cur_lr = lr
            elif lr_schedule == "cosine":
                cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * t_frac))
            else:
                cur_lr = lr + (lr_end - lr) * t_frac

            # Dispatch per-frequency loss+grad in parallel
            futures = [pool.submit(fn, theta) for fn in freq_loss_and_grads]
            results = [f.result() for f in futures]

            # Collect per-frequency losses
            per_freq_losses = [float(r[0]) for r in results]

            # Select worst-case frequency
            worst_idx = int(np.argmax(per_freq_losses))
            worst_f_star = freq_targets[worst_idx].f_star
            worst_loss = per_freq_losses[worst_idx]
            worst_grad = results[worst_idx][1]

            # L2 + GMM regularisation
            params = reparam.decode(theta)
            L_l2 = float(l2_regularization(params, params_init))

            use_gmm = gmm_prior is not None and lambda_gmm > 0.0
            if use_gmm:
                cell_C, cell_rho = params
                L_gmm = float(gmm_flat_top_penalty(
                    cell_C, cell_rho, reparam.cloak_mask, gmm_prior, n_C_params
                ))
            else:
                L_gmm = 0.0

            def _reg_total(t):
                p = reparam.decode(t)
                term = lambda_l2 * l2_regularization(p, params_init)
                if use_gmm:
                    cC, cr = p
                    term = term + lambda_gmm * gmm_flat_top_penalty(
                        cC, cr, reparam.cloak_mask, gmm_prior, n_C_params
                    )
                return term

            reg_grad = jax.grad(_reg_total)(theta)
            grads = jax.tree.map(lambda a, b: a + b, worst_grad, reg_grad)

            loss_val_float = worst_loss + lambda_l2 * L_l2 + lambda_gmm * L_gmm
            loss_history.append(loss_val_float)
            cloak_history.append(worst_loss)
            l2_history.append(L_l2)

            grad_norm = float(jnp.sqrt(sum(
                jnp.sum(l["W"]**2) + jnp.sum(l["b"]**2) for l in grads
            )))

            # Per-frequency breakdown (* marks the worst)
            per_freq_str = "  ".join(
                f"f*={ft.f_star:.2f}:{pfl:.4e}{'*' if i == worst_idx else ''}"
                for i, (ft, pfl) in enumerate(zip(freq_targets, per_freq_losses))
            )
            gmm_str = f"  GMM = {L_gmm:.4e}" if use_gmm else ""
            print(
                f"  Step {step:4d} | worst = {worst_loss:.4e} (f*={worst_f_star:.2f})"
                f"  L2 = {L_l2:.4e}"
                f"{gmm_str}"
                f"  lr={cur_lr:.2e}"
                f"  |grad|={grad_norm:.4e}"
                f"\n    {per_freq_str}"
            )

            if loss_val_float < best_loss:
                best_loss = loss_val_float
                best_theta = jax.tree.map(jnp.copy, theta)

            if step_callback is not None:
                step_callback(step, loss_val_float, worst_loss, L_l2, 0.0, params, theta=theta, opt_state=opt_state)

            if plot_callback is not None and step % plot_every == 0:
                sol_list = primary_fwd_pred(params)
                plot_callback(step, np.asarray(sol_list[0]))

            updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
            theta = jax.tree.map(lambda p, u: p + u, theta, updates)
    finally:
        pool.shutdown(wait=False)

    # Final state
    final_params = reparam.decode(theta)

    if plot_callback is not None:
        sol_list = primary_fwd_pred(final_params)
        plot_callback(n_iters, np.asarray(sol_list[0]))

    return NeuralOptimizationResult(
        theta=theta,
        best_theta=best_theta,
        opt_state=opt_state,
        params=final_params,
        loss_history=loss_history,
        cloak_history=cloak_history,
        l2_history=l2_history,
    )
