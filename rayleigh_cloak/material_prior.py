"""Dataset-based material prior for the cloaking optimisation.

Loads a Gaussian-mixture .npz produced by ``dataset.cellular_chiral.fit_gmm``
and provides a JAX-traceable flat-top penalty:

    penalty(C, rho) = mean over cloak cells of  max(0, threshold - log p(features))

where the feature vector is (λ, μ, ρ) for the ``lame`` prior (n_C_params=2) or
(C₁₁₁₁, C₂₂₂₂, C₁₂₁₂, C₁₁₂₂, ρ) for the ``flat4`` prior (n_C_params=4).

The penalty is zero whenever the cell sits comfortably inside the dataset's
density support (log p ≥ threshold) and grows linearly as the cell drifts
into the tails. See ``fit_gmm.py`` for how ``threshold`` is set (a quantile
of the dataset's own log p, default the 25th percentile so the borders are
already penalised).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GMMPrior:
    """Pre-loaded GMM, ready for use in a JAX loss.

    All arrays live on the JAX device. The Cholesky factor of the *precision*
    (i.e. inverse covariance) is stored directly — it's what sklearn already
    computes during fit, and it lets us evaluate log-prob without an inverse.

    The feature dimension ``D`` depends on the cell representation the prior was
    fit for: ``D=3`` for the ``lame`` set (λ, μ, ρ; ``n_C_params=2``) and
    ``D=5`` for the ``flat4`` set (C₁₁₁₁, C₂₂₂₂, C₁₂₁₂, C₁₁₂₂, ρ;
    ``n_C_params=4``). ``n_C_params`` and ``feature_order`` are static Python
    objects (not device arrays) so they can guard against a prior/cell mismatch.

    Attributes
    ----------
    weights              : (K,)              mixture weights
    means                : (K, D)            in standardised feature space
    precisions_cholesky  : (K, D, D)         Cholesky of inverse covariance
    feature_mean         : (D,)              standardisation mean (raw → std)
    feature_std          : (D,)              standardisation std
    threshold            : scalar            flat-top threshold τ
    n_C_params           : int               cell representation this prior fits
    feature_order        : tuple[str, ...]   feature names, in column order
    """
    weights: jnp.ndarray
    means: jnp.ndarray
    precisions_cholesky: jnp.ndarray
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    threshold: jnp.ndarray
    n_C_params: int
    feature_order: tuple[str, ...]


# Positions of dataset log-p quantiles saved by ``fit_gmm.py`` in
# ``log_p_quantiles``. Kept in sync with ``dataset.cellular_chiral.fit_gmm``.
_LOG_P_QUANTILE_POSITIONS = (0.01, 0.05, 0.25, 0.50, 0.75)


def load_gmm_prior(
    path: str | Path,
    dtype=jnp.float32,
    quantile: float | None = None,
) -> GMMPrior:
    """Load a .npz produced by ``fit_gmm.py`` into a :class:`GMMPrior`.

    Pass ``quantile`` ∈ [0.01, 0.75] to override the flat-top threshold τ:
    the dataset log-p quantile at that position becomes τ, so cells whose
    log p falls below it get penalised. Lower quantile → looser margin
    (only deep outliers penalised); higher → stricter (push toward the
    densest part of the manifold). ``None`` keeps the τ baked in at fit
    time. The GMM density itself is unchanged, so no refit is needed.
    """
    data = np.load(str(path), allow_pickle=True)
    cov_type = str(data["covariance_type"])
    if cov_type != "full":
        raise NotImplementedError(
            f"GMMPrior currently supports covariance_type='full' only; "
            f"got {cov_type!r}. Refit with `--covariance-type=full`."
        )
    if quantile is None:
        tau = float(data["threshold"])
    else:
        q_lo, q_hi = _LOG_P_QUANTILE_POSITIONS[0], _LOG_P_QUANTILE_POSITIONS[-1]
        if not (q_lo <= quantile <= q_hi):
            raise ValueError(
                f"quantile must be in [{q_lo}, {q_hi}] (the range of dataset "
                f"log-p quantiles stored in the .npz); got {quantile}"
            )
        tau = float(np.interp(
            quantile, _LOG_P_QUANTILE_POSITIONS, np.asarray(data["log_p_quantiles"]),
        ))
    feature_order = tuple(str(s) for s in np.asarray(data["feature_order"]))
    # ``n_C_params`` was added alongside the flat4 feature set; older .npz files
    # (λ, μ, ρ) predate it, so fall back to the isotropic flat2 representation.
    if "n_C_params" in data.files:
        n_C_params = int(data["n_C_params"])
    elif feature_order == ("lambda", "mu", "rho"):
        n_C_params = 2
    else:
        raise ValueError(
            f"{path}: missing 'n_C_params' and unrecognised feature_order "
            f"{feature_order}; refit with the current fit_gmm.py."
        )
    return GMMPrior(
        weights=jnp.asarray(data["weights"], dtype=dtype),
        means=jnp.asarray(data["means"], dtype=dtype),
        precisions_cholesky=jnp.asarray(data["precisions_cholesky"], dtype=dtype),
        feature_mean=jnp.asarray(data["feature_mean"], dtype=dtype),
        feature_std=jnp.asarray(data["feature_std"], dtype=dtype),
        threshold=jnp.asarray(tau, dtype=dtype),
        n_C_params=n_C_params,
        feature_order=feature_order,
    )


# ── log-prob and penalty ────────────────────────────────────────────


def _gmm_log_prob_standardised(x_std: jnp.ndarray, prior: GMMPrior) -> jnp.ndarray:
    """log p(x) for x already in standardised space.

    Uses sklearn's convention where ``precisions_cholesky`` is L such that
    Σ⁻¹ = L L^T, i.e. y = (x - μ) @ L gives the Mahalanobis vector with
    ‖y‖² = (x - μ)^T Σ⁻¹ (x - μ).

    Parameters
    ----------
    x_std : (..., 3) standardised features
    prior : GMMPrior
    """
    d = prior.means.shape[-1]
    diff = x_std[..., None, :] - prior.means              # (..., K, d)
    y = jnp.einsum("...kd,kde->...ke", diff, prior.precisions_cholesky)  # (..., K, d)
    mahal_sq = jnp.sum(y * y, axis=-1)                    # (..., K)
    log_det_pc = jnp.sum(
        jnp.log(jnp.diagonal(prior.precisions_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )                                                     # (K,)
    log_per_k = -0.5 * d * jnp.log(2.0 * jnp.pi) + log_det_pc - 0.5 * mahal_sq
    log_w = jnp.log(prior.weights)
    return jax.scipy.special.logsumexp(log_w + log_per_k, axis=-1)  # (...)


def _cell_features(cell_C_flat: jnp.ndarray, cell_rho: jnp.ndarray) -> jnp.ndarray:
    """Stack per-cell stiffness and density into the GMM feature matrix.

    The flat stiffness layout already matches the GMM's stiffness column order,
    so the features are simply ``[*cell_C_flat, rho]``:

      * ``n_C_params == 2`` (lame, D=3): [λ, μ, ρ] — flat2 is [λ, μ].
      * ``n_C_params == 4`` (flat4, D=5): [C₁₁₁₁, C₂₂₂₂, C₁₂₁₂, C₁₁₂₂, ρ] —
        flat4 is [C₁₁₁₁, C₂₂₂₂, C₁₂₁₂, C₁₁₂₂] (see ``materials.C_to_flatC``).

    Returns ``(n_cells, n_C_params + 1)`` in raw (un-standardised) space.
    """
    return jnp.concatenate([cell_C_flat, cell_rho[..., None]], axis=-1)


def gmm_flat_top_penalty(
    cell_C_flat: jnp.ndarray,
    cell_rho: jnp.ndarray,
    cloak_mask: jnp.ndarray,
    prior: GMMPrior,
    n_C_params: int,
) -> jnp.ndarray:
    """Mean of ``max(0, τ - log p(λ, μ, ρ))`` over cloak cells.

    Parameters
    ----------
    cell_C_flat : (n_cells, n_C_params)  per-cell stiffness in flat form
    cell_rho    : (n_cells,)             per-cell density
    cloak_mask  : (n_cells,) bool        which cells are inside the cloak
    prior       : GMMPrior
    n_C_params  : layout of cell_C_flat

    Returns
    -------
    scalar JAX value — the regularisation term to be multiplied by ``weight``.

    Raises
    ------
    ValueError
        If the prior's cell representation (``prior.n_C_params``) does not match
        the ``n_C_params`` of the cells being optimised, or if the prior's
        feature dimension is not ``n_C_params + 1`` (stiffness columns + ρ).
    """
    if prior.n_C_params != n_C_params:
        raise ValueError(
            f"material prior was fit for n_C_params={prior.n_C_params} "
            f"(features {list(prior.feature_order)}) but the cells use "
            f"n_C_params={n_C_params}. Refit the GMM with the matching "
            f"--feature-set, or point the config at the right prior .npz."
        )
    prior_dim = int(prior.means.shape[-1])
    if prior_dim != n_C_params + 1:
        raise ValueError(
            f"material prior feature dim ({prior_dim}) must equal "
            f"n_C_params + 1 = {n_C_params + 1} (stiffness columns + rho); "
            f"feature_order={list(prior.feature_order)}."
        )
    feat = _cell_features(cell_C_flat, cell_rho)           # (n_cells, D)
    feat_std = (feat - prior.feature_mean) / prior.feature_std
    log_p = _gmm_log_prob_standardised(feat_std, prior)   # (n_cells,)
    pen = jnp.maximum(0.0, prior.threshold - log_p)       # flat-top
    mask = cloak_mask.astype(pen.dtype)
    # Mean over cloak cells (avoid division by zero on the empty-mask edge).
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(pen * mask) / denom
