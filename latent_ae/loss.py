"""Loss components for the frequency-aware autoencoder.

Total = λ_rec·L_rec + λ_perf·L_perf + λ_rank·L_rank + λ_freq·L_freq_reg
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from surrogate.dataset import SurrogateBatch

from latent_ae.model import EncodeOutput


def reconstruction_loss(
    out: EncodeOutput,
    batch: SurrogateBatch,
    model,
    rho_weight: float = 1.0,
    cloak_mask: Tensor | None = None,
) -> Tensor:
    """MSE on (C, rho) in normalized units.

    The decoder emits normalized outputs (C_mean / C_std scaling), so the loss
    must normalize the targets the same way. Restricted to cloak cells when
    ``cloak_mask`` is given.

    Shapes: batch.C (B, X, Y, P), batch.rho (B, X, Y), mask (X, Y) 0/1.
    """
    C_tgt, rho_tgt = model.normalize(batch.C, batch.rho)
    # Non-cloak cells are zero in both the target (cloak_only=True dataset) and
    # the prediction (decoder masking), so the zeros contribute nothing either
    # way — we still mask out the non-cloak contribution to keep the mean
    # focused on cloak cells.
    if cloak_mask is None:
        mse_C = F.mse_loss(out.C_recon, C_tgt)
        mse_rho = F.mse_loss(out.rho_recon, rho_tgt)
    else:
        mask = cloak_mask.to(out.C_recon)          # (X, Y)
        n_active = mask.sum().clamp_min(1.0)
        sq_C = (out.C_recon - C_tgt).pow(2).sum(dim=-1)  # (B, X, Y)
        mse_C = (sq_C * mask).sum() / (out.C_recon.shape[0] * n_active * batch.C.shape[-1])
        sq_rho = (out.rho_recon - rho_tgt).pow(2)
        mse_rho = (sq_rho * mask).sum() / (out.rho_recon.shape[0] * n_active)
    return mse_C + rho_weight * mse_rho


def performance_loss(out: EncodeOutput, batch: SurrogateBatch) -> Tensor:
    """MSE on the scalar loss target. Target is already log_clip-transformed upstream."""
    return F.mse_loss(out.loss_pred, batch.loss)


def ranking_loss(
    z_sh: Tensor,
    f_star: Tensor,
    loss_target: Tensor,
    f_min: float,
    f_max: float,
    n_freq_bins: int = 8,
    rank_topk: int = 8,
    rank_margin: float = 0.5,
    rank_gap: float = 0.25,
) -> Tensor:
    """Within-frequency pairwise margin ranking on z_sh.

    For each pair (i, j) in the same f_star bin with |loss_i - loss_j| > rank_gap,
    if loss_i < loss_j, penalize hinge(margin + d_i - d_j), where
        d_k = ||z_sh_k - z_sh_anchor||²
    and z_sh_anchor = stop-grad mean of the rank_topk lowest-loss z_sh in batch.

    Returns the mean hinge penalty over valid pairs (zero if none).
    """
    B = z_sh.shape[0]
    if B < 2:
        return z_sh.sum() * 0.0

    # Anchor: mean of top-k lowest-loss z_sh, detached.
    k = min(rank_topk, B)
    topk_idx = torch.topk(loss_target, k=k, largest=False).indices
    z_anchor = z_sh[topk_idx].mean(dim=0).detach()   # (D,)

    d = (z_sh - z_anchor).pow(2).sum(dim=-1)         # (B,)

    # Bin by f_star
    bin_edges = torch.linspace(f_min, f_max, n_freq_bins + 1, device=z_sh.device)
    f_clamped = f_star.clamp(bin_edges[0], bin_edges[-1] - 1e-9)
    bin_idx = torch.bucketize(f_clamped, bin_edges) - 1
    bin_idx = bin_idx.clamp(0, n_freq_bins - 1)      # (B,)

    # Pair mask: same bin, loss difference exceeds rank_gap, i!=j
    same_bin = bin_idx.unsqueeze(1) == bin_idx.unsqueeze(0)       # (B, B)
    loss_diff = loss_target.unsqueeze(1) - loss_target.unsqueeze(0)
    order_ij = loss_diff < -rank_gap                              # loss_i < loss_j − gap
    pair_mask = same_bin & order_ij
    pair_mask.fill_diagonal_(False)

    if not pair_mask.any():
        return z_sh.sum() * 0.0

    d_i = d.unsqueeze(1).expand(B, B)
    d_j = d.unsqueeze(0).expand(B, B)
    hinge = F.relu(rank_margin + d_i - d_j)                       # (B, B)
    return (hinge * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)


def freq_decorrelation_loss(
    z_sh: Tensor,
    f_star: Tensor,
    fourier_module,
) -> Tensor:
    """Covariance penalty: encourage z_sh to be uncorrelated with Fourier(f_star).

    Fourier(f_star) has shape (B, FF). We compute the cross-covariance matrix
    (B-normalized) between z_sh and Fourier(f_star) and take its squared Frobenius
    norm. Small weight (~0.05) is enough.
    """
    B = z_sh.shape[0]
    if B < 2:
        return z_sh.sum() * 0.0
    ff = fourier_module(f_star)                                   # (B, FF)
    z_c = z_sh - z_sh.mean(dim=0, keepdim=True)
    f_c = ff - ff.mean(dim=0, keepdim=True)
    cov = (z_c.T @ f_c) / (B - 1)                                 # (D, FF)
    return cov.pow(2).mean()
