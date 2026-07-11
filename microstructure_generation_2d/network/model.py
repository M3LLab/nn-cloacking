"""2D conditional diffusion (continuous-time, v-DDIM-like).

Adapted from microstructure_generation_3d/network/model.py. Only the two
sampling modes we actually need are ported: `sample_unconditional` and
`sample_with_tensor`. The classifier-guided 3D interpolation is intentionally
omitted (per plan §4).

Supports two spatial layouts:

* legacy (``compressed=False``): 64x64 padded occupancy with a 7-pixel void
  border. Loss is masked to the central 50x50 cell region.
* v4+ (``compressed=True``): the 25x25 NW mirror-quadrant of the cell. No
  padding, no border crop. Mirror tile (``unfold_mirror``) reconstructs the
  full 50x50 cell at inference time.
"""
from __future__ import annotations

import math
from functools import partial
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model_utils import (
    alpha_cosine_log_snr,
    beta_linear_log_snr,
    log_snr_to_alpha_sigma,
    right_pad_dims_to,
)
from .unet import UNetModel

TRUNCATED_TIME = 0.7
TENSOR_DIM = 5  # (C11, C22, C12, C66, vol)
_CELL_SIZE = 50
_PAD_TO = 64
_OFF = (_PAD_TO - _CELL_SIZE) // 2  # 7-pixel void border for legacy 64x64 mode
_QUADRANT_SIZE = 25  # compressed-mode active content; loss cropped here when padded


def _parse_attention_sizes(spec) -> tuple[int, ...]:
    """attention_sizes accepts a YAML list like [8, 4] or a CSV string '8,4'.

    Values are interpreted as *spatial sizes* (H == W) where self-attention
    should be applied, matching the existing v1-v3 YAML convention where
    ``attention_resolutions: '8,4'`` meant attention at H=8 and H=4.
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        return tuple(int(s) for s in spec.split(",") if s.strip())
    return tuple(int(s) for s in spec)


class OccupancyDiffusion(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        base_channels: int = 64,
        attention_resolutions: str = "8,4",
        attention_sizes=None,
        dim_mults=None,
        with_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        verbose: bool = False,
        use_tensor_condition: bool = True,
        eps: float = 1e-6,
        noise_schedule: str = "linear",
        compressed: bool = False,
        num_res_blocks: int = 1,
        parameterization: str = "x0",
        min_snr_gamma: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.compressed = compressed
        self.parameterization = parameterization
        self.min_snr_gamma = min_snr_gamma

        if dim_mults is None:
            if image_size == 64:
                channel_mult = (1, 2, 4, 8, 8)
            elif image_size == 32:
                channel_mult = (1, 2, 4, 8)
            elif image_size == 25:
                channel_mult = (1, 2, 4, 8)  # 25 -> 13 -> 7 -> 4
            elif image_size == 16:
                channel_mult = (1, 2, 4)
            else:
                raise ValueError(
                    f"unsupported image size {image_size}; pass dim_mults explicitly"
                )
        else:
            channel_mult = tuple(int(m) for m in dim_mults)

        attn_sizes = _parse_attention_sizes(
            attention_sizes if attention_sizes is not None else attention_resolutions
        )

        self.eps = eps
        self.verbose = verbose
        self.use_tensor_condition = use_tensor_condition

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        self.denoise_fn = UNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult,
            dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            world_dims=2,
            num_heads=num_heads,
            attention_sizes=attn_sizes,
            with_attention=with_attention,
            verbose=verbose,
            tensor_condition_dim=TENSOR_DIM,
            num_res_blocks=num_res_blocks,
        )

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch: int, device, steps: int):
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        times = times.unsqueeze(0).expand(batch, -1)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        return times.unbind(dim=-1)

    def training_loss(self, img, tensor_feature, sample_weight=None, *args, **kwargs):
        batch = img.shape[0]
        times = torch.zeros((batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)
        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise

        # v-parameterization: target is v = α·ε − σ·x₀
        if self.parameterization == "v":
            target = alpha * noise - sigma * img
        else:
            target = img

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                pred_first = self.denoise_fn(noised_img, noise_level, tensor_feature).detach_()
                if self.parameterization == "v":
                    # convert v→x₀ for self-conditioning channel
                    self_cond = (alpha * noised_img - sigma * pred_first).detach_()
                else:
                    self_cond = pred_first

        pred = self.denoise_fn(noised_img, noise_level, tensor_feature, self_cond)

        # Crop to active region
        if self.compressed:
            pred   = pred[...,   :_QUADRANT_SIZE, :_QUADRANT_SIZE]
            target = target[..., :_QUADRANT_SIZE, :_QUADRANT_SIZE]
        else:
            pred   = pred[...,   _OFF:_OFF+_CELL_SIZE, _OFF:_OFF+_CELL_SIZE]
            target = target[..., _OFF:_OFF+_CELL_SIZE, _OFF:_OFF+_CELL_SIZE]

        mse = F.mse_loss(pred, target, reduction="none")

        if self.min_snr_gamma > 0.0:
            snr = torch.exp(noise_level)                                      # (B,)
            gamma = self.min_snr_gamma
            if self.parameterization == "v":
                # weight from the min-SNR paper for v-prediction
                weight = torch.minimum(snr, snr.new_full((), gamma)) / (snr + 1.0)
            else:
                weight = torch.minimum(snr, snr.new_full((), gamma))
            weight = weight.view(-1, *([1] * (mse.dim() - 1)))
            mse = weight * mse

        per_sample = mse.flatten(1).mean(dim=1)  # (B,)  per-sample loss
        if sample_weight is not None:
            sw = sample_weight.to(per_sample.dtype).reshape(-1)
            return (per_sample * sw).sum() / sw.sum().clamp_min(1e-8)
        return per_sample.mean()

    def denoise_step(self, img, x_start, time, time_next, tensor_cond, tensor_zero, tensor_w):
        """One DDIM-style update; returns ``(img_next, x_start)``.

        No ``no_grad`` here: when the caller is in a grad context, gradients flow
        back through ``tensor_cond`` (and ``img`` / ``x_start``). The late-step
        binarization uses a straight-through estimator (hard ``sign`` in the
        forward pass, identity gradient in the backward pass) so the gradient
        does not vanish once ``time < TRUNCATED_TIME``. Under ``no_grad`` these
        out-of-place ops are numerically identical to the original in-place ones.
        """
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, img), (log_snr, log_snr_next)
        )
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
        noise_cond = self.log_snr(time)

        x_zero = self.denoise_fn(img, noise_cond, tensor_zero, x_start)
        if tensor_cond is not None and tensor_w != 0.0:
            x_with = self.denoise_fn(img, noise_cond, tensor_cond, x_start)
            x_start = x_zero + tensor_w * (x_with - x_zero)
        else:
            x_start = x_zero

        if time[0] < TRUNCATED_TIME:
            x_start = x_start + (x_start.sign() - x_start).detach()  # straight-through sign
        x_start = x_start.clamp(-1.0, 1.0)

        pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
        img = x_start * alpha_next + pred_noise * sigma_next
        return img, x_start

    def prepare_sampling(self, batch_size: int, steps: int):
        """Set up an explicit sampling loop the caller drives with `denoise_step`.

        Returns ``(img0, tensor_zero, time_pairs)`` where ``time_pairs`` is a list
        of ``(step, (time, time_next))``. Use this when you need a grad-enabled,
        per-step conditioned loop (e.g. training a neural field through the
        sampler): build a grad-carrying ``tensor_cond`` each step, call
        ``denoise_step``, compute your loss on ``x_start``, then ``.detach()`` the
        carried ``img``/``x_start`` before the next step to bound memory.
        """
        device = self.device
        shape = (batch_size, 1, self.image_size, self.image_size)
        tensor_zero = -torch.ones(
            (batch_size, TENSOR_DIM), device=device, dtype=torch.float32
        )
        img = torch.randn(shape, device=device)
        time_pairs = list(enumerate(self.get_sampling_timesteps(batch_size, device, steps)))
        return img, tensor_zero, time_pairs

    @torch.no_grad()
    def _sample_loop(self, img, tensor_cond, tensor_zero, steps, tensor_w, verbose):
        time_pairs = self.get_sampling_timesteps(img.shape[0], self.device, steps)
        x_start = None
        _iter = tqdm(time_pairs, desc="sampling", leave=False) if verbose else time_pairs

        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next)
            )
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
            noise_cond = self.log_snr(time)

            pred_zero = self.denoise_fn(img, noise_cond, tensor_zero, x_start)
            if self.parameterization == "v":
                x_zero = alpha * img - sigma * pred_zero
            else:
                x_zero = pred_zero
            if tensor_cond is not None and tensor_w != 0.0:
                pred_with = self.denoise_fn(img, noise_cond, tensor_cond, x_start)
                if self.parameterization == "v":
                    x_with = alpha * img - sigma * pred_with
                else:
                    x_with = pred_with
                x_start = x_zero + tensor_w * (x_with - x_zero)
            else:
                x_start = x_zero

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
            x_start.clamp_(-1.0, 1.0)

            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_next + pred_noise * sigma_next

        return img

    def _as_cond(self, cond, batch_size, device):
        """Normalize a per-step condition to a (batch_size, TENSOR_DIM) tensor.

        Grad-preserving: a torch input keeps its graph (only `.to`/`expand` are
        applied). A numpy/list input is detached by construction.
        """
        if cond is None:
            return None
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(np.asarray(cond, dtype=np.float32))
        cond = cond.to(device=device, dtype=torch.float32)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0).expand(batch_size, -1)
        return cond.contiguous()

    @torch.no_grad()
    def sample_with_tensor_fn(
        self,
        cond_fn,
        batch_size: int = 1,
        steps: int = 50,
        tensor_w: float = 1.0,
        verbose: bool = True,
    ):
        """Like `sample_with_tensor`, but the condition is recomputed every step.

        `cond_fn(step, total_steps, x_start)` is called at the start of each step
        and must return the conditioning for that step as a (TENSOR_DIM,) or
        (batch_size, TENSOR_DIM) array/tensor. `x_start` is the previous step's
        predicted-clean image (None on the first step), so the callback can decode
        it, update an external model (e.g. a neural field), and return a fresh
        condition. The diffusion update stays under no_grad; do any gradient work
        inside `cond_fn` with `torch.enable_grad()`. For a loop where gradients
        must flow *through* the sampler into the condition, drive the loop yourself
        with `prepare_sampling` + `denoise_step` instead.
        """
        img, tensor_zero, time_pairs = self.prepare_sampling(batch_size, steps)
        x_start = None
        _iter = tqdm(time_pairs, desc="sampling", leave=False) if verbose else time_pairs

        for step, (time, time_next) in _iter:
            tensor_cond = self._as_cond(cond_fn(step, steps, x_start), batch_size, self.device)
            img, x_start = self.denoise_step(
                img, x_start, time, time_next, tensor_cond, tensor_zero, tensor_w
            )
        return img

    @torch.no_grad()
    def sample_unconditional(
        self,
        batch_size: int = 16,
        steps: int = 50,
        truncated_index: float = 0.0,
        verbose: bool = True,
    ):
        shape = (batch_size, 1, self.image_size, self.image_size)
        device = self.device
        tensor_zero = -torch.ones(
            (batch_size, TENSOR_DIM), device=device, dtype=torch.float32
        )
        img = torch.randn(shape, device=device)
        return self._sample_loop(img, None, tensor_zero, steps, tensor_w=0.0, verbose=verbose)

    @torch.no_grad()
    def sample_with_tensor(
        self,
        tensor_c: np.ndarray,
        batch_size: int = 16,
        steps: int = 50,
        truncated_index: float = 0.0,
        tensor_w: float = 1.0,
        verbose: bool = True,
    ):
        shape = (batch_size, 1, self.image_size, self.image_size)
        device = self.device

        tc = torch.from_numpy(tensor_c.astype(np.float32)).to(device)
        tensor_cond = tc.unsqueeze(0).expand(batch_size, -1).contiguous()
        tensor_zero = -torch.ones(
            (batch_size, TENSOR_DIM), device=device, dtype=torch.float32
        )
        img = torch.randn(shape, device=device)
        return self._sample_loop(
            img, tensor_cond, tensor_zero, steps, tensor_w=tensor_w, verbose=verbose
        )

    @torch.no_grad()
    def sample_with_tensor_batch(
        self,
        tensor_c: np.ndarray,
        steps: int = 50,
        tensor_w: float = 1.0,
        verbose: bool = True,
    ):
        """Sample one image per row of ``tensor_c`` (shape ``(B, TENSOR_DIM)``).

        Unlike ``sample_with_tensor`` (a single condition broadcast across the
        batch), each batch element carries its own condition. Used to realise a
        different microstructure per cloak cell in one diffusion run.
        """
        tc = torch.as_tensor(np.asarray(tensor_c, dtype=np.float32), device=self.device)
        if tc.ndim != 2 or tc.shape[1] != TENSOR_DIM:
            raise ValueError(f"tensor_c must be (B, {TENSOR_DIM}); got {tuple(tc.shape)}")
        batch_size = tc.shape[0]
        tensor_zero = -torch.ones(
            (batch_size, TENSOR_DIM), device=self.device, dtype=torch.float32
        )
        img = torch.randn((batch_size, 1, self.image_size, self.image_size), device=self.device)
        return self._sample_loop(
            img, tc.contiguous(), tensor_zero, steps, tensor_w=tensor_w, verbose=verbose
        )
