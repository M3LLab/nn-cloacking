"""2D conditional diffusion (continuous-time, v-DDIM-like) over 64x64 occupancy.

Adapted from microstructure_generation_3d/network/model.py. Only the two
sampling modes we actually need are ported: `sample_unconditional` and
`sample_with_tensor`. The classifier-guided 3D interpolation is intentionally
omitted (per plan §4).
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
TENSOR_DIM = 4  # (C11, C12, C66, vol)
_CELL_SIZE = 50
_PAD_TO = 64
_OFF = (_PAD_TO - _CELL_SIZE) // 2  # 7-pixel void border; loss masked to cell region only


class OccupancyDiffusion(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        base_channels: int = 64,
        attention_resolutions: str = "8,4",
        with_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        verbose: bool = False,
        use_tensor_condition: bool = True,
        eps: float = 1e-6,
        noise_schedule: str = "linear",
    ):
        super().__init__()
        self.image_size = image_size

        if image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 16:
            channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = tuple(image_size // int(r) for r in attention_resolutions.split(","))

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
            attention_resolutions=attention_ds,
            with_attention=with_attention,
            verbose=verbose,
            tensor_condition_dim=TENSOR_DIM,
        )

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch: int, device, steps: int):
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        times = times.unsqueeze(0).expand(batch, -1)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        return times.unbind(dim=-1)

    def training_loss(self, img, tensor_feature, *args, **kwargs):
        batch = img.shape[0]
        times = torch.zeros((batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)
        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(noised_img, noise_level, tensor_feature).detach_()
        pred = self.denoise_fn(noised_img, noise_level, tensor_feature, self_cond)
        return F.mse_loss(pred[..., _OFF:_OFF+_CELL_SIZE, _OFF:_OFF+_CELL_SIZE],
                          img[..., _OFF:_OFF+_CELL_SIZE, _OFF:_OFF+_CELL_SIZE])

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

            x_zero = self.denoise_fn(img, noise_cond, tensor_zero, x_start)
            if tensor_cond is not None and tensor_w != 0.0:
                x_with = self.denoise_fn(img, noise_cond, tensor_cond, x_start)
                x_start = x_zero + tensor_w * (x_with - x_zero)
            else:
                x_start = x_zero

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()
            x_start.clamp_(-1.0, 1.0)

            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)
            img = x_start * alpha_next + pred_noise * sigma_next

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
