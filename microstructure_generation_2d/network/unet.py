"""2D UNet for conditional occupancy diffusion.

Adapted from microstructure_generation_3d/network/unet.py. Key differences:

- `world_dims` defaults to 2 and is propagated everywhere (the 3D version had
  it hardcoded in two CrossAttention sites).
- No in-network symmetrization. v1-v3 trained on a 64x64 padded cell; v4+
  trains on the compressed 25x25 NW mirror-quadrant and the inverse mirror
  tile (`unfold_mirror`) is applied deterministically at inference time.
- Down/up path tracks spatial sizes explicitly (ceil-mode stride-2) so the
  network works on non-power-of-2 inputs like 25 -> 13 -> 7 -> 4. Each
  Upsample is instantiated with the matching down-side target size.
- Attention placement is configured by spatial size via `attention_sizes`
  (a set of int H values), not by downsample-factor `attention_resolutions`.
- `tensor_condition_dim` is 5 (C11, C22, C12, C66, vol); vol is the last channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model_utils import (
    AttentionBlock,
    CrossAttention,
    Downsample,
    LearnedSinusoidalPosEmb,
    ResnetBlock,
    Upsample,
    activation_function,
    conv_nd,
    normalization,
    our_Identity,
)
from ..utils.utils import default


class UNetModel(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        base_channels: int = 64,
        dim_mults=(1, 2, 4, 8, 8),
        dropout: float = 0.1,
        num_heads: int = 4,
        world_dims: int = 2,
        attention_sizes=(4, 8),
        with_attention: bool = True,
        verbose: bool = False,
        tensor_condition_dim: int = 5,
        use_tensor_condition: bool = True,
    ):
        super().__init__()
        self.world_dims = world_dims
        self.use_tensor_condition = use_tensor_condition
        self.tensor_condition_dim = tensor_condition_dim
        self.verbose = verbose

        channels = [base_channels, *map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))
        emb_dim = base_channels * 4

        attention_sizes = set(int(s) for s in attention_sizes)

        # Pre-compute per-level spatial sizes via ceil-mode stride-2 halving so
        # the up-path can target the matching down-side size on non-power-of-2
        # inputs (e.g. 25 -> 13 -> 7 -> 4 for image_size=25, 4 levels).
        num_resolutions = len(in_out)
        down_sizes = [image_size]
        for i in range(num_resolutions - 1):
            down_sizes.append((down_sizes[-1] + 1) // 2)
        self.down_sizes = down_sizes
        mid_size = down_sizes[-1]

        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim),
        )

        if use_tensor_condition:
            self.tensor_pos_embs = nn.ModuleList(
                [LearnedSinusoidalPosEmb(base_channels) for _ in range(tensor_condition_dim)]
            )
            self.tensor_embs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(base_channels + 1, emb_dim),
                        activation_function(),
                        nn.Linear(emb_dim, emb_dim),
                    )
                    for _ in range(tensor_condition_dim)
                ]
            )

        # input channels: x + self-conditioning (= 2 channels of occupancy)
        self.input_emb = conv_nd(world_dims, 2, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            block_size = down_sizes[ind]
            self.downs.append(nn.ModuleList([
                ResnetBlock(
                    world_dims, dim_in, dim_out, emb_dim=emb_dim, dropout=dropout,
                    use_tensor_condition=use_tensor_condition,
                    tensor_condition_dim=tensor_condition_dim,
                ),
                our_Identity(),  # placeholder (3D code had a disabled cross attn here)
                our_Identity(),  # placeholder (vol cross attn; disabled in 3D)
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(dim_out, num_heads=num_heads),
                ) if block_size in attention_sizes and with_attention else our_Identity(),
                Downsample(dim_out, dims=world_dims) if not is_last else our_Identity(),
            ]))

        mid_dim = channels[-1]
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            tensor_condition_dim=tensor_condition_dim,
        )
        self.mid_cross_attn = our_Identity()
        self.mid_cross_attn2 = (
            CrossAttention(
                feature_dim=mid_dim, tensor_dim=emb_dim,
                num_heads=num_heads, image_size=mid_size, world_dims=world_dims,
                drop_out=dropout,
            ) if use_tensor_condition else our_Identity()
        )
        self.mid_self_attn = (
            nn.Sequential(
                normalization(mid_dim),
                activation_function(),
                AttentionBlock(mid_dim, num_heads=num_heads),
            ) if mid_size in attention_sizes and with_attention else our_Identity()
        )
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            tensor_condition_dim=tensor_condition_dim,
        )

        # Up path iterates num_resolutions-1 blocks. Each upsample reverses one
        # down step: iteration ind targets down_sizes[num_resolutions-2-ind].
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            target_size = down_sizes[num_resolutions - 2 - ind]
            # ResnetBlock at this stage runs at the *deeper* (smaller) size,
            # which is target_size of the *previous* upsample, i.e. one level
            # smaller than target_size for the very first up-iter (mid_size)
            # and otherwise down_sizes[num_resolutions-1-ind].
            block_size = down_sizes[num_resolutions - 1 - ind]
            self.ups.append(nn.ModuleList([
                ResnetBlock(
                    world_dims, dim_out * 2, dim_in, emb_dim=emb_dim, dropout=dropout,
                    use_tensor_condition=use_tensor_condition,
                    tensor_condition_dim=tensor_condition_dim,
                ),
                our_Identity(),
                our_Identity(),
                nn.Sequential(
                    normalization(dim_in),
                    activation_function(),
                    AttentionBlock(dim_in, num_heads=num_heads),
                ) if block_size in attention_sizes and with_attention else our_Identity(),
                Upsample(dim_in, dims=world_dims, target_size=target_size),
            ]))

        self.end = nn.Sequential(normalization(base_channels), activation_function())
        self.out = conv_nd(world_dims, base_channels, 1, 3, padding=1)

    def _build_tensor_emb(self, tensor_condition: torch.Tensor, device, dtype):
        # tensor_condition: (B, D). Per-channel sinusoid -> MLP -> (B, emb_dim, D)
        B, D = tensor_condition.shape
        assert D == self.tensor_condition_dim, (D, self.tensor_condition_dim)
        emb_dim = self.tensor_embs[0][-1].out_features
        out = torch.zeros((B, emb_dim, D), device=device, dtype=dtype)
        for i in range(D):
            out[:, :, i] = self.tensor_embs[i](self.tensor_pos_embs[i](tensor_condition[:, i]))
        return out

    def forward(self, x, t, tensor_condition=None, x_self_cond=None):
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x, x_self_cond), dim=1)
        x = self.input_emb(x)
        t_emb = self.time_emb(self.time_pos_emb(t))

        vol_condition = None
        tensor_emb = None
        if self.use_tensor_condition:
            tensor_emb = self._build_tensor_emb(tensor_condition, x.device, t_emb.dtype)
            # last channel (vol) is used by mid_cross_attn2 as a token sequence
            vol_condition = tensor_emb[:, :, -1:].permute(0, 2, 1)

        h = []
        for resnet, _ca, _ca2, self_attn, downsample in self.downs:
            x = resnet(x, t_emb, tensor_emb)
            x = self_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb, tensor_emb)
        x = self.mid_cross_attn(x)
        x = self.mid_cross_attn2(x, vol_condition)
        x = self.mid_self_attn(x)
        x = self.mid_block2(x, t_emb, tensor_emb)

        for resnet, _ca, _ca2, self_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t_emb, tensor_emb)
            x = self_attn(x)
            x = upsample(x)

        x = self.end(x)
        x = self.out(x)
        return x
