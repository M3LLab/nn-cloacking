"""2D UNet for conditional occupancy diffusion.

Adapted from microstructure_generation_3d/network/unet.py. Key differences:

- `world_dims` defaults to 2 and is propagated everywhere (the 3D version had
  it hardcoded in two CrossAttention sites).
- The 3D code symmetrized the input/output over the S3 permutations of the
  three spatial axes. We DO NOT symmetrize here: the squared-assembly CA cell
  is only D4-symmetric at the tiling level, not internally (see plan §7), and
  the model is trained on the raw cell.
- `tensor_condition_dim` is 4 (C11, C12, C66, vol).
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
        attention_resolutions=(4, 8),
        with_attention: bool = True,
        verbose: bool = False,
        tensor_condition_dim: int = 4,
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
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
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
                ) if ds in attention_resolutions and with_attention else our_Identity(),
                Downsample(dim_out, dims=world_dims) if not is_last else our_Identity(),
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            tensor_condition_dim=tensor_condition_dim,
        )
        self.mid_cross_attn = our_Identity()
        self.mid_cross_attn2 = (
            CrossAttention(
                feature_dim=mid_dim, tensor_dim=emb_dim,
                num_heads=num_heads, image_size=res, world_dims=world_dims,
                drop_out=dropout,
            ) if use_tensor_condition else our_Identity()
        )
        self.mid_self_attn = (
            nn.Sequential(
                normalization(mid_dim),
                activation_function(),
                AttentionBlock(mid_dim, num_heads=num_heads),
            ) if ds in attention_resolutions and with_attention else our_Identity()
        )
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            tensor_condition_dim=tensor_condition_dim,
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
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
                ) if ds in attention_resolutions and with_attention else our_Identity(),
                Upsample(dim_in, dims=world_dims) if not is_last else our_Identity(),
            ]))
            if not is_last:
                ds //= 2

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
            # 4th channel (vol) is used by mid_cross_attn2 as a token sequence
            vol_condition = tensor_emb[:, :, 3:4].permute(0, 2, 1)

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
