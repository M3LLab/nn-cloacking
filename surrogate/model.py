"""
NN model to predict transmission loss from cell geometry.

Shared CNN trunk → latent geometry code z; continuous Fourier-features
decoder maps (z, f) → scalar loss. Two call modes share all weights:
    forward_at       — one f per sample, returns (B,)
    forward_spectrum — F-point grid broadcast across the batch, returns (B, F)
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from surrogate.dataset import SurrogateBatch


class FrequencyOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    predicted_cloaking: Float[Tensor, "B"]


class SpectrumOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    # B - batch size
    # F - number of frequencies
    predicted_cloaking: Float[Tensor, "B F"]


class FourierFeatures(nn.Module):
    """Sinusoidal encoding of a scalar frequency.

    Normalizes f to [0, 1] over [f_min, f_max], then applies
    [sin(π k f̂), cos(π k f̂)] for k = 1..num_bands.
    """

    def __init__(self, num_bands: int = 8, f_min: float = 1.0, f_max: float = 4.0):
        super().__init__()
        self.f_min = f_min
        self.f_max = f_max
        bands = torch.arange(1, num_bands + 1, dtype=torch.float32) * math.pi
        self.register_buffer("bands", bands)
        self.out_dim = 2 * num_bands

    def forward(self, f: Tensor) -> Tensor:
        f_norm = (f - self.f_min) / (self.f_max - self.f_min)
        angles = f_norm.unsqueeze(-1) * self.bands
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class SpectrumDecoder(nn.Module):
    """MLP that maps concat(z, Fourier(f)) → scalar prediction."""

    def __init__(self, z_dim: int, ff_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + ff_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),          nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_ff: Tensor) -> Tensor:
        return self.net(z_ff).squeeze(-1)


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-v1 block: depthwise conv → LayerNorm → pointwise expand → GELU → pointwise project, residual."""

    def __init__(self, dim: int, kernel: int = 7, mlp_ratio: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel,
                                padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(mlp_ratio * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)           # B H W C for channel-last norm + MLP
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class SmallConvNeXt(nn.Module):
    """ConvNeXt-style trunk sized for 10×10–50×50 cell grids.

    Stride-1 stem (preserves spatial dim) then three stages of 2 blocks each
    with stride-2 downsamples between. Spatial flow: 10→10→5→2 / 20→20→10→5 /
    50→50→25→12 before global average pooling. ~1.6M params at default widths.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        dims: tuple[int, int, int] = (64, 128, 256),
        depths: tuple[int, int, int] = (2, 2, 2),
        kernel: int = 7,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1),
            LayerNorm2d(dims[0]),
        )
        self.stages = nn.ModuleList([
            nn.Sequential(*[ConvNeXtBlock(dims[0], kernel) for _ in range(depths[0])])
        ])
        self.downsamples = nn.ModuleList()
        for i in range(1, len(dims)):
            self.downsamples.append(nn.Sequential(
                LayerNorm2d(dims[i - 1]),
                nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
            ))
            self.stages.append(nn.Sequential(
                *[ConvNeXtBlock(dims[i], kernel) for _ in range(depths[i])]
            ))
        self.head_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages[0](x)
        for ds, stage in zip(self.downsamples, self.stages[1:]):
            x = stage(ds(x))
        x = x.mean(dim=(2, 3))  # global average pool
        return self.head(self.head_norm(x))


def _build_convnext_tiny_trunk(in_channels: int, z_dim: int) -> nn.Module:
    """Torchvision ConvNeXt-Tiny with stem swapped for `in_channels` and head → `z_dim`.

    Random init (weights=None). Works for 50×50 grids but FAILS on ≤20×20 due
    to ConvNeXt's 32× cumulative downsampling.
    """
    from torchvision.models import convnext_tiny

    trunk = convnext_tiny(weights=None)
    stem_conv: nn.Conv2d = trunk.features[0][0]
    trunk.features[0][0] = nn.Conv2d(
        in_channels,
        stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
    )
    final_linear: nn.Linear = trunk.classifier[-1]
    trunk.classifier[-1] = nn.Linear(final_linear.in_features, z_dim)
    return trunk


class ForwardFEM_CNN(nn.Module):
    def __init__(
        self,
        n_C_params: int,
        z_dim: int = 128,
        fourier_bands: int = 8,
        f_min: float = 1.0,
        f_max: float = 4.0,
        decoder_hidden: int = 128,
        backbone: str = "small_convnext",
    ):
        super().__init__()
        in_channels = n_C_params + 1  # C params stacked with rho

        if backbone == "small_convnext":
            self.trunk = SmallConvNeXt(in_channels=in_channels, out_dim=z_dim)
        elif backbone == "convnext_tiny":
            self.trunk = _build_convnext_tiny_trunk(in_channels, z_dim)
        else:
            raise ValueError(
                f"Unknown backbone {backbone!r}. "
                f"Expected 'small_convnext' or 'convnext_tiny'."
            )
        self.backbone = backbone

        self.fourier = FourierFeatures(fourier_bands, f_min, f_max)
        self.decoder = SpectrumDecoder(z_dim, self.fourier.out_dim, decoder_hidden)

    def encode(self, batch: SurrogateBatch) -> Tensor:
        """Run the trunk once; returns the geometry latent code z, shape (B, d)."""
        x = torch.cat([batch.C, batch.rho.unsqueeze(-1)], dim=-1)  # B X Y (P+1)
        x = rearrange(x, "b x y c -> b c x y")
        return self.trunk(x)

    def forward_at(
        self,
        batch: SurrogateBatch,
        f: Tensor | None = None,
    ) -> FrequencyOutput:
        """One frequency per sample (shape B). Defaults to batch.f_star."""
        if f is None:
            f = batch.f_star
        z = self.encode(batch)
        ff = self.fourier(f)
        pred = self.decoder(torch.cat([z, ff], dim=-1))
        return FrequencyOutput(predicted_cloaking=pred)

    def forward_spectrum(
        self,
        batch: SurrogateBatch,
        f_grid: Tensor,
    ) -> SpectrumOutput:
        """Shared frequency grid (shape F) evaluated for every sample."""
        z = self.encode(batch)                                # (B, d)
        ff = self.fourier(f_grid)                             # (F, FF)
        n_batch, n_freq = z.shape[0], f_grid.shape[0]
        z_rep = z.unsqueeze(1).expand(-1, n_freq, -1)         # (B, F, d)
        ff_rep = ff.unsqueeze(0).expand(n_batch, -1, -1)      # (B, F, FF)
        pred = self.decoder(torch.cat([z_rep, ff_rep], dim=-1))
        return SpectrumOutput(predicted_cloaking=pred)

    def forward(self, batch: SurrogateBatch) -> FrequencyOutput:
        return self.forward_at(batch)

