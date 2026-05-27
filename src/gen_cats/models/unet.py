"""Time-conditioned U-Net for denoising diffusion on 128x128 images."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def default_ch_mults(spatial_size: int, max_levels: int = 4) -> tuple[int, ...]:
    """Pick downsample depth so resolution halves until 1x1 without redundant 1x1 downs.

    Fixed ``(1, 2, 4, 8)`` is correct for 128x128 / 16x16 but breaks skip alignment at 8x8.
    """
    if spatial_size < 2 or spatial_size & (spatial_size - 1):
        raise ValueError(f"spatial_size must be a power of 2 >= 2, got {spatial_size}")
    n_levels = min(max_levels, int(math.log2(spatial_size)))
    return tuple(min(2**i, 8) for i in range(n_levels))


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding → (B, dim)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """Compact U-Net for diffusion.

    4-level encoder/decoder with skip connections.
    Input/output: (B, in_ch, H, W) — fully convolutional.

    Args:
        in_ch: input channels (3 for pixel-space, embedding_dim for latent)
        base_ch: width of first layer (grid: 32 or 64)
        spatial_size: H=W input resolution; sets ``ch_mults`` when omitted
        ch_mults: channel multipliers per level (overrides ``spatial_size``)
        time_dim: timestep embedding dimension
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 64,
        spatial_size: int | None = 128,
        ch_mults: tuple[int, ...] | None = None,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if ch_mults is None:
            if spatial_size is None:
                spatial_size = 128
            ch_mults = default_ch_mults(spatial_size)
        self.ch_mults = ch_mults
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch
        for mult in ch_mults:
            out_ch = base_ch * mult
            self.down_blocks.append(ResBlock(ch, out_ch, time_dim))
            channels.append(out_ch)
            self.downsamples.append(Downsample(out_ch))
            ch = out_ch

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.upsamples.append(Upsample(ch))
            self.up_blocks.append(ResBlock(ch + out_ch, out_ch, time_dim))
            ch = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.init_conv(x)

        skips = []
        for down_block, downsample in zip(self.down_blocks, self.downsamples, strict=True):
            h = down_block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for up_block, upsample in zip(self.up_blocks, self.upsamples, strict=True):
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)

        return self.final(h)
