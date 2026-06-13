"""DCGAN-style generator for 64x64 or 128x128 RGB image synthesis."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

NormKind = Literal["batch", "instance"]

SUPPORTED_IMAGE_SIZES = frozenset({64, 128})


def _norm2d(channels: int, kind: NormKind) -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.BatchNorm2d(channels)


def _validate_image_size(image_size: int) -> None:
    if image_size not in SUPPORTED_IMAGE_SIZES:
        msg = f"image_size must be one of {sorted(SUPPORTED_IMAGE_SIZES)}, got {image_size}"
        raise ValueError(msg)


def _generator_upsample_channels(base_ch: int, image_size: int) -> list[tuple[int, int]]:
    """Channel pairs for each upsample stage after the initial 4x4 projection."""
    _validate_image_size(image_size)
    n_stages = int(math.log2(image_size)) - 2
    full = [
        (base_ch * 8, base_ch * 8),
        (base_ch * 8, base_ch * 4),
        (base_ch * 4, base_ch * 2),
        (base_ch * 2, base_ch),
        (base_ch, 3),
    ]
    if n_stages >= len(full):
        return full
    prefix = full[: n_stages - 1]
    last_in = full[n_stages - 2][1]
    return [*prefix, (last_in, 3)]


class Generator(nn.Module):
    """Transposed-conv generator: z (B, latent_dim) -> (B, 3, H, H).

    ``image_size`` 64: FC -> 4x4 -> ... -> 64x64 (chimera).
    ``image_size`` 128: FC -> 4x4 -> ... -> 128x128 (cat sweep).

    SN-GAN uses ``instance`` norm (BatchNorm + spectral-norm D often collapses to grey).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_ch: int = 64,
        norm: NormKind = "batch",
        image_size: int = 128,
    ) -> None:
        super().__init__()
        _validate_image_size(image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size

        def norm_layer(ch: int) -> nn.Module:
            return _norm2d(ch, norm)

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, base_ch * 8, 4, 1, 0, bias=False),
            norm_layer(base_ch * 8),
            nn.ReLU(inplace=True),
        ]

        for in_ch, out_ch in _generator_upsample_channels(base_ch, image_size):
            layers.extend(
                [
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    norm_layer(out_ch) if out_ch != 3 else nn.Identity(),
                    nn.Tanh() if out_ch == 3 else nn.ReLU(inplace=True),
                ]
            )

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d | nn.InstanceNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(-1, self.latent_dim, 1, 1))
