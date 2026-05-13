"""DCGAN-style generator for 128x128 RGB image synthesis."""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Transposed-conv generator: z (B, latent_dim) -> (B, 3, 128, 128).

    Architecture: FC -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    """

    def __init__(self, latent_dim: int = 128, base_ch: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # z -> (base_ch*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            # 4 -> 8
            nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            # 8 -> 16
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            # 16 -> 32
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(-1, self.latent_dim, 1, 1))
