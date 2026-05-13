"""DCGAN-style discriminator with optional Spectral Normalization."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    """Wrap conv/linear with spectral_norm if requested."""
    if use_sn:
        return nn.utils.parametrizations.spectral_norm(layer)
    return layer


class Discriminator(nn.Module):
    """Conv discriminator: (B, 3, 128, 128) -> (B, 1).

    For WGAN-GP: no spectral norm, output is unbounded (no sigmoid).
    For SN-GAN: spectral norm on all conv+linear layers.
    """

    def __init__(self, base_ch: int = 64, use_spectral_norm: bool = False) -> None:
        super().__init__()
        sn = use_spectral_norm

        self.net = nn.Sequential(
            # 128 -> 64
            _maybe_sn(nn.Conv2d(3, base_ch, 4, 2, 1, bias=False), sn),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            _maybe_sn(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False), sn),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            _maybe_sn(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False), sn),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            _maybe_sn(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False), sn),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            _maybe_sn(nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1, bias=False), sn),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 -> 1
            _maybe_sn(nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False), sn),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def compute_gradient_penalty(
    discriminator: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    **_kwargs: Any,
) -> torch.Tensor:
    """WGAN-GP gradient penalty: E[(||grad D(interp)||_2 - 1)^2]."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
