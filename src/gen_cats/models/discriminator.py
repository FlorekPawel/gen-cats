"""DCGAN-style discriminator with optional Spectral Normalization."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from gen_cats.models.generator import _validate_image_size


def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    """Wrap conv/linear with spectral_norm if requested."""
    if use_sn:
        return nn.utils.parametrizations.spectral_norm(layer)
    return layer


def _discriminator_channel_schedule(base_ch: int, image_size: int) -> list[tuple[int, int]]:
    """Conv channel pairs for each stride-2 downsample (excluding final 4x4 -> 1)."""
    _validate_image_size(image_size)
    n_down = int(math.log2(image_size)) - 2
    widths = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
    pairs: list[tuple[int, int]] = [(3, widths[0])]
    for i in range(n_down - 1):
        pairs.append((widths[i], widths[i + 1]))
    return pairs


class Discriminator(nn.Module):
    """Conv discriminator: (B, 3, H, H) -> (B, 1) for H in {64, 128}.

    For WGAN-GP: no spectral norm, output is unbounded (no sigmoid).
    For SN-GAN: spectral norm on all conv+linear layers.
    """

    def __init__(
        self,
        base_ch: int = 64,
        use_spectral_norm: bool = False,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        _validate_image_size(image_size)
        self.image_size = image_size
        sn = use_spectral_norm

        layers: list[nn.Module] = []
        for in_ch, out_ch in _discriminator_channel_schedule(base_ch, image_size):
            layers.extend(
                [
                    _maybe_sn(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False), sn),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(_maybe_sn(nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False), sn))
        self.net = nn.Sequential(*layers)
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
