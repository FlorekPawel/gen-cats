"""Lightweight PixelCNN prior over VQ-VAE codebook indices."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """2-D masked convolution for autoregressive factorization (van den Oord et al.)."""

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        padding: int | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        if mask_type not in {"A", "B"}:
            msg = f"mask_type must be 'A' or 'B', got {mask_type!r}"
            raise ValueError(msg)
        self.register_buffer("mask", torch.ones_like(self.weight))
        center = kernel_size // 2
        self.mask[:, :, center:, :] = 0
        self.mask[:, :, center, center + 1 :] = 0
        if mask_type == "B":
            self.mask[:, :, center, center] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding)


class PixelCNN(nn.Module):
    """Predict codebook index at each spatial location given previous indices.

    Args:
        num_embeddings: VQ codebook size K
        hidden_channels: width of conv stack
        n_layers: number of masked conv blocks (alternating mask B)
    """

    def __init__(
        self,
        num_embeddings: int,
        hidden_channels: int = 128,
        n_layers: int = 10,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_channels = hidden_channels

        self.embed = nn.Embedding(num_embeddings, hidden_channels)
        blocks: list[nn.Module] = [
            MaskedConv2d("A", hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_layers):
            blocks.extend(
                [
                    MaskedConv2d("B", hidden_channels, hidden_channels),
                    nn.GroupNorm(8, hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden_channels, num_embeddings, 1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B, H, W) long → logits (B, K, H, W)."""
        h = self.embed(indices).permute(0, 3, 1, 2).contiguous()
        h = self.blocks(h)
        return self.head(h)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        spatial_size: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive sample of index maps (n, H, W)."""
        self.eval()
        out = torch.zeros(n, spatial_size, spatial_size, dtype=torch.long, device=device)
        for row in range(spatial_size):
            for col in range(spatial_size):
                logits = self.forward(out)[:, :, row, col]
                if temperature <= 0:
                    idx = logits.argmax(dim=1)
                else:
                    probs = F.softmax(logits / temperature, dim=1)
                    idx = torch.multinomial(probs, 1).squeeze(1)
                out[:, row, col] = idx
        return out
