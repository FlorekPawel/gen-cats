"""VQ-VAE-1 with discrete codebook for 128x128 RGB images."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Straight-through estimator vector quantization.

    Args:
        num_embeddings: codebook size (512 or 1024)
        embedding_dim: dimension of each codebook vector
        commitment_cost: weight for commitment loss term
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, C, H, W) → (B*H*W, C)
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, self.embedding_dim)

        # nearest neighbor lookup
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )
        indices = dist.argmin(dim=1)
        quantized_flat = self.embedding(indices)

        # losses
        codebook_loss = F.mse_loss(quantized_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through estimator
        quantized_flat = flat + (quantized_flat - flat).detach()
        quantized = quantized_flat.view(z_perm.shape).permute(0, 3, 1, 2).contiguous()

        return quantized, vq_loss, indices


class VQVAEEncoder(nn.Module):
    """Encoder: (B, 3, 128, 128) -> (B, embedding_dim, H', W')."""

    def __init__(self, embedding_dim: int = 64, feature_map_size: int = 16) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            # 128 -> 64
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 -> 32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 32 -> 16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        if feature_map_size == 8:
            layers.extend(
                [
                    # 16 -> 8
                    nn.Conv2d(256, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, embedding_dim, 3, 1, 1),
                ]
            )
        else:
            layers.append(nn.Conv2d(256, embedding_dim, 3, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQVAEDecoder(nn.Module):
    """Decoder: (B, embedding_dim, H', W') -> (B, 3, 128, 128)."""

    def __init__(self, embedding_dim: int = 64, feature_map_size: int = 16) -> None:
        super().__init__()
        if feature_map_size == 8:
            layers: list[nn.Module] = [
                nn.Conv2d(embedding_dim, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # 8 -> 16
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ]
        else:
            layers = [
                nn.Conv2d(embedding_dim, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ]

        layers.extend(
            [
                # 16 -> 32
                nn.ConvTranspose2d(128, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # 32 -> 64
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # 64 -> 128
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


class VQVAE(nn.Module):
    """VQ-VAE-1 with discrete codebook.

    Grid params:
        num_embeddings ∈ {512, 1024}
        feature_map_size ∈ {16, 8}  (spatial dim after encoder)
        recon_loss ∈ {"l1", "mse"}
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        feature_map_size: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = VQVAEEncoder(embedding_dim, feature_map_size)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQVAEDecoder(embedding_dim, feature_map_size)
        self.embedding_dim = embedding_dim
        self.feature_map_size = feature_map_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to quantized latent (for LDM use in Milestone 5)."""
        z_e = self.encoder(x)
        z_q, _, _ = self.quantizer(z_e)
        return z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        vq_loss: torch.Tensor,
        recon_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if recon_type == "l1":
            recon_loss = F.l1_loss(recon, x, reduction="mean")
        else:
            recon_loss = F.mse_loss(recon, x, reduction="mean")

        total = recon_loss + vq_loss
        return {"total": total, "recon": recon_loss, "vq": vq_loss}
