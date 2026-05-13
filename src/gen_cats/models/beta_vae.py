"""Beta-VAE with convolutional encoder/decoder for 128x128 RGB images."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Conv encoder: (B, 3, 128, 128) -> (B, latent_dim) mean + logvar."""

    def __init__(self, latent_dim: int = 128, base_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = base_ch * 8 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """ConvTranspose decoder: (B, latent_dim) -> (B, 3, 128, 128)."""

    def __init__(self, latent_dim: int = 128, base_ch: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch * 8 * 4 * 4)
        self.base_ch = base_ch
        self.net = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            # 8 -> 16
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            # 16 -> 32
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.base_ch * 8, 4, 4)
        return self.net(h)


class BetaVAE(nn.Module):
    """Beta-VAE: encoder → reparameterize → decoder.

    Grid params: latent_dim ∈ {64, 128}, beta ∈ {1.0, 4.0}.
    """

    def __init__(self, latent_dim: int = 128, beta: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        recon_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if recon_type == "l1":
            recon_loss = F.l1_loss(recon, x, reduction="mean")
        else:
            recon_loss = F.mse_loss(recon, x, reduction="mean")

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return {"total": total, "recon": recon_loss, "kl": kl_loss}
