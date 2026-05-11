"""Trainer for WGAN-GP and SN-GAN models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gen_cats.factory import create_optimizer
from gen_cats.models.discriminator import Discriminator, compute_gradient_penalty
from gen_cats.models.generator import Generator
from gen_cats.training.base_trainer import BaseTrainer


class GANTrainer(BaseTrainer):
    """Handles WGAN-GP and SN-GAN via config.model_type dispatch.

    WGAN-GP: Wasserstein loss + gradient penalty, n_critic updates per G step.
    SN-GAN: Hinge loss, spectral norm in discriminator.

    Early stopping for GANs: monitors generator loss stability.
    """

    generator: Generator
    discriminator: Discriminator

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.early_stopping.mode = "min"
        self.config.early_stop_metric = "g_loss"

    def build_models(self) -> None:
        self.generator = Generator(latent_dim=self.config.latent_dim).to(self.device)
        use_sn = self.config.model_type == "sn_gan"
        self.discriminator = Discriminator(use_spectral_norm=use_sn).to(self.device)

    def build_optimizers(self) -> None:
        lr_g = self.config.lr_g or self.config.lr
        lr_d = self.config.lr_d or self.config.lr
        betas_g = (0.5, 0.999) if self.config.model_type == "wgan_gp" else (0.0, 0.9)
        betas_d = betas_g
        self.opt_g = create_optimizer(self.generator.parameters(), lr=lr_g, betas=betas_g)
        self.opt_d = create_optimizer(self.discriminator.parameters(), lr=lr_d, betas=betas_d)

    def _train_discriminator(self, real: torch.Tensor) -> dict[str, float]:
        batch_size = real.size(0)
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake = self.generator(z).detach()

        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)

        if self.config.model_type == "wgan_gp":
            d_loss = d_fake.mean() - d_real.mean()
            gp = compute_gradient_penalty(self.discriminator, real, fake, self.device)
            d_loss = d_loss + self.config.gp_lambda * gp
        else:
            # Hinge loss for SN-GAN
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()

        return {
            "d_loss": d_loss.item(),
            "d_real": d_real.mean().item(),
            "d_fake": d_fake.mean().item(),
        }

    def _train_generator(self) -> dict[str, float]:
        z = torch.randn(self.config.batch_size, self.config.latent_dim, device=self.device)
        fake = self.generator(z)
        d_fake = self.discriminator(fake)

        g_loss = -d_fake.mean()

        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()

        return {"g_loss": g_loss.item()}

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        losses: dict[str, float] = {}

        for _ in range(self.config.n_critic):
            d_losses = self._train_discriminator(batch)
        losses.update(d_losses)

        g_losses = self._train_generator()
        losses.update(g_losses)

        return losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        """GAN validation: compute D scores on real/fake for monitoring."""
        self.generator.eval()
        self.discriminator.eval()

        total_g_loss = 0.0
        total_d_real = 0.0
        total_d_fake = 0.0
        n = 0

        for batch in val_loader:
            if isinstance(batch, list | tuple):
                batch = batch[0]
            batch = batch.to(self.device)
            bs = batch.size(0)

            z = torch.randn(bs, self.config.latent_dim, device=self.device)
            fake = self.generator(z)

            d_real = self.discriminator(batch).mean().item()
            d_fake_score = self.discriminator(fake).mean().item()

            total_d_real += d_real * bs
            total_d_fake += d_fake_score * bs
            total_g_loss += (-d_fake_score) * bs
            n += bs

        return {
            "g_loss": total_g_loss / max(n, 1),
            "d_real": total_d_real / max(n, 1),
            "d_fake": total_d_fake / max(n, 1),
        }

    def generate_samples(self, n: int) -> torch.Tensor:
        self.generator.eval()
        z = torch.randn(n, self.config.latent_dim, device=self.device)
        return self.generator(z)

    def state_dicts(self) -> dict[str, Any]:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }

    def load_state_dicts(self, checkpoint: dict[str, Any]) -> None:
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.opt_g.load_state_dict(checkpoint["opt_g"])
        self.opt_d.load_state_dict(checkpoint["opt_d"])
