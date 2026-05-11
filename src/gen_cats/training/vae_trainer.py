"""Trainer for Beta-VAE and VQ-VAE models."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from gen_cats.factory import create_optimizer
from gen_cats.models.beta_vae import BetaVAE
from gen_cats.models.vqvae import VQVAE
from gen_cats.training.base_trainer import BaseTrainer


class VAETrainer(BaseTrainer):
    """Handles both BetaVAE and VQVAE via model_type dispatch."""

    model: BetaVAE | VQVAE

    def build_models(self) -> None:
        if self.config.model_type == "vqvae":
            self.model = VQVAE(
                num_embeddings=self.config.num_embeddings,
                embedding_dim=self.config.embedding_dim,
                commitment_cost=self.config.commitment_cost,
                feature_map_size=self.config.feature_map_size,
            ).to(self.device)
        else:
            self.model = BetaVAE(
                latent_dim=self.config.latent_dim,
                beta=self.config.beta,
            ).to(self.device)

    def build_optimizers(self) -> None:
        self.optimizer = create_optimizer(self.model.parameters(), lr=self.config.lr)

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        self.model.train()

        if isinstance(self.model, VQVAE):
            recon, vq_loss, _ = self.model(batch)
            losses = self.model.loss(batch, recon, vq_loss, self.config.recon_loss)
        else:
            recon, mu, logvar = self.model(batch)
            losses = self.model.loss(batch, recon, mu, logvar, self.config.recon_loss)

        self.optimizer.zero_grad()
        losses["total"].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        n_batches = 0

        for batch in val_loader:
            if isinstance(batch, list | tuple):
                batch = batch[0]
            batch = batch.to(self.device)

            if isinstance(self.model, VQVAE):
                recon, vq_loss, _ = self.model(batch)
                losses = self.model.loss(batch, recon, vq_loss, self.config.recon_loss)
            else:
                recon, mu, logvar = self.model(batch)
                losses = self.model.loss(batch, recon, mu, logvar, self.config.recon_loss)

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        avg["val_loss"] = avg["total"]
        return avg

    def generate_samples(self, n: int) -> torch.Tensor:
        self.model.eval()
        if isinstance(self.model, BetaVAE):
            return self.model.sample(n, self.device)
        # VQ-VAE: decode random codebook entries
        h = self.model.feature_map_size
        indices = torch.randint(
            0, self.model.quantizer.num_embeddings, (n, h, h), device=self.device
        )
        z_q = self.model.quantizer.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.model.decode(z_q)

    def state_dicts(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dicts(self, checkpoint: dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
