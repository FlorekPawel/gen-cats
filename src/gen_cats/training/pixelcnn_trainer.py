"""Train a PixelCNN prior on frozen VQ-VAE codebook indices."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gen_cats.factory import create_optimizer
from gen_cats.models.pixelcnn import PixelCNN
from gen_cats.models.vqvae_checkpoint import load_frozen_vqvae
from gen_cats.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PixelCNNTrainer(BaseTrainer):
    """Predict VQ code indices autoregressively; VQ-VAE encoder/decoder stay frozen."""

    pixelcnn: PixelCNN
    vqvae: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config.early_stop_metric = "val_loss"

    def build_models(self) -> None:
        vqvae_seed = self.config.vqvae_seed
        self.vqvae, self._vqvae_cfg, self._vqvae_ckpt = load_frozen_vqvae(
            self.config.checkpoint_dir,
            self.device,
            seed=vqvae_seed,
            run_name=self.config.vqvae_run_name,
        )
        num_embeddings = self.vqvae.quantizer.num_embeddings
        self.pixelcnn = PixelCNN(
            num_embeddings=num_embeddings,
            hidden_channels=self.config.prior_hidden_channels,
            n_layers=self.config.prior_n_layers,
        ).to(self.device)

    def build_optimizers(self) -> None:
        self.optimizer = create_optimizer(self.pixelcnn.parameters(), lr=self.config.lr)

    def _indices_from_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vqvae.encode_indices(images)

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        self.pixelcnn.train()
        if isinstance(batch, list | tuple):
            batch = batch[0]
        batch = batch.to(self.device)
        indices = self._indices_from_images(batch)
        logits = self.pixelcnn(indices)
        loss = F.cross_entropy(logits, indices)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        self.pixelcnn.eval()
        total = 0.0
        n = 0
        for batch in val_loader:
            if isinstance(batch, list | tuple):
                batch = batch[0]
            batch = batch.to(self.device)
            indices = self._indices_from_images(batch)
            logits = self.pixelcnn(indices)
            loss = F.cross_entropy(logits, indices)
            total += loss.item() * batch.size(0)
            n += batch.size(0)
        avg = total / max(n, 1)
        return {"val_loss": avg, "loss": avg}

    @torch.no_grad()
    def generate_samples(self, n: int) -> torch.Tensor:
        self.pixelcnn.eval()
        h = self.vqvae.feature_map_size
        indices = self.pixelcnn.sample(
            n,
            h,
            self.device,
            temperature=self.config.sample_temperature,
        )
        return self.vqvae.decode_indices(indices)

    def state_dicts(self) -> dict[str, Any]:
        return {
            "pixelcnn": self.pixelcnn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vqvae_checkpoint": str(self._vqvae_ckpt),
            "vqvae_config": self._vqvae_cfg,
        }

    def load_state_dicts(self, checkpoint: dict[str, Any]) -> None:
        self.pixelcnn.load_state_dict(checkpoint["pixelcnn"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
