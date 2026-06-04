"""Trainer for DDIM (pixel-space) and Tiny LDM (latent diffusion with frozen VQ-VAE)."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gen_cats.factory import create_optimizer
from gen_cats.models.ddim import DDIMScheduler
from gen_cats.models.unet import UNet
from gen_cats.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class DiffusionTrainer(BaseTrainer):
    """Handles pixel-space DDIM and Tiny LDM.

    For "ddim": U-Net predicts noise on 128x128 pixel images (deep U-Net, EMA on by default).
    For "tiny_ldm": U-Net predicts noise in scaled VQ-VAE latent space with frozen encoder/decoder.
    """

    unet: UNet
    scheduler: DDIMScheduler
    latent_scale: float = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config.early_stop_metric = "val_loss"
        if self.config.model_type in ("ddim", "tiny_ldm"):
            self.config.use_ema = True
        if self.config.model_type == "tiny_ldm" and self.config.vqvae_selection == "slug":
            self.config.require_vqvae_slug = True

    def build_models(self) -> None:
        is_ldm = self.config.model_type == "tiny_ldm"

        if is_ldm:
            self._load_frozen_vqvae()
            in_ch = self.vqvae.embedding_dim
        else:
            in_ch = 3
            self.vqvae = None  # type: ignore[assignment]

        spatial = 128 if not is_ldm else self.vqvae.feature_map_size
        unet_max_levels = self.config.unet_max_levels
        if unet_max_levels is None:
            log_sp = int(math.log2(spatial))
            # LDM: 16→2x2 / 8→4x4. DDIM: 128→8x8 (avoid 1x1 mid).
            unet_max_levels = max(2, log_sp - 1) if is_ldm else max(2, log_sp - 3)
        self.unet = UNet(
            in_ch=in_ch,
            base_ch=self.config.base_channels,
            spatial_size=spatial,
            unet_max_levels=unet_max_levels,
        ).to(self.device)
        logger.info(
            "U-Net ch_mults=%s base_ch=%d spatial=%d max_levels=%s",
            self.unet.ch_mults,
            self.config.base_channels,
            spatial,
            unet_max_levels,
        )

        self.scheduler = DDIMScheduler(
            timesteps=self.config.timesteps,
            schedule=self.config.noise_schedule,
        ).to(self.device)

        if self.config.use_ema:
            self.ema_unet = copy.deepcopy(self.unet)
            self.ema_unet.requires_grad_(False)
        else:
            self.ema_unet = None

    def _load_frozen_vqvae(self) -> None:
        """Load best VQ-VAE checkpoint and freeze."""
        from gen_cats.models.vqvae_checkpoint import load_frozen_vqvae

        self.vqvae, _, self._vqvae_ckpt = load_frozen_vqvae(
            self.config.checkpoint_dir,
            self.device,
            self.config,
            strict=self.config.require_vqvae_slug,
        )
        logger.info("Using VQ-VAE checkpoint: %s", self._vqvae_ckpt)

    def on_train_start(self, val_loader: DataLoader[Any]) -> None:
        if self.config.model_type == "tiny_ldm" and self.latent_scale <= 1.0 + 1e-6:
            self.calibrate_latent_scale(val_loader)

    @torch.no_grad()
    def calibrate_latent_scale(self, val_loader: DataLoader[Any]) -> None:
        """Estimate latent std on validation data (Tiny LDM only)."""
        if self.vqvae is None:
            return

        self.vqvae.eval()
        chunks: list[torch.Tensor] = []
        max_batches = self.config.latent_scale_batches

        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            if isinstance(batch, list | tuple):
                batch = batch[0]
            batch = batch.to(self.device)
            z = self.vqvae.encode_continuous(batch)
            chunks.append(z.reshape(z.size(0), -1))

        if not chunks:
            logger.warning("No validation batches for latent scale calibration")
            return

        scale = torch.cat(chunks, dim=0).std().clamp(min=1e-4)
        self.latent_scale = float(scale.item())
        logger.info(
            "Tiny LDM latent scale from %d val batch(es): %.6f",
            min(len(chunks), max_batches),
            self.latent_scale,
        )

    def _update_ema(self) -> None:
        if self.ema_unet is None:
            return
        decay = self.config.ema_decay
        for ema_p, p in zip(self.ema_unet.parameters(), self.unet.parameters(), strict=True):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """LDM: continuous pre-quant latents, scaled to ~unit std. Pixels: identity."""
        if self.vqvae is not None:
            with torch.no_grad():
                z = self.vqvae.encode_continuous(x)
            return z / self.latent_scale
        return x

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """LDM: unscale, vector-quantize, then VQ decoder. Pixels: identity."""
        if self.vqvae is not None:
            with torch.no_grad():
                return self.vqvae.decode_latent(z * self.latent_scale)
        return z

    def build_optimizers(self) -> None:
        self.optimizer = create_optimizer(self.unet.parameters(), lr=self.config.lr)

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        self.unet.train()
        x = self._encode(batch)

        noise = torch.randn_like(x)
        t = torch.randint(0, self.config.timesteps, (x.size(0),), device=self.device)
        x_noisy, _ = self.scheduler.q_sample(x, t, noise)

        pred_noise = self.unet(x_noisy, t)
        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_ema()

        return {"loss": loss.item()}

    def _denoise_model(self) -> UNet:
        """Model used for val / sampling (EMA when enabled)."""
        if self.ema_unet is not None:
            return self.ema_unet
        return self.unet

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        model = self._denoise_model()
        model.eval()
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            if isinstance(batch, list | tuple):
                batch = batch[0]
            batch = batch.to(self.device)
            x = self._encode(batch)

            noise = torch.randn_like(x)
            t = torch.randint(0, self.config.timesteps, (x.size(0),), device=self.device)
            x_noisy, _ = self.scheduler.q_sample(x, t, noise)

            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        avg_loss = total_loss / max(n, 1)
        return {"val_loss": avg_loss}

    @torch.no_grad()
    def generate_samples(self, n: int) -> torch.Tensor:
        model = self._denoise_model()
        model.eval()

        if self.vqvae is not None:
            h = self.vqvae.feature_map_size
            shape = (n, self.vqvae.embedding_dim, h, h)
        else:
            shape = (n, 3, 128, 128)

        z = self.scheduler.ddim_sample(
            model,
            shape,
            self.device,
            ddim_steps=self.config.ddim_steps,
            clamp_x0=self.vqvae is None,
        )
        return self._decode(z)

    def state_dicts(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "unet": self.unet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema_unet is not None:
            d["ema_unet"] = self.ema_unet.state_dict()
        if self.vqvae is not None:
            d["vqvae_checkpoint"] = str(self._vqvae_ckpt)
            d["latent_scale"] = self.latent_scale
        return d

    def _reload_vqvae_from_path(self, path: str | Path) -> None:
        if self.vqvae is None:
            return
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning("VQ-VAE checkpoint missing at %s", ckpt_path)
            return
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.vqvae.load_state_dict(ckpt["model"])
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)
        self._vqvae_ckpt = ckpt_path
        logger.info("Reloaded frozen VQ-VAE from %s", ckpt_path)

    def load_state_dicts(self, checkpoint: dict[str, Any]) -> None:
        self.unet.load_state_dict(checkpoint["unet"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.ema_unet is not None and "ema_unet" in checkpoint:
            self.ema_unet.load_state_dict(checkpoint["ema_unet"])
        if "latent_scale" in checkpoint:
            self.latent_scale = float(checkpoint["latent_scale"])
        vqvae_path = checkpoint.get("vqvae_checkpoint")
        if vqvae_path:
            self._reload_vqvae_from_path(vqvae_path)
