"""Load frozen VQ-VAE checkpoints for LDM and PixelCNN prior training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from gen_cats.config import TrainConfig, checkpoint_run_slug
from gen_cats.models.vqvae import VQVAE

logger = logging.getLogger(__name__)


def resolve_vqvae_checkpoint(
    checkpoint_dir: str | Path,
    seed: int,
    run_name: str = "",
) -> Path:
    """Resolve path to a VQ-VAE ``best`` checkpoint."""
    root = Path(checkpoint_dir) / "vqvae"
    if run_name.strip():
        slug = checkpoint_run_slug(TrainConfig(model_type="vqvae", run_name=run_name))
        candidate = root / slug / f"best_seed{seed}.pt"
        if candidate.exists():
            return candidate

    def mtime_key(path: Path) -> float:
        return path.stat().st_mtime

    candidates = sorted(root.glob(f"**/best_seed{seed}.pt"), key=mtime_key, reverse=True)
    if not candidates:
        candidates = sorted(root.glob("**/best_seed*.pt"), key=mtime_key, reverse=True)
    if not candidates:
        msg = f"No VQ-VAE checkpoint found under {root}/"
        raise FileNotFoundError(msg)
    return candidates[0]


def load_frozen_vqvae(
    checkpoint_dir: str | Path,
    device: torch.device,
    seed: int = 42,
    run_name: str = "",
) -> tuple[VQVAE, dict[str, Any], Path]:
    """Load VQ-VAE weights, freeze, and return (model, saved_config, path)."""
    ckpt_path = resolve_vqvae_checkpoint(checkpoint_dir, seed, run_name)
    logger.info("Loading frozen VQ-VAE from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vqvae_cfg = ckpt.get("config", {})

    model = VQVAE(
        num_embeddings=int(vqvae_cfg.get("num_embeddings", 512)),
        embedding_dim=int(vqvae_cfg.get("embedding_dim", 64)),
        feature_map_size=int(vqvae_cfg.get("feature_map_size", 16)),
        commitment_cost=float(vqvae_cfg.get("commitment_cost", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.requires_grad_(False)
    return model, vqvae_cfg, ckpt_path
