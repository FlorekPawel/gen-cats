"""Load frozen VQ-VAE checkpoints for LDM and PixelCNN prior training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from gen_cats.config import (
    TrainConfig,
    checkpoint_run_slug,
    effective_vqvae_seed,
    vqvae_slug_config,
)
from gen_cats.models.vqvae import VQVAE

logger = logging.getLogger(__name__)


def _vqvae_path_from_ldm(checkpoint_dir: Path, seed: int) -> Path | None:
    """Reuse the VQ-VAE path stored on a Tiny LDM checkpoint for this seed."""
    root = checkpoint_dir / "tiny_ldm"
    if not root.exists():
        return None
    candidates = sorted(
        root.glob(f"**/best_seed{seed}.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for ckpt_path in candidates:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw = ckpt.get("vqvae_checkpoint")
        if raw:
            path = Path(raw)
            if path.exists():
                logger.info("Resolved VQ-VAE via Tiny LDM checkpoint %s", ckpt_path)
                return path
    return None


def resolve_vqvae_checkpoint(checkpoint_dir: str | Path, cfg: TrainConfig) -> Path:
    """Resolve ``best`` VQ-VAE weights for the current run's seed and grid cell.

    Lookup order:
    1. ``vqvae_run_name`` override
    2. Slug from VQ-VAE hyperparameters on ``cfg`` (same grid as ``make sweep-vae``)
    3. Path recorded on the matching ``tiny_ldm`` checkpoint (same ``seed``)
    4. Newest ``best_seed{seed}`` under ``checkpoints/vqvae/``
    """
    root = Path(checkpoint_dir)
    seed = effective_vqvae_seed(cfg)
    vqvae_root = root / "vqvae"

    if cfg.vqvae_run_name.strip():
        slug = checkpoint_run_slug(TrainConfig(model_type="vqvae", run_name=cfg.vqvae_run_name))
        candidate = vqvae_root / slug / f"best_seed{seed}.pt"
        if candidate.exists():
            return candidate

    slug = checkpoint_run_slug(vqvae_slug_config(cfg))
    candidate = vqvae_root / slug / f"best_seed{seed}.pt"
    if candidate.exists():
        logger.info("Resolved VQ-VAE by grid slug %s (seed=%d)", slug, seed)
        return candidate

    from_ldm = _vqvae_path_from_ldm(root, seed)
    if from_ldm is not None:
        return from_ldm

    def mtime_key(path: Path) -> float:
        return path.stat().st_mtime

    candidates = sorted(
        vqvae_root.glob(f"**/best_seed{seed}.pt"),
        key=mtime_key,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(vqvae_root.glob("**/best_seed*.pt"), key=mtime_key, reverse=True)
    if not candidates:
        msg = f"No VQ-VAE checkpoint found under {vqvae_root}/ (seed={seed}, slug={slug})"
        raise FileNotFoundError(msg)
    logger.warning(
        "VQ-VAE grid slug %s not found; using newest checkpoint %s",
        slug,
        candidates[0],
    )
    return candidates[0]


def load_frozen_vqvae(
    checkpoint_dir: str | Path,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[VQVAE, dict[str, Any], Path]:
    """Load VQ-VAE weights, freeze, and return (model, saved_config, path)."""
    ckpt_path = resolve_vqvae_checkpoint(checkpoint_dir, cfg)
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
