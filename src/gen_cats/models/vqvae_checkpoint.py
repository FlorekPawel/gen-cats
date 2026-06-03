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
from gen_cats.models.vqvae_prior_selection import (
    VqvaeSelection,
    resolve_vqvae_from_manifest,
)

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


def _resolve_vqvae_by_slug(
    vqvae_root: Path,
    cfg: TrainConfig,
    seed: int,
    *,
    strict: bool,
) -> Path | None:
    if cfg.vqvae_run_name.strip():
        run_slug = checkpoint_run_slug(TrainConfig(model_type="vqvae", run_name=cfg.vqvae_run_name))
        candidate = vqvae_root / run_slug / f"best_seed{seed}.pt"
        if candidate.exists():
            logger.info("Resolved VQ-VAE by vqvae_run_name slug %s", run_slug)
            return candidate
        if strict:
            raise FileNotFoundError(
                f"VQ-VAE not found for vqvae_run_name={cfg.vqvae_run_name!r} at {candidate}"
            )
        return None

    slug = checkpoint_run_slug(vqvae_slug_config(cfg))
    candidate = vqvae_root / slug / f"best_seed{seed}.pt"
    if candidate.exists():
        logger.info("Resolved VQ-VAE by grid slug %s (seed=%d)", slug, seed)
        return candidate

    if strict:
        msg = (
            f"VQ-VAE slug {slug!r} (seed={seed}) not found at {candidate}; "
            f"refusing fallback (require_vqvae_slug=True). "
            f"Expected hyperparameters: num_embeddings={cfg.num_embeddings}, "
            f"feature_map_size={cfg.feature_map_size}, recon_loss={cfg.recon_loss!r}. "
            f"Run `make select-vqvae-priors` after sweep-vae or set vqvae_selection=auto."
        )
        raise FileNotFoundError(msg)
    return None


def _resolve_vqvae_fallback(vqvae_root: Path, seed: int) -> Path:
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
        msg = f"No VQ-VAE checkpoint for seed={seed} under {vqvae_root}/."
        raise FileNotFoundError(msg)
    logger.warning("VQ-VAE fallback: using newest checkpoint %s", candidates[0])
    return candidates[0]


def resolve_vqvae_checkpoint(
    checkpoint_dir: str | Path,
    cfg: TrainConfig,
    *,
    strict: bool | None = None,
) -> Path:
    """Resolve ``best`` VQ-VAE weights for the current run's seed.

    ``cfg.vqvae_selection``:

    - **manifest** — ``checkpoints/vqvae/prior_best_by_seed.json`` (from
      ``make select-vqvae-priors``); per seed, lowest val ``recon`` across the sweep grid.
    - **slug** — hyperparameters on ``cfg`` (``NUM_EMBEDDINGS``, etc.).
    - **auto** — manifest if present, else slug, else optional fallback.

    Also honors ``vqvae_run_name`` before manifest/slug when set.
    """
    if strict is None:
        strict = cfg.require_vqvae_slug

    root = Path(checkpoint_dir)
    seed = effective_vqvae_seed(cfg)
    vqvae_root = root / "vqvae"
    selection: VqvaeSelection = cfg.vqvae_selection  # type: ignore[assignment]
    if selection not in ("slug", "manifest", "auto"):
        raise ValueError(f"Unknown vqvae_selection={cfg.vqvae_selection!r}")

    if cfg.vqvae_run_name.strip():
        path = _resolve_vqvae_by_slug(vqvae_root, cfg, seed, strict=True)
        if path is not None:
            return path

    if selection in ("manifest", "auto"):
        manifest_path = resolve_vqvae_from_manifest(checkpoint_dir, cfg)
        if manifest_path is not None:
            return manifest_path
        if selection == "manifest":
            raise FileNotFoundError(
                f"No VQ-VAE prior manifest entry for seed={seed}. "
                f"Run `make select-vqvae-priors` after `make sweep-vae`."
            )

    slug_path = _resolve_vqvae_by_slug(vqvae_root, cfg, seed, strict=strict or selection == "slug")
    if slug_path is not None:
        return slug_path

    if selection == "auto" and not strict:
        from_ldm = _vqvae_path_from_ldm(root, seed)
        if from_ldm is not None:
            return from_ldm
        return _resolve_vqvae_fallback(vqvae_root, seed)

    slug = checkpoint_run_slug(vqvae_slug_config(cfg))
    msg = (
        f"No VQ-VAE checkpoint for slug={slug!r} seed={seed} under {vqvae_root}/. "
        f"Train VQ-VAE, run `make select-vqvae-priors`, or pass matching "
        f"num_embeddings / feature_map_size / recon_loss."
    )
    raise FileNotFoundError(msg)


def load_frozen_vqvae(
    checkpoint_dir: str | Path,
    device: torch.device,
    cfg: TrainConfig,
    *,
    strict: bool | None = None,
) -> tuple[VQVAE, dict[str, Any], Path]:
    """Load VQ-VAE weights, freeze, and return (model, saved_config, path)."""
    ckpt_path = resolve_vqvae_checkpoint(checkpoint_dir, cfg, strict=strict)
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
