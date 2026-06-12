"""FID evaluation across all trained model families."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

import numpy as np
import torch

from gen_cats.config import TrainConfig
from gen_cats.evaluation.checkpoint_resolve import load_trainer_for_eval
from gen_cats.evaluation.fid import compute_fid_from_loaders
from gen_cats.factory import create_dataloaders

logger = logging.getLogger(__name__)

MODEL_TYPES: list[str] = [
    "beta_vae",
    "vqvae",
    "pixelcnn",
    "tiny_ldm",
    "wgan_gp",
    "sn_gan",
    "ddim",
]

VQVAE_LINKED_MODELS = frozenset({"vqvae", "pixelcnn", "tiny_ldm"})

VQVAE_GRID_FIELDS = (
    "num_embeddings",
    "feature_map_size",
    "recon_loss",
    "embedding_dim",
    "commitment_cost",
)


def build_eval_config(
    model_type: str,
    seed: int,
    *,
    device: str,
    data_dir: str,
    checkpoint_dir: str,
    run_name: str = "",
    vqvae_overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Build ``TrainConfig`` for FID evaluation of one model and seed."""
    cfg = TrainConfig(
        model_type=model_type,
        seed=seed,
        device=device,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        vqvae_seed=None,
        vqvae_selection="auto",
    )
    if vqvae_overrides and model_type in VQVAE_LINKED_MODELS:
        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        cfg_dict.update(vqvae_overrides)
        return TrainConfig(**cfg_dict)
    return cfg


def evaluate_model(
    model_type: str,
    seeds: list[int],
    *,
    device: str,
    data_dir: str,
    checkpoint_dir: str,
    run_name: str = "",
    n_samples: int = 1000,
    vqvae_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute FID for a model type across seeds; return summary dict."""
    per_seed: dict[str, float] = {}

    for seed in seeds:
        cfg = build_eval_config(
            model_type,
            seed,
            device=device,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            vqvae_overrides=vqvae_overrides,
        )

        try:
            loaded = load_trainer_for_eval(cfg)
            if loaded is None:
                logger.warning("No best checkpoint for %s seed=%d, skipping", model_type, seed)
                continue
            trainer, _ = loaded
            _train_loader, val_loader = create_dataloaders(trainer.config)

            def gen_fn(n: int, _t: object = trainer) -> torch.Tensor:
                return _t.generate_samples(n).cpu()  # type: ignore[attr-defined]

            fid = compute_fid_from_loaders(
                val_loader,
                gen_fn,
                n_samples=n_samples,
                device=torch.device(cfg.device),
            )
            per_seed[str(seed)] = fid
            logger.info("FID %s seed=%d: %.2f", model_type, seed, fid)

        except Exception:
            logger.exception("Failed to evaluate %s seed=%d", model_type, seed)

    if not per_seed:
        return {
            "model": model_type,
            "mean_fid": float("nan"),
            "std_fid": float("nan"),
        }

    fid_scores = list(per_seed.values())
    result: dict[str, Any] = {
        "model": model_type,
        "seeds": list(per_seed.keys()),
        "per_seed": per_seed,
        "mean_fid": float(np.mean(fid_scores)),
        "std_fid": float(np.std(fid_scores)),
        "fid_scores": fid_scores,
    }
    if model_type == "vqvae":
        result["note"] = "Uses random codebook indices in generate_samples; not a learned prior."
    return result


def evaluate_all(
    model_types: list[str],
    seeds: list[int],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run FID for each model type."""
    return [evaluate_model(mt, seeds, **kwargs) for mt in model_types]
