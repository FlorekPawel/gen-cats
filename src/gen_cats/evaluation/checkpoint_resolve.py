"""Resolve trained checkpoints for evaluation (no training-side slug guesswork)."""

from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from gen_cats.config import TrainConfig, checkpoint_run_slug
from gen_cats.factory import create_trainer
from gen_cats.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def resolve_best_checkpoint(
    checkpoint_dir: str | Path,
    model_type: str,
    seed: int,
    *,
    run_name: str = "",
    tag: str = "best",
) -> Path | None:
    """Find ``{tag}_seed{seed}.pt`` under ``checkpoints/<model_type>/``.

    Lookup order:
    1. ``run_name`` slug (if set)
    2. Default-hyperparameter slug
    3. Newest ``**/{tag}_seed{seed}.pt`` under the model folder (sweep grids)
    """
    root = Path(checkpoint_dir) / model_type
    if not root.is_dir():
        return None

    filename = f"{tag}_seed{seed}.pt"

    if run_name.strip():
        slug = checkpoint_run_slug(TrainConfig(model_type=model_type, run_name=run_name))
        path = root / slug / filename
        if path.is_file():
            return path

    slug = checkpoint_run_slug(TrainConfig(model_type=model_type, seed=seed, run_name=run_name))
    path = root / slug / filename
    if path.is_file():
        return path

    candidates = sorted(
        root.glob(f"**/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.info(
            "Multiple %s %s for seed=%d; using newest %s",
            model_type,
            filename,
            seed,
            candidates[0],
        )
    return candidates[0]


def config_from_checkpoint(ckpt: dict[str, Any], base: TrainConfig) -> TrainConfig:
    """Rebuild ``TrainConfig`` from checkpoint payload (architecture + training hparams)."""
    saved = ckpt.get("config") or {}
    merged = {f.name: getattr(base, f.name) for f in fields(base)}
    keep_from_base = frozenset({"device", "data_dir", "checkpoint_dir"})
    for key, value in saved.items():
        if key in merged and key not in keep_from_base:
            merged[key] = value
    merged["seed"] = base.seed
    merged["device"] = base.device
    merged["data_dir"] = base.data_dir
    merged["checkpoint_dir"] = base.checkpoint_dir
    if base.run_name.strip():
        merged["run_name"] = base.run_name
    return TrainConfig(**merged)


def load_trainer_for_eval(
    cfg: TrainConfig,
    *,
    tag: str = "best",
) -> tuple[BaseTrainer, Path] | None:
    """Load a trainer for inference/FID using an on-disk best (or latest) checkpoint."""
    ckpt_path = resolve_best_checkpoint(
        cfg.checkpoint_dir,
        cfg.model_type,
        cfg.seed,
        run_name=cfg.run_name,
        tag=tag,
    )
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eval_cfg = config_from_checkpoint(ckpt, cfg)
    trainer = create_trainer(eval_cfg)
    trainer._ckpt_dir = ckpt_path.parent
    trainer.build_models()
    trainer.build_optimizers()

    if not trainer.load_checkpoint(tag, weights_only=True):
        trainer.load_state_dicts(ckpt)

    logger.info("Eval loaded %s from %s", cfg.model_type, ckpt_path)
    return trainer, ckpt_path
