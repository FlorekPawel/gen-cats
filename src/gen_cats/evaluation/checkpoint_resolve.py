"""Resolve trained checkpoints for evaluation (no training-side slug guesswork)."""

from __future__ import annotations

import logging
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from gen_cats.config import GRIDS, TrainConfig, checkpoint_run_slug, config_to_dict
from gen_cats.factory import create_trainer
from gen_cats.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

_CKPT_SEED_RE = re.compile(r"^(?:best|latest)_seed(\d+)$")


def _seed_from_checkpoint_name(path: Path) -> int | None:
    match = _CKPT_SEED_RE.match(path.stem)
    if not match:
        return None
    return int(match.group(1))


def discover_checkpoints(
    checkpoint_dir: str | Path,
    model_type: str,
    seeds: list[int] | None = None,
    *,
    run_name: str = "",
    tag: str = "best",
) -> list[Path]:
    """List every ``{tag}_seed{N}.pt`` under ``checkpoints/<model_type>/`` (all grid cells)."""
    root = Path(checkpoint_dir) / model_type
    if not root.is_dir():
        return []

    seed_filter = set(seeds) if seeds is not None else None
    paths: list[Path] = []

    if run_name.strip():
        slug = checkpoint_run_slug(TrainConfig(model_type=model_type, run_name=run_name))
        search_root = root / slug
        if not search_root.is_dir():
            return []
        glob_roots = [search_root]
    else:
        glob_roots = [root]

    for search in glob_roots:
        for path in search.glob(f"**/{tag}_seed*.pt"):
            seed = _seed_from_checkpoint_name(path)
            if seed is None:
                continue
            if seed_filter is not None and seed not in seed_filter:
                continue
            paths.append(path)

    return sorted(paths, key=lambda p: (p.parent.name, seed or 0))


def resolve_best_checkpoint(
    checkpoint_dir: str | Path,
    model_type: str,
    seed: int,
    *,
    run_name: str = "",
    tag: str = "best",
) -> Path | None:
    """Resolve a single checkpoint (exact slug / run_name only — no newest fallback)."""
    paths = discover_checkpoints(
        checkpoint_dir,
        model_type,
        [seed],
        run_name=run_name,
        tag=tag,
    )
    if not paths:
        return None
    if len(paths) > 1:
        logger.warning(
            "Multiple %s checkpoints for seed=%d; using %s (use full grid eval for all)",
            model_type,
            seed,
            paths[0],
        )
    return paths[0]


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


def run_hyperparameters(cfg: TrainConfig) -> dict[str, Any]:
    """Grid-relevant hyperparameters for one checkpoint run (for FID tables)."""
    grid_keys = list(GRIDS.get(cfg.model_type, {}).keys())
    if not grid_keys:
        grid_keys = [
            "latent_dim",
            "beta",
            "num_embeddings",
            "feature_map_size",
            "recon_loss",
            "n_critic",
            "batch_size",
            "noise_schedule",
            "base_channels",
            "ddim_steps",
        ]
    full = config_to_dict(cfg)
    return {k: full[k] for k in grid_keys if k in full}


def load_trainer_from_checkpoint(
    ckpt_path: Path,
    base: TrainConfig,
    *,
    tag: str = "best",
) -> tuple[BaseTrainer, Path]:
    """Load trainer from an explicit checkpoint file path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    seed = _seed_from_checkpoint_name(ckpt_path)
    eval_cfg = config_from_checkpoint(ckpt, base)
    if seed is not None:
        eval_cfg = TrainConfig(
            **{**config_to_dict(eval_cfg), "seed": seed},
        )
    trainer = create_trainer(eval_cfg)
    trainer._ckpt_dir = ckpt_path.parent
    trainer.build_models()
    trainer.build_optimizers()

    if not trainer.load_checkpoint(tag, weights_only=True):
        trainer.load_state_dicts(ckpt)

    logger.info("Eval loaded %s seed=%s from %s", eval_cfg.model_type, eval_cfg.seed, ckpt_path)
    return trainer, ckpt_path


def load_trainer_for_eval(
    cfg: TrainConfig,
    *,
    tag: str = "best",
) -> tuple[BaseTrainer, Path] | None:
    """Load one trainer (first grid cell if several exist for this seed)."""
    ckpt_path = resolve_best_checkpoint(
        cfg.checkpoint_dir,
        cfg.model_type,
        cfg.seed,
        run_name=cfg.run_name,
        tag=tag,
    )
    if ckpt_path is None:
        return None
    return load_trainer_from_checkpoint(ckpt_path, cfg, tag=tag)
