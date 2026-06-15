"""Generate extra sample grids from every ``best_seed*.pt`` checkpoint."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from gen_cats.config import TrainConfig
from gen_cats.evaluation.checkpoint_resolve import (
    load_trainer_from_checkpoint,
    seed_from_checkpoint_name,
)
from gen_cats.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

ADDITIONAL_SAMPLES_FILENAME = "additional_samples.png"


def discover_all_best_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """List every ``best_seed{N}.pt`` under ``checkpoint_dir`` (all model types / grid cells)."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return []

    paths: list[Path] = []
    for path in sorted(root.glob("**/best_seed*.pt")):
        if seed_from_checkpoint_name(path) is None:
            continue
        paths.append(path)
    return paths


def additional_samples_path(ckpt_path: Path) -> Path:
    """PNG path beside ``best_seed{N}.pt``."""
    return ckpt_path.parent / ADDITIONAL_SAMPLES_FILENAME


def save_sample_grid(
    samples: torch.Tensor,
    output_path: str | Path,
    *,
    nrow: int = 4,
) -> Path:
    """Write an (N, C, H, W) tensor grid in [-1, 1] to a PNG file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(path)
    return path


def _seed_for_generation(train_seed: int, sample_seed_offset: int) -> int:
    return train_seed + sample_seed_offset


def generate_additional_samples_for_checkpoint(
    ckpt_path: Path,
    base: TrainConfig,
    *,
    n_samples: int | None = None,
    sample_seed_offset: int = 10_000,
    skip_existing: bool = False,
) -> Path | None:
    """Load ``best`` weights from ``ckpt_path`` and write ``additional_samples.png``."""
    out_path = additional_samples_path(ckpt_path)
    if skip_existing and out_path.is_file():
        logger.info("Skip existing %s", out_path)
        return None

    train_seed = seed_from_checkpoint_name(ckpt_path)
    if train_seed is None:
        logger.warning("Unrecognized checkpoint name %s", ckpt_path.name)
        return None

    trainer, _ = load_trainer_from_checkpoint(ckpt_path, base)
    gen_seed = _seed_for_generation(train_seed, sample_seed_offset)
    _seed_everything(trainer, gen_seed)

    count = n_samples if n_samples is not None else trainer.config.n_sample_images
    with torch.no_grad():
        samples = trainer.generate_samples(count)

    save_sample_grid(samples, out_path)
    logger.info("Saved %d additional samples to %s (gen_seed=%d)", count, out_path, gen_seed)
    return out_path


def generate_all_additional_samples(
    checkpoint_dir: str | Path,
    *,
    device: str = "mps",
    data_dir: str = "data/processed",
    n_samples: int | None = None,
    sample_seed_offset: int = 10_000,
    skip_existing: bool = False,
) -> list[Path]:
    """Scan ``checkpoint_dir`` and generate ``additional_samples.png`` for each ``best`` ckpt."""
    base = TrainConfig(device=device, data_dir=data_dir, checkpoint_dir=str(checkpoint_dir))
    written: list[Path] = []

    for ckpt_path in discover_all_best_checkpoints(checkpoint_dir):
        try:
            out = generate_additional_samples_for_checkpoint(
                ckpt_path,
                base,
                n_samples=n_samples,
                sample_seed_offset=sample_seed_offset,
                skip_existing=skip_existing,
            )
            if out is not None:
                written.append(out)
        except Exception:
            logger.exception("Failed additional samples for %s", ckpt_path)

    return written


def _seed_everything(trainer: BaseTrainer, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    trainer.seed_everything(seed)
