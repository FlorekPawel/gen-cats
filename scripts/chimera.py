"""Chimera experiment: train 64x64 WGAN-GP on mixed Dogs vs Cats (all project seeds)."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gen_cats.config import CHIMERA_IMAGE_SIZE, SEEDS, TrainConfig
from gen_cats.data.cat_dataset import CatFaceDataset
from gen_cats.data.dogcat_dataset import dogcat_npy_path
from gen_cats.factory import create_trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_chimera(
    seed: int,
    *,
    data_dir: str,
    checkpoint_dir: str,
    device: str,
    max_epochs: int,
    image_size: int = CHIMERA_IMAGE_SIZE,
) -> dict[str, Any]:
    """Train chimera WGAN-GP at ``image_size`` (default 64) for one seed."""
    npy_path = dogcat_npy_path(data_dir, image_size)
    if not npy_path.is_file():
        msg = (
            f"{npy_path} not found. Run: make process-dogcat-chimera "
            f"(or process_data.py --dataset dogcat --size {image_size})"
        )
        raise FileNotFoundError(msg)

    ds = CatFaceDataset(npy_path, augment=True)
    if ds.data.shape[1] != image_size or ds.data.shape[2] != image_size:
        msg = (
            f"Expected {image_size}x{image_size} images in {npy_path}, "
            f"got {ds.data.shape[1]}x{ds.data.shape[2]}"
        )
        raise ValueError(msg)

    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()

    train_subset = torch.utils.data.Subset(ds, train_indices)
    val_subset = torch.utils.data.Subset(ds, val_indices)

    train_loader: DataLoader[torch.Tensor] = DataLoader(
        train_subset, batch_size=64, shuffle=True, drop_last=True
    )
    val_loader: DataLoader[torch.Tensor] = DataLoader(val_subset, batch_size=64, shuffle=False)

    cfg = TrainConfig(
        model_type="wgan_gp",
        seed=seed,
        device=device,
        max_epochs=max_epochs,
        checkpoint_dir=checkpoint_dir + "/chimera",
        experiment_name="chimera-dogcat",
        batch_size=64,
        image_size=image_size,
        run_name=f"chimera_{image_size}_seed{seed}",
    )

    trainer = create_trainer(cfg)
    logger.info(
        "Chimera WGAN-GP seed=%d on %d dog+cat images at %dx%d",
        seed,
        len(ds),
        image_size,
        image_size,
    )
    results = trainer.fit(train_loader, val_loader)
    return {"seed": seed, **results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimera: 64x64 WGAN-GP on dogs+cats")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--image-size",
        type=int,
        default=CHIMERA_IMAGE_SIZE,
        choices=[64, 128],
        help="must match processed dogcat .npy (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed only (default: all SEEDS)",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS
    all_results: list[dict[str, Any]] = []

    for seed in seeds:
        try:
            result = run_chimera(
                seed,
                data_dir=args.data_dir,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
                max_epochs=args.max_epochs,
                image_size=args.image_size,
            )
            all_results.append(result)
            logger.info("Chimera seed=%d complete: %s", seed, result)
        except Exception:
            logger.exception("Chimera failed for seed=%d", seed)
            all_results.append({"seed": seed, "status": "failed"})

    n_ok = sum(1 for r in all_results if r.get("status") != "failed")
    logger.info("Chimera finished: %d/%d seeds successful", n_ok, len(seeds))


if __name__ == "__main__":
    main()
