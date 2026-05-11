"""Chimera experiment: train SN-GAN on mixed Dogs vs Cats dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from gen_cats.config import TrainConfig
from gen_cats.data.cat_dataset import CatFaceDataset
from gen_cats.factory import create_trainer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimera: SN-GAN on dogs+cats")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npy_path = Path(args.data_dir) / "dogcat_train.npy"
    if not npy_path.exists():
        logger.error(
            "dogcat_train.npy not found. Run: python scripts/process_data.py --dogcat first."
        )
        return

    ds = CatFaceDataset(npy_path, augment=True)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val

    rng = np.random.default_rng(args.seed)
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
        model_type="sn_gan",
        seed=args.seed,
        device=args.device,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.checkpoint_dir + "/chimera",
        experiment_name="chimera-dogcat",
        batch_size=64,
    )

    trainer = create_trainer(cfg)
    logger.info("Starting chimera SN-GAN training on %d dog+cat images", len(ds))
    results = trainer.fit(train_loader, val_loader)
    logger.info("Chimera training complete: %s", results)


if __name__ == "__main__":
    main()
