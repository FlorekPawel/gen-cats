"""Factory functions for creating models, optimizers, datasets, and trainers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from gen_cats.config import TrainConfig
from gen_cats.data.cat_dataset import CatFaceDataset

if TYPE_CHECKING:
    from gen_cats.training.base_trainer import BaseTrainer


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Create train and val DataLoaders from processed .npy files."""
    data_dir = Path(config.data_dir)
    train_ds = CatFaceDataset(data_dir / "train.npy", augment=config.augment)
    val_ds = CatFaceDataset(data_dir / "val.npy", augment=False)

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=False,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=False,
    )
    return train_loader, val_loader


def create_optimizer(
    params: Any,
    lr: float,
    betas: tuple[float, float] = (0.5, 0.999),
) -> torch.optim.Adam:
    """Standard Adam optimizer used across all model families."""
    return torch.optim.Adam(params, lr=lr, betas=betas)


def create_trainer(config: TrainConfig) -> BaseTrainer:
    """Factory: config.model_type → appropriate Trainer subclass."""
    from gen_cats.training.dm_trainer import DiffusionTrainer
    from gen_cats.training.gan_trainer import GANTrainer
    from gen_cats.training.pixelcnn_trainer import PixelCNNTrainer
    from gen_cats.training.vae_trainer import VAETrainer

    registry: dict[str, type[BaseTrainer]] = {
        "beta_vae": VAETrainer,
        "vqvae": VAETrainer,
        "wgan_gp": GANTrainer,
        "sn_gan": GANTrainer,
        "ddim": DiffusionTrainer,
        "tiny_ldm": DiffusionTrainer,
        "pixelcnn": PixelCNNTrainer,
    }

    cls = registry.get(config.model_type)
    if cls is None:
        msg = f"Unknown model_type: {config.model_type}. Available: {list(registry.keys())}"
        raise ValueError(msg)

    return cls(config)
