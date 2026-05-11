"""Experiment configuration dataclasses with grid search support."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class TrainConfig:
    """Base training hyperparameters shared across all model families."""

    model_type: str = "beta_vae"
    batch_size: int = 64
    lr: float = 2e-4
    lr_d: float | None = None
    lr_g: float | None = None
    max_epochs: int = 100
    seed: int = 42
    data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    device: str = "mps"
    num_workers: int = 0
    log_interval: int = 1
    sample_interval: int = 5
    n_sample_images: int = 16
    augment: bool = True
    experiment_name: str = "gen-cats"
    run_name: str = ""

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    early_stop_metric: str = "val_loss"

    # Model-specific params stored as flat fields
    # VAE
    latent_dim: int = 128
    beta: float = 1.0
    recon_loss: str = "mse"

    # VQ-VAE
    num_embeddings: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25
    feature_map_size: int = 16

    # GAN
    n_critic: int = 5
    gp_lambda: float = 10.0
    use_spectral_norm: bool = False
    d_augment: bool = False

    # Diffusion
    timesteps: int = 1000
    noise_schedule: str = "linear"
    base_channels: int = 64
    ddim_steps: int = 50
    use_ema: bool = False
    ema_decay: float = 0.999


def config_grid(base: TrainConfig, grid: dict[str, list[Any]]) -> list[TrainConfig]:
    """Expand base config over a parameter grid → list of configs.

    Example:
        grid = {"latent_dim": [64, 128], "beta": [1.0, 4.0]}
        → 4 configs with all combinations
    """
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo, strict=True))
        cfg_dict = {f.name: getattr(base, f.name) for f in fields(base)}
        cfg_dict.update(overrides)
        configs.append(TrainConfig(**cfg_dict))
    return configs


def config_grid_with_seeds(
    base: TrainConfig,
    grid: dict[str, list[Any]],
    seeds: list[int] | None = None,
) -> list[TrainConfig]:
    """Expand grid and multiply by seeds → full experiment matrix."""
    if seeds is None:
        seeds = [42, 123, 7]
    param_configs = config_grid(base, grid)
    all_configs = []
    for cfg in param_configs:
        for seed in seeds:
            cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
            cfg_dict["seed"] = seed
            all_configs.append(TrainConfig(**cfg_dict))
    return all_configs


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Flat dict for MLflow param logging."""
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


SEEDS: list[int] = [42, 0, 3407]

GRIDS: dict[str, dict[str, list[Any]]] = {
    "beta_vae": {
        "latent_dim": [64, 128],
        "beta": [1.0, 4.0],
    },
    "vqvae": {
        "num_embeddings": [512, 1024],
        "feature_map_size": [16, 8],
        "recon_loss": ["l1", "mse"],
    },
    "wgan_gp": {
        "n_critic": [3, 5],
        "batch_size": [64, 128],
        "lr": [2e-4],
        "lr_d": [2e-4, 4e-4],
    },
    "sn_gan": {
        "batch_size": [64, 128],
        "lr": [2e-4],
        "lr_d": [2e-4, 4e-4],
        "d_augment": [False, True],
    },
    "ddim": {
        "noise_schedule": ["linear", "cosine"],
        "base_channels": [32, 64],
    },
}
