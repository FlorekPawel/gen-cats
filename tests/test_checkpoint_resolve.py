"""Tests for evaluation checkpoint resolution."""

from __future__ import annotations

from pathlib import Path

import torch

from gen_cats.config import TrainConfig, checkpoint_run_slug, config_to_dict
from gen_cats.evaluation.checkpoint_resolve import (
    config_from_checkpoint,
    discover_checkpoints,
    resolve_best_checkpoint,
)


def _write_fake_ckpt(path: Path, *, model_type: str, seed: int, latent_dim: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(model_type=model_type, seed=seed, latent_dim=latent_dim)
    torch.save(
        {
            "config": config_to_dict(cfg),
            "model": {},
            "optimizer": {},
        },
        path,
    )


class TestResolveBestCheckpoint:
    def test_discover_all_grid_cells(self, tmp_path: Path) -> None:
        for latent_dim in (64, 128):
            cfg = TrainConfig(model_type="beta_vae", latent_dim=latent_dim, seed=42)
            slug = checkpoint_run_slug(cfg)
            _write_fake_ckpt(
                tmp_path / "beta_vae" / slug / "best_seed42.pt",
                model_type="beta_vae",
                seed=42,
                latent_dim=latent_dim,
            )

        found = discover_checkpoints(tmp_path, "beta_vae", [42])
        assert len(found) == 2
        assert {p.parent.name for p in found} == {
            checkpoint_run_slug(TrainConfig(model_type="beta_vae", latent_dim=64, seed=42)),
            checkpoint_run_slug(TrainConfig(model_type="beta_vae", latent_dim=128, seed=42)),
        }

    def test_resolve_single_slug_when_multiple_exist(self, tmp_path: Path) -> None:
        for latent_dim in (64, 128):
            cfg = TrainConfig(model_type="beta_vae", latent_dim=latent_dim, seed=42)
            slug = checkpoint_run_slug(cfg)
            _write_fake_ckpt(
                tmp_path / "beta_vae" / slug / "best_seed42.pt",
                model_type="beta_vae",
                seed=42,
                latent_dim=latent_dim,
            )

        found = resolve_best_checkpoint(tmp_path, "beta_vae", 42)
        assert found is not None
        assert found in discover_checkpoints(tmp_path, "beta_vae", [42])

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert resolve_best_checkpoint(tmp_path, "ddim", 42) is None
        assert discover_checkpoints(tmp_path, "ddim", [42]) == []

    def test_config_from_checkpoint_restores_latent_dim(self) -> None:
        base = TrainConfig(model_type="beta_vae", seed=42, latent_dim=128)
        ckpt = {"config": config_to_dict(TrainConfig(model_type="beta_vae", latent_dim=64, seed=7))}
        restored = config_from_checkpoint(ckpt, base)
        assert restored.latent_dim == 64
        assert restored.seed == 42
