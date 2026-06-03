"""Tests for per-seed best VQ-VAE selection manifest."""

from __future__ import annotations

from pathlib import Path

import torch

from gen_cats.config import TrainConfig, checkpoint_run_slug, vqvae_slug_config
from gen_cats.models.vqvae_checkpoint import resolve_vqvae_checkpoint
from gen_cats.models.vqvae_prior_selection import (
    build_vqvae_prior_manifest,
    manifest_path,
    save_vqvae_prior_manifest,
    select_best_vqvae_for_seed,
)


def _write_vqvae_ckpt(
    path: Path,
    *,
    seed: int,
    best_metric: float,
    num_embeddings: int = 512,
    feature_map_size: int = 16,
    recon_loss: str = "mse",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "model_type": "vqvae",
                "seed": seed,
                "num_embeddings": num_embeddings,
                "embedding_dim": 64,
                "feature_map_size": feature_map_size,
                "commitment_cost": 0.25,
                "recon_loss": recon_loss,
            },
            "train_state": {"best_metric": best_metric},
            "model": {},
        },
        path,
    )


class TestVqvaePriorSelection:
    def test_picks_lowest_val_recon_per_seed(self, tmp_path: Path) -> None:
        root = tmp_path / "vqvae"
        slug_a = checkpoint_run_slug(
            vqvae_slug_config(
                TrainConfig(model_type="vqvae", num_embeddings=512, feature_map_size=16)
            )
        )
        slug_b = checkpoint_run_slug(
            vqvae_slug_config(
                TrainConfig(
                    model_type="vqvae", num_embeddings=1024, feature_map_size=8, recon_loss="l1"
                )
            )
        )
        _write_vqvae_ckpt(root / slug_a / "best_seed42.pt", seed=42, best_metric=0.20)
        _write_vqvae_ckpt(
            root / slug_b / "best_seed42.pt", seed=42, best_metric=0.10, num_embeddings=1024
        )

        picked = select_best_vqvae_for_seed(tmp_path, 42)
        assert picked is not None
        assert picked.best_metric == 0.10
        assert picked.num_embeddings == 1024
        assert picked.slug == slug_b

    def test_manifest_resolve_auto(self, tmp_path: Path) -> None:
        root = tmp_path / "vqvae"
        slug = checkpoint_run_slug(
            vqvae_slug_config(TrainConfig(model_type="vqvae", num_embeddings=256))
        )
        _write_vqvae_ckpt(
            root / slug / "best_seed7.pt",
            seed=7,
            best_metric=0.05,
            num_embeddings=256,
            feature_map_size=8,
        )
        save_vqvae_prior_manifest(tmp_path, [7])

        cfg = TrainConfig(
            model_type="pixelcnn",
            seed=7,
            checkpoint_dir=str(tmp_path),
            vqvae_selection="auto",
            num_embeddings=512,
            feature_map_size=16,
        )
        path = resolve_vqvae_checkpoint(tmp_path, cfg)
        assert path == root / slug / "best_seed7.pt"

    def test_build_manifest_skips_missing_seed(self, tmp_path: Path) -> None:
        root = tmp_path / "vqvae"
        slug = checkpoint_run_slug(vqvae_slug_config(TrainConfig(model_type="vqvae")))
        _write_vqvae_ckpt(root / slug / "best_seed42.pt", seed=42, best_metric=0.1)
        payload = build_vqvae_prior_manifest(tmp_path, [42, 99])
        assert "42" in payload["seeds"]
        assert "99" not in payload["seeds"]
        assert manifest_path(tmp_path).exists() is False
