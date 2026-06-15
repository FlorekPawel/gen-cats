"""Tests for additional sample generation from best checkpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from gen_cats.config import TrainConfig, checkpoint_run_slug, config_to_dict
from gen_cats.evaluation.additional_samples import (
    additional_samples_path,
    discover_all_best_checkpoints,
    generate_additional_samples_for_checkpoint,
    save_sample_grid,
)


def _write_fake_ckpt(path: Path, *, model_type: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(model_type=model_type, seed=seed)
    torch.save(
        {
            "config": config_to_dict(cfg),
            "model": {},
            "optimizer": {},
        },
        path,
    )


class TestDiscoverAllBestCheckpoints:
    def test_finds_all_model_types_and_slugs(self, tmp_path: Path) -> None:
        for model_type in ("beta_vae", "wgan_gp"):
            for latent_dim in (64, 128):
                cfg = TrainConfig(model_type=model_type, latent_dim=latent_dim, seed=42)
                slug = checkpoint_run_slug(cfg)
                _write_fake_ckpt(
                    tmp_path / model_type / slug / "best_seed42.pt",
                    model_type=model_type,
                    seed=42,
                )

        found = discover_all_best_checkpoints(tmp_path)
        assert len(found) == 4
        assert all(p.name == "best_seed42.pt" for p in found)

    def test_ignores_non_best_names(self, tmp_path: Path) -> None:
        cfg = TrainConfig(model_type="beta_vae", seed=42)
        slug = checkpoint_run_slug(cfg)
        _write_fake_ckpt(
            tmp_path / "beta_vae" / slug / "latest_seed42.pt",
            model_type="beta_vae",
            seed=42,
        )
        assert discover_all_best_checkpoints(tmp_path) == []


class TestAdditionalSamplesPaths:
    def test_output_beside_checkpoint(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "beta_vae" / "abc" / "best_seed7.pt"
        expected = tmp_path / "beta_vae" / "abc" / "additional_samples.png"
        assert additional_samples_path(ckpt) == expected


class TestSaveSampleGrid:
    def test_writes_png(self, tmp_path: Path) -> None:
        samples = torch.randn(4, 3, 64, 64)
        out = save_sample_grid(samples, tmp_path / "additional_samples.png")
        assert out.is_file()


class TestGenerateAdditionalSamples:
    @patch("gen_cats.evaluation.additional_samples.load_trainer_from_checkpoint")
    def test_generates_with_offset_seed(self, mock_load: MagicMock, tmp_path: Path) -> None:
        cfg = TrainConfig(model_type="beta_vae", seed=42, device="cpu")
        slug = checkpoint_run_slug(cfg)
        ckpt = tmp_path / "beta_vae" / slug / "best_seed42.pt"
        _write_fake_ckpt(ckpt, model_type="beta_vae", seed=42)

        trainer = MagicMock()
        trainer.config = TrainConfig(
            model_type="beta_vae",
            seed=42,
            n_sample_images=8,
            device="cpu",
        )
        trainer.generate_samples.return_value = torch.randn(8, 3, 128, 128)
        mock_load.return_value = (trainer, ckpt)

        base = TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
        out = generate_additional_samples_for_checkpoint(
            ckpt,
            base,
            sample_seed_offset=10_000,
        )

        assert out == additional_samples_path(ckpt)
        assert out.is_file()
        trainer.seed_everything.assert_called_once_with(10_042)
        trainer.generate_samples.assert_called_once_with(8)

    @patch("gen_cats.evaluation.additional_samples.load_trainer_from_checkpoint")
    def test_skip_existing(self, mock_load: MagicMock, tmp_path: Path) -> None:
        cfg = TrainConfig(model_type="beta_vae", seed=42, device="cpu")
        slug = checkpoint_run_slug(cfg)
        ckpt = tmp_path / "beta_vae" / slug / "best_seed42.pt"
        _write_fake_ckpt(ckpt, model_type="beta_vae", seed=42)
        additional_samples_path(ckpt).write_bytes(b"png")

        base = TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
        out = generate_additional_samples_for_checkpoint(ckpt, base, skip_existing=True)

        assert out is None
        mock_load.assert_not_called()
