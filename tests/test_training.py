"""Tests for training infrastructure: config, early stopping, base trainer, experiment runner."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import torch
from gen_cats.config import (
    TrainConfig,
    checkpoint_run_slug,
    config_grid,
    config_grid_with_seeds,
    config_to_dict,
    effective_vqvae_seed,
    vqvae_slug_config,
)
from gen_cats.training.base_trainer import BaseTrainer
from gen_cats.training.early_stopping import EarlyStopping
from gen_cats.training.experiment_runner import ExperimentRunner
from torch.utils.data import DataLoader, TensorDataset


class TestTrainConfig:
    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.max_epochs == 1000
        assert cfg.patience == 15
        assert cfg.device == "mps"
        assert cfg.sample_interval == 5
        assert cfg.min_epochs == 20
        assert cfg.prior_n_layers == 10

    def test_override(self) -> None:
        cfg = TrainConfig(batch_size=32, lr=1e-3)
        assert cfg.batch_size == 32
        assert cfg.lr == 1e-3

    def test_effective_vqvae_seed_matches_run(self) -> None:
        cfg = TrainConfig(model_type="pixelcnn", seed=123, vqvae_seed=None)
        assert effective_vqvae_seed(cfg) == 123

    def test_vqvae_slug_ignores_pixelcnn_fields(self) -> None:
        a = vqvae_slug_config(
            TrainConfig(model_type="pixelcnn", prior_n_layers=20, num_embeddings=512)
        )
        b = vqvae_slug_config(TrainConfig(model_type="vqvae", prior_n_layers=5, num_embeddings=512))
        assert checkpoint_run_slug(a) == checkpoint_run_slug(b)


class TestCheckpointRunSlug:
    def test_differs_by_hparams(self) -> None:
        a = TrainConfig(model_type="beta_vae", latent_dim=64, seed=42)
        b = TrainConfig(model_type="beta_vae", latent_dim=128, seed=42)
        assert checkpoint_run_slug(a) != checkpoint_run_slug(b)

    def test_stable_for_same_config(self) -> None:
        c = TrainConfig(model_type="wgan_gp", latent_dim=64, seed=7)
        assert checkpoint_run_slug(c) == checkpoint_run_slug(
            TrainConfig(model_type="wgan_gp", latent_dim=64, seed=7)
        )

    def test_run_name_override(self) -> None:
        d = TrainConfig(run_name="sweep/run-a")
        assert checkpoint_run_slug(d) == "sweep_run-a"


class TestConfigGrid:
    def test_grid_expansion(self) -> None:
        base = TrainConfig(model_type="beta_vae")
        grid = {"latent_dim": [64, 128], "beta": [1.0, 4.0]}
        configs = config_grid(base, grid)
        assert len(configs) == 4
        dims = {c.latent_dim for c in configs}
        betas = {c.beta for c in configs}
        assert dims == {64, 128}
        assert betas == {1.0, 4.0}

    def test_grid_with_seeds(self) -> None:
        base = TrainConfig()
        grid = {"latent_dim": [64, 128]}
        configs = config_grid_with_seeds(base, grid, seeds=[42, 123])
        assert len(configs) == 4
        seeds = [c.seed for c in configs]
        assert seeds.count(42) == 2
        assert seeds.count(123) == 2

    def test_config_to_dict(self) -> None:
        cfg = TrainConfig(model_type="wgan_gp", seed=123)
        d = config_to_dict(cfg)
        assert d["model_type"] == "wgan_gp"
        assert d["seed"] == 123
        assert isinstance(d, dict)


class TestEarlyStopping:
    def test_no_stop_improving(self) -> None:
        es = EarlyStopping(patience=3)
        for val in [1.0, 0.9, 0.8, 0.7]:
            assert not es.step(val)

    def test_stop_after_patience(self) -> None:
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(1.0)
        es.step(0.5)
        assert not es.step(0.6)
        assert not es.step(0.6)
        assert es.step(0.6)

    def test_max_mode(self) -> None:
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5)
        es.step(0.8)
        assert not es.step(0.7)
        assert es.step(0.7)

    def test_reset(self) -> None:
        es = EarlyStopping(patience=1)
        es.step(1.0)
        es.step(1.5)
        assert es.should_stop
        es.reset()
        assert not es.should_stop
        assert es.best_score is None


class DummyTrainer(BaseTrainer):
    """Minimal concrete trainer for testing BaseTrainer mechanics."""

    def build_models(self) -> None:
        self.model = torch.nn.Linear(3 * 128 * 128, 10)
        self.model.to(self.device)

    def build_optimizers(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        x = batch.view(batch.size(0), -1)
        out = self.model(x)
        loss = (out**2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list | tuple):
                    batch = batch[0]
                batch = batch.to(self.device)
                x = batch.view(batch.size(0), -1)
                out = self.model(x)
                total += (out**2).mean().item()
                n += 1
        return {"val_loss": total / max(n, 1)}

    def generate_samples(self, n: int) -> torch.Tensor:
        return torch.randn(n, 3, 128, 128)

    def state_dicts(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dicts(self, checkpoint: dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def _make_loaders(
    n_train: int = 32, n_val: int = 8, batch_size: int = 8
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_data = torch.randn(n_train, 3, 128, 128)
    val_data = torch.randn(n_val, 3, 128, 128)
    train_loader: DataLoader[Any] = DataLoader(TensorDataset(train_data), batch_size=batch_size)
    val_loader: DataLoader[Any] = DataLoader(TensorDataset(val_data), batch_size=batch_size)
    return train_loader, val_loader


class TestBaseTrainer:
    def test_seed_determinism(self) -> None:
        cfg = TrainConfig(device="cpu", max_epochs=1, seed=42)
        t = DummyTrainer(cfg)
        t.seed_everything(42)
        a = torch.randn(5)
        t.seed_everything(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_fit_runs(self, mock_mlflow: MagicMock, tmp_path: str) -> None:
        cfg = TrainConfig(
            device="cpu",
            max_epochs=3,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=1,
            log_interval=1,
        )
        trainer = DummyTrainer(cfg)
        train_loader, val_loader = _make_loaders()
        results = trainer.fit(train_loader, val_loader)

        assert "final_epoch" in results
        assert results["final_epoch"] == 3
        mock_mlflow.set_experiment.assert_called_once()
        assert mock_mlflow.log_metrics.call_count > 0

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_fit_writes_best_samples(self, mock_mlflow: MagicMock, tmp_path: str) -> None:
        cfg = TrainConfig(
            device="cpu",
            max_epochs=2,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
            log_interval=1,
        )
        trainer = DummyTrainer(cfg)
        ckpt_dir = trainer._ckpt_dir
        train_loader, val_loader = _make_loaders()
        trainer.fit(train_loader, val_loader)

        assert (ckpt_dir / "samples_best.png").is_file()
        assert (ckpt_dir / f"best_seed{cfg.seed}.pt").is_file()
        best_artifact_calls = [
            c
            for c in mock_mlflow.log_artifact.call_args_list
            if c.args and str(c.args[0]).endswith("samples_best.png")
        ]
        assert len(best_artifact_calls) == 1

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_early_stopping_triggers(self, _mock_mlflow: MagicMock, tmp_path: str) -> None:
        cfg = TrainConfig(
            device="cpu",
            max_epochs=300,
            checkpoint_dir=str(tmp_path),
            patience=2,
            min_epochs=0,
            sample_interval=300,
        )
        trainer = DummyTrainer(cfg)
        train_loader, val_loader = _make_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] < 300
        assert results.get("early_stopped") is True
        assert trainer.state.finished is True

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_early_stop_not_resumed(self, _mock_mlflow: MagicMock, tmp_path: str) -> None:
        cfg = TrainConfig(
            device="cpu",
            max_epochs=300,
            checkpoint_dir=str(tmp_path),
            patience=2,
            min_epochs=0,
            sample_interval=300,
        )
        train_loader, val_loader = _make_loaders()
        first = DummyTrainer(cfg).fit(train_loader, val_loader)
        assert first["final_epoch"] < 300

        second = DummyTrainer(cfg).fit(train_loader, val_loader)
        assert second.get("skipped") is True
        assert second["final_epoch"] == first["final_epoch"]

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_min_epochs_defers_early_stop(self, _mock_mlflow: MagicMock, tmp_path: str) -> None:
        cfg = TrainConfig(
            device="cpu",
            max_epochs=20,
            checkpoint_dir=str(tmp_path),
            patience=2,
            min_epochs=100,
            sample_interval=100,
        )
        trainer = DummyTrainer(cfg)
        train_loader, val_loader = _make_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 20
        assert results.get("early_stopped") is not True

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_checkpoint_save_load(self, _mock_mlflow: MagicMock, tmp_path: str) -> None:
        cfg = TrainConfig(
            device="cpu",
            max_epochs=2,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
        )
        trainer = DummyTrainer(cfg)
        trainer.build_models()
        trainer.build_optimizers()
        trainer.state.epoch = 5
        trainer.save_checkpoint("test")

        trainer2 = DummyTrainer(cfg)
        trainer2.build_models()
        trainer2.build_optimizers()
        assert trainer2.load_checkpoint("test")
        assert trainer2.state.epoch == 5


class TestExperimentRunner:
    @patch("gen_cats.training.base_trainer.mlflow")
    def test_runner_runs_all(self, _mock_mlflow: MagicMock, tmp_path: str) -> None:
        base = TrainConfig(
            device="cpu",
            max_epochs=2,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
        )
        grid = {"lr": [1e-3, 1e-4]}
        runner = ExperimentRunner(
            base_config=base,
            grid=grid,
            trainer_factory=DummyTrainer,
            seeds=[42, 7],
        )

        assert runner.total_runs == 4

        train_loader, val_loader = _make_loaders()
        results = runner.run_all(train_loader, val_loader)
        assert len(results) == 4
        assert all(r["status"] == "success" for r in results)
