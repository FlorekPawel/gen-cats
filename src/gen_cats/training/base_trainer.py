"""Abstract base trainer with seed management, checkpointing, MLflow logging, and early stopping."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from gen_cats.config import TrainConfig, checkpoint_run_slug, config_to_dict
from gen_cats.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    """Mutable training state for checkpoint save/restore."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    finished: bool = False
    early_stopped: bool = False


class BaseTrainer(ABC):
    """Template Method trainer: subclasses implement `train_step` and `validate`.

    Handles: seed enforcement, device placement, MLflow logging,
    checkpointing, early stopping, sample grid generation.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.state = TrainState()
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode="min",
        )

        slug = checkpoint_run_slug(config)
        self._ckpt_dir = Path(config.checkpoint_dir) / config.model_type / slug
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def seed_everything(self, seed: int) -> None:
        """Hard-reset all RNGs before training."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @abstractmethod
    def build_models(self) -> None:
        """Instantiate model(s) and move to device."""

    @abstractmethod
    def build_optimizers(self) -> None:
        """Create optimizer(s)."""

    @abstractmethod
    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Single training step. Returns dict of losses to log."""

    @abstractmethod
    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        """Validation pass. Returns dict with at least the early_stop_metric key."""

    @abstractmethod
    def generate_samples(self, n: int) -> torch.Tensor:
        """Generate n sample images for visualization. Returns (N, C, H, W) in [-1, 1]."""

    @abstractmethod
    def state_dicts(self) -> dict[str, Any]:
        """Return all model/optimizer state_dicts for checkpointing."""

    @abstractmethod
    def load_state_dicts(self, checkpoint: dict[str, Any]) -> None:
        """Restore model/optimizer states from checkpoint."""

    def use_early_stopping(self) -> bool:
        """Whether to stop training when the monitored metric plateaus."""
        return True

    def save_checkpoint(self, tag: str = "latest") -> Path:
        path = self._ckpt_dir / f"{tag}_seed{self.config.seed}.pt"
        payload: dict[str, Any] = {
            "config": config_to_dict(self.config),
            "train_state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
                "finished": self.state.finished,
                "early_stopped": self.state.early_stopped,
            },
            "early_stopping": {
                "counter": self.early_stopping.counter,
                "best_score": self.early_stopping.best_score,
                "should_stop": self.early_stopping.should_stop,
            },
            **self.state_dicts(),
        }
        torch.save(payload, path)
        logger.debug("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, tag: str = "latest", *, weights_only: bool = False) -> bool:
        path = self._ckpt_dir / f"{tag}_seed{self.config.seed}.pt"
        if not path.exists():
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dicts(checkpoint)
        if weights_only:
            return True

        ts = checkpoint["train_state"]
        self.state.epoch = ts["epoch"]
        self.state.global_step = ts["global_step"]
        self.state.best_metric = ts["best_metric"]
        self.state.finished = ts.get("finished", False)
        self.state.early_stopped = ts.get("early_stopped", False)
        es = checkpoint.get("early_stopping")
        if es is not None:
            self.early_stopping.counter = es["counter"]
            self.early_stopping.best_score = es["best_score"]
            self.early_stopping.should_stop = es["should_stop"]
        if self.state.finished:
            logger.info(
                "Run already finished at epoch %d (early_stopped=%s), will not resume training",
                self.state.epoch + 1,
                self.state.early_stopped,
            )
        else:
            logger.info("Resumed from %s (epoch %d)", path, self.state.epoch)
        return True

    def _save_sample_grid(self, samples: torch.Tensor, img_path: Path) -> Path:
        """Write an (N, C, H, W) tensor grid in [-1, 1] to a PNG file."""
        from PIL import Image

        grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(grid_np)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)
        return img_path

    def _log_samples(self, epoch: int) -> None:
        """Generate and log sample grid to MLflow (weights at current epoch)."""
        with torch.no_grad():
            samples = self.generate_samples(self.config.n_sample_images)
        img_path = self._save_sample_grid(samples, self._ckpt_dir / f"samples_epoch{epoch}.png")
        mlflow.log_artifact(str(img_path), artifact_path="samples")

    def _best_checkpoint_path(self) -> Path:
        return self._ckpt_dir / f"best_seed{self.config.seed}.pt"

    def _generate_best_samples(self, *, log_mlflow: bool) -> bool:
        """Load best checkpoint weights and write ``samples_best.png``."""
        best_path = self._best_checkpoint_path()
        if not best_path.exists():
            logger.warning("No best checkpoint at %s; skipping final sample generation", best_path)
            return False
        if not self.load_checkpoint("best", weights_only=True):
            return False

        self.seed_everything(self.config.seed)
        with torch.no_grad():
            samples = self.generate_samples(self.config.n_sample_images)
        out_path = self._ckpt_dir / "samples_best.png"
        self._save_sample_grid(samples, out_path)
        logger.info("Saved best-checkpoint samples to %s", out_path)
        if log_mlflow:
            mlflow.log_artifact(str(out_path), artifact_path="samples")
        return True

    def _run_name(self) -> str:
        if self.config.run_name:
            return self.config.run_name
        return f"{self.config.model_type}_seed{self.config.seed}"

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> dict[str, Any]:
        """Full training loop with MLflow tracking."""
        self.seed_everything(self.config.seed)
        self.build_models()
        self.build_optimizers()
        self.load_checkpoint("latest")

        if self.state.finished:
            best_samples = self._ckpt_dir / "samples_best.png"
            if not best_samples.exists():
                self._generate_best_samples(log_mlflow=False)
            return {
                "final_epoch": self.state.epoch + 1,
                "best_metric": self.state.best_metric,
                "early_stopped": self.state.early_stopped,
                "skipped": True,
            }

        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=self._run_name()):
            mlflow.log_params(config_to_dict(self.config))

            epoch_bar = tqdm(
                range(self.state.epoch, self.config.max_epochs),
                desc=self._run_name(),
                unit="ep",
            )
            for epoch in epoch_bar:
                self.state.epoch = epoch

                epoch_losses: dict[str, list[float]] = {}
                batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
                for batch in batch_bar:
                    if isinstance(batch, list | tuple):
                        batch = batch[0]
                    batch = batch.to(self.device)
                    step_losses = self.train_step(batch)
                    self.state.global_step += 1

                    for k, v in step_losses.items():
                        epoch_losses.setdefault(k, []).append(v)

                    batch_bar.set_postfix({k: f"{v:.4f}" for k, v in step_losses.items()})

                avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

                if epoch % self.config.log_interval == 0:
                    mlflow.log_metrics(
                        {f"train/{k}": v for k, v in avg_losses.items()},
                        step=epoch,
                    )

                val_metrics = self.validate(val_loader)
                mlflow.log_metrics(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=epoch,
                )

                monitor_val = val_metrics.get(self.config.early_stop_metric, 0.0)

                if monitor_val < self.state.best_metric:
                    self.state.best_metric = monitor_val
                    self.save_checkpoint("best")

                self.save_checkpoint("latest")

                if epoch % self.config.sample_interval == 0:
                    self._log_samples(epoch)

                epoch_bar.set_postfix(
                    {
                        **{k: f"{v:.4f}" for k, v in avg_losses.items()},
                        f"val_{self.config.early_stop_metric}": f"{monitor_val:.4f}",
                    }
                )

                if (
                    epoch + 1 >= self.config.min_epochs
                    and self.use_early_stopping()
                    and self.early_stopping.step(monitor_val)
                ):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    self.state.finished = True
                    self.state.early_stopped = True
                    self.save_checkpoint("latest")
                    break
            else:
                self.state.finished = True
                self.save_checkpoint("latest")

            final_metrics = {
                "final_epoch": self.state.epoch + 1,
                "best_metric": self.state.best_metric,
                "early_stopped": self.state.early_stopped,
            }
            self._generate_best_samples(log_mlflow=True)
            mlflow.log_metrics(final_metrics)

        return final_metrics
