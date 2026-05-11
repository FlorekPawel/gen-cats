"""ExperimentRunner: automates grid search x seeds, sequential run execution."""

from __future__ import annotations

import logging
from typing import Any

from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_cats.config import TrainConfig, config_grid_with_seeds

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run a grid of experiments: all param combos x all seeds.

    Args:
        base_config: base TrainConfig to expand
        grid: param name → list of values
        seeds: list of random seeds (default: [42, 123, 7])
        trainer_factory: callable(config) → BaseTrainer instance
    """

    def __init__(
        self,
        base_config: TrainConfig,
        grid: dict[str, list[Any]],
        trainer_factory: Any,
        seeds: list[int] | None = None,
    ) -> None:
        self.configs = config_grid_with_seeds(base_config, grid, seeds)
        self.trainer_factory = trainer_factory
        self.results: list[dict[str, Any]] = []

    @property
    def total_runs(self) -> int:
        return len(self.configs)

    def run_all(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> list[dict[str, Any]]:
        """Execute all runs sequentially, collecting results."""
        run_bar = tqdm(self.configs, desc="Sweep", unit="run")
        for i, cfg in enumerate(run_bar):
            run_bar.set_postfix(model=cfg.model_type, seed=cfg.seed)
            try:
                trainer = self.trainer_factory(cfg)
                metrics = trainer.fit(train_loader, val_loader)
                result = {
                    "run_index": i,
                    "model_type": cfg.model_type,
                    "seed": cfg.seed,
                    "status": "success",
                    **metrics,
                }
            except Exception:
                logger.exception("Run %d failed", i + 1)
                result = {
                    "run_index": i,
                    "model_type": cfg.model_type,
                    "seed": cfg.seed,
                    "status": "failed",
                }

            self.results.append(result)

        n_ok = sum(1 for r in self.results if r["status"] == "success")
        logger.info("Completed %d/%d runs successfully", n_ok, self.total_runs)
        return self.results
