"""Run a full grid search sweep for a model family."""

from __future__ import annotations

import argparse
import logging

from gen_cats.config import GRIDS, SEEDS, TrainConfig
from gen_cats.factory import create_dataloaders, create_trainer
from gen_cats.training.experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FAMILY_MODELS: dict[str, list[str]] = {
    "vae": ["beta_vae", "vqvae"],
    "gan": ["wgan_gp", "sn_gan"],
    "dm": ["ddim", "tiny_ldm"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grid sweep for model family")
    parser.add_argument("--family", choices=["vae", "gan", "dm"], required=True)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    model_types = FAMILY_MODELS[args.family]

    for model_type in model_types:
        grid = GRIDS.get(model_type)
        if grid is None:
            logger.warning("No grid defined for %s, skipping", model_type)
            continue

        base = TrainConfig(
            model_type=model_type,
            data_dir=args.data_dir,
        )

        runner = ExperimentRunner(
            base_config=base,
            grid=grid,
            trainer_factory=create_trainer,
            seeds=SEEDS,
        )

        logger.info("Starting sweep: %s (%d runs)", model_type, runner.total_runs)

        train_loader, val_loader = create_dataloaders(base)
        results = runner.run_all(train_loader, val_loader)

        n_ok = sum(1 for r in results if r["status"] == "success")
        logger.info("Sweep %s done: %d/%d successful", model_type, n_ok, len(results))


if __name__ == "__main__":
    main()
