"""Train PixelCNN for all project seeds, then compare each to Tiny LDM."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields
from pathlib import Path

from gen_cats.config import SEEDS, TrainConfig
from gen_cats.evaluation.prior_comparison import compare_priors_all_seeds
from gen_cats.factory import create_dataloaders, create_trainer
from gen_cats.training.experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VQVAE_GRID_FIELDS = (
    "num_embeddings",
    "feature_map_size",
    "recon_loss",
    "embedding_dim",
    "commitment_cost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelCNN baseline: train 3 seeds + compare vs Tiny LDM",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/prior_comparison")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--vqvae-run-name", type=str, default="")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    for name in VQVAE_GRID_FIELDS:
        flag = f"--{name.replace('_', '-')}"
        ftype = int if name != "recon_loss" else str
        choices = ["l1", "mse"] if name == "recon_loss" else None
        parser.add_argument(flag, type=ftype, default=None, choices=choices)
    return parser.parse_args()


def _prior_base_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        model_type="pixelcnn",
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        vqvae_run_name=args.vqvae_run_name,
        sample_interval=10,
        n_sample_images=args.n_samples,
    )
    overrides = {
        name: getattr(args, name) for name in VQVAE_GRID_FIELDS if getattr(args, name) is not None
    }
    if not overrides:
        return cfg
    cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    cfg_dict.update(overrides)
    return TrainConfig(**cfg_dict)


def main() -> None:
    args = parse_args()
    base = _prior_base_config(args)

    if not args.skip_train:
        runner = ExperimentRunner(
            base_config=base,
            grid={},
            trainer_factory=create_trainer,
            seeds=SEEDS,
        )
        logger.info(
            "Training PixelCNN for seeds %s (VQ-VAE: same seed + grid num_embeddings=%d, "
            "feature_map_size=%d, recon_loss=%s)",
            SEEDS,
            base.num_embeddings,
            base.feature_map_size,
            base.recon_loss,
        )
        train_loader, val_loader = create_dataloaders(base)
        results = runner.run_all(train_loader, val_loader)
        n_ok = sum(1 for r in results if r["status"] == "success")
        logger.info("PixelCNN training: %d/%d successful", n_ok, len(results))
        if n_ok == 0:
            raise RuntimeError("All PixelCNN training runs failed")

    if not args.skip_compare:
        compare_priors_all_seeds(SEEDS, base, Path(args.output_dir))


if __name__ == "__main__":
    main()
