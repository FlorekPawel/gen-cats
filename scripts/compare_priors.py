"""CLI: compare PixelCNN vs Tiny LDM generation speed and sample grids."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields
from pathlib import Path

from gen_cats.config import SEEDS, TrainConfig
from gen_cats.evaluation.prior_comparison import (
    compare_priors_all_seeds,
    compare_priors_for_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _prior_cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build config; VQ-VAE fields match ``make sweep-vae`` / ``make train-vae MODEL=vqvae``."""
    overrides: dict[str, object] = {
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
        "n_sample_images": args.n_samples,
        "vqvae_run_name": args.vqvae_run_name,
    }
    if args.num_embeddings is not None:
        overrides["num_embeddings"] = args.num_embeddings
    if args.feature_map_size is not None:
        overrides["feature_map_size"] = args.feature_map_size
    if args.recon_loss is not None:
        overrides["recon_loss"] = args.recon_loss
    base = {f.name: getattr(TrainConfig(), f.name) for f in fields(TrainConfig)}
    base.update(overrides)
    return TrainConfig(**base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PixelCNN and Tiny LDM priors")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/prior_comparison")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Single-seed mode (ignored with --all-seeds)",
    )
    parser.add_argument("--vqvae-run-name", type=str, default="")
    parser.add_argument("--num-embeddings", type=int, default=None)
    parser.add_argument("--feature-map-size", type=int, default=None)
    parser.add_argument("--recon-loss", type=str, default=None, choices=["l1", "mse"])
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help=f"Compare all project seeds {SEEDS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_cfg = _prior_cfg_from_args(args)
    out_dir = Path(args.output_dir)

    if args.all_seeds:
        compare_priors_all_seeds(SEEDS, prior_cfg, out_dir)
        logger.info("Saved per-seed grids under %s/seed_*", out_dir)
        return

    compare_priors_for_seed(
        prior_cfg=prior_cfg,
        seed=args.seed,
        n_samples=args.n_samples,
        output_dir=out_dir,
    )
    logger.info("Saved grids and metrics to %s", out_dir)


if __name__ == "__main__":
    main()
