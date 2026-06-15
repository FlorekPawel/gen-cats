"""CLI: generate additional sample grids from all best_seed*.pt checkpoints."""

from __future__ import annotations

import argparse
import logging

from gen_cats.evaluation.additional_samples import (
    discover_all_best_checkpoints,
    generate_all_additional_samples,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate additional_samples.png beside every best_seed*.pt checkpoint",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Grid size (default: n_sample_images from each checkpoint config)",
    )
    parser.add_argument(
        "--sample-seed-offset",
        type=int,
        default=10_000,
        help="RNG seed = train_seed + offset so grids differ from samples_best.png",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints that already have additional_samples.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpts = discover_all_best_checkpoints(args.checkpoint_dir)
    if not ckpts:
        logger.warning("No best_seed*.pt checkpoints under %s", args.checkpoint_dir)
        return

    logger.info("Found %d best checkpoint(s)", len(ckpts))
    written = generate_all_additional_samples(
        args.checkpoint_dir,
        device=args.device,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        sample_seed_offset=args.sample_seed_offset,
        skip_existing=args.skip_existing,
    )
    logger.info("Wrote %d additional sample grid(s)", len(written))


if __name__ == "__main__":
    main()
