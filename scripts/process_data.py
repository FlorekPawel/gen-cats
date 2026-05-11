"""Process raw images into cropped .npy arrays (cats or dogs+cats)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CATS_DIR = PROJECT_ROOT / "data" / "raw" / "cats"
RAW_DOGCAT_DIR = PROJECT_ROOT / "data" / "raw" / "dogcat"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def process_cats(args: argparse.Namespace) -> None:
    from gen_cats.data.processing import process_dataset

    stats = process_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        size=args.size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Cats done: {stats['train']} train, {stats['val']} val, {stats['skipped']} skipped")


def process_dogcat(args: argparse.Namespace) -> None:
    from gen_cats.data.dogcat_dataset import process_dogcat_dataset

    stats = process_dogcat_dataset(
        raw_dir=args.dogcat_raw_dir,
        output_dir=args.output_dir,
        size=args.size,
        seed=args.seed,
    )
    print(f"Dogs+Cats done: {stats['total']} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process images → .npy")
    parser.add_argument(
        "--dataset",
        choices=["cats", "dogcat", "all"],
        default="cats",
        help="Which dataset to process",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_CATS_DIR)
    parser.add_argument("--dogcat-raw-dir", type=Path, default=RAW_DOGCAT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset in ("cats", "all"):
        process_cats(args)
    if args.dataset in ("dogcat", "all"):
        process_dogcat(args)


if __name__ == "__main__":
    main()
