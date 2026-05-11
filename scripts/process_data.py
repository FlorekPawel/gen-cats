"""Crop cat faces from raw images and save as .npy arrays."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gen_cats.data.processing import process_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cats"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Process cat images → cropped .npy")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = process_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        size=args.size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Done: {stats['train']} train, {stats['val']} val, {stats['skipped']} skipped")


if __name__ == "__main__":
    main()
