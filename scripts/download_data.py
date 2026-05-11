"""Download Cat Dataset and Dogs vs Cats from Kaggle."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CATS_DIR = PROJECT_ROOT / "data" / "raw" / "cats"
RAW_DOGCAT_DIR = PROJECT_ROOT / "data" / "raw" / "dogcat"


def download_cat_dataset(dest: Path = RAW_CATS_DIR) -> Path:
    """Download Cat Dataset → dest directory. Returns path to downloaded data."""
    import kagglehub

    logger.info("Downloading Cat Dataset from Kaggle...")
    cached_path = Path(kagglehub.dataset_download("crawford/cat-dataset"))
    logger.info("Kaggle cache: %s", cached_path)

    dest.mkdir(parents=True, exist_ok=True)

    cat_dir = cached_path / "CAT_00"
    if not cat_dir.exists():
        cat_dirs = sorted(p for p in cached_path.rglob("CAT_*") if p.is_dir())
        if not cat_dirs:
            msg = f"No CAT_* directories found in {cached_path}"
            raise FileNotFoundError(msg)
    else:
        cat_dirs = sorted(cached_path.glob("CAT_*"))

    n_copied = 0
    for cat_dir in cat_dirs:
        for f in cat_dir.iterdir():
            if f.is_file():
                target = dest / f.name
                if not target.exists():
                    shutil.copy2(f, target)
                    n_copied += 1

    logger.info("Copied %d files → %s", n_copied, dest)
    return dest


def download_dogcat_dataset(dest: Path = RAW_DOGCAT_DIR) -> Path:
    """Download Dogs vs Cats dataset → dest directory."""
    import kagglehub

    logger.info("Downloading Dogs vs Cats from Kaggle...")
    cached_path = Path(kagglehub.dataset_download("karakaggle/kaggle-cat-vs-dog-dataset"))
    logger.info("Kaggle cache: %s", cached_path)

    dest.mkdir(parents=True, exist_ok=True)

    pet_dir = cached_path / "PetImages"
    if not pet_dir.exists():
        candidates = sorted(p for p in cached_path.rglob("PetImages") if p.is_dir())
        if not candidates:
            msg = f"No 'PetImages' directory found in {cached_path}"
            raise FileNotFoundError(msg)
        pet_dir = candidates[0]

    n_copied = 0
    for class_dir in sorted(pet_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        prefix = class_dir.name.lower()  # "cat" or "dog"
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                target = dest / f"{prefix}.{f.name}"
                if not target.exists():
                    shutil.copy2(f, target)
                    n_copied += 1

    logger.info("Copied %d files → %s", n_copied, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets from Kaggle")
    parser.add_argument(
        "--dataset",
        choices=["cats", "dogcat", "all"],
        default="cats",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    if args.dataset in ("cats", "all"):
        download_cat_dataset()
    if args.dataset in ("dogcat", "all"):
        download_dogcat_dataset()


if __name__ == "__main__":
    main()
