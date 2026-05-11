"""Download Cat Dataset from Kaggle using kagglehub."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cats"


def download_cat_dataset(dest: Path = RAW_DIR) -> Path:
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


if __name__ == "__main__":
    download_cat_dataset()
