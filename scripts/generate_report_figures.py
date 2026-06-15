"""Generate report figures under notebooks/plots/ (same outputs as eda.ipynb)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"
CAT_TRAIN = REPO_ROOT / "data" / "processed" / "train.npy"
DOG_CAT_CANDIDATES = (
    REPO_ROOT / "data" / "processed" / "dogcat_train_64.npy",
    REPO_ROOT / "data" / "processed" / "dogcat_train.npy",
)
RAW_CAT_SAMPLE = REPO_ROOT / "data" / "raw" / "cats" / "00000001_011.jpg"


def _load_npy(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        msg = f"{label} not found at {path}"
        raise FileNotFoundError(msg)
    return np.load(path, mmap_mode="r")


def _synthetic_panel(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def _save_grid(images: np.ndarray, path: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ax, img in zip(axes.flat, images[:4], strict=True):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _save_preprocessing_comparison(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if CAT_TRAIN.is_file() and RAW_CAT_SAMPLE.is_file():
        train = np.load(CAT_TRAIN, mmap_mode="r")
        after = train[3]
        before = Image.open(RAW_CAT_SAMPLE).convert("RGB").resize((128, 128))
        before_arr = np.array(before)
    else:
        logger.warning("Using synthetic before/after panel (run make process-data for real crops)")
        rng = np.random.default_rng(0)
        before_arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        after = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(before_arr)
    axes[0].set_title("Before processing")
    axes[0].axis("off")
    axes[1].imshow(after)
    axes[1].set_title("After processing")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _resolve_dogcat_npy() -> Path | None:
    for path in DOG_CAT_CANDIDATES:
        if path.is_file():
            return path
    return None


def generate_all(*, allow_synthetic: bool = True) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if CAT_TRAIN.is_file():
        train = _load_npy(CAT_TRAIN, "Cat train split")
        _save_grid(train, PLOTS_DIR / "processed_samples.png", dpi=500)
    elif allow_synthetic:
        logger.warning("Missing %s — writing synthetic cat grid", CAT_TRAIN)
        _save_grid(_synthetic_panel(4, 128, seed=1), PLOTS_DIR / "processed_samples.png", dpi=500)
    else:
        raise FileNotFoundError(CAT_TRAIN)

    _save_preprocessing_comparison(PLOTS_DIR / "preprocessing_comparison.png")

    dogcat_path = _resolve_dogcat_npy()
    if dogcat_path is not None:
        dogcat = _load_npy(dogcat_path, "Dog+cat tensor")
        _save_grid(dogcat, PLOTS_DIR / "processed_samples_dogcat.png", dpi=300)
    elif allow_synthetic:
        logger.warning("Missing dogcat .npy — writing synthetic mixed-species grid")
        _save_grid(
            _synthetic_panel(4, 64, seed=2), PLOTS_DIR / "processed_samples_dogcat.png", dpi=300
        )
    else:
        raise FileNotFoundError("No dogcat_train_64.npy or dogcat_train.npy found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX report figures")
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="fail instead of writing synthetic placeholders",
    )
    args = parser.parse_args()
    generate_all(allow_synthetic=not args.require_data)


if __name__ == "__main__":
    main()
