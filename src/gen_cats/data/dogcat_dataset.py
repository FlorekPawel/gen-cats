"""Dogs vs Cats mixed dataset for chimera experiment (64x64 by default)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from gen_cats.config import CHIMERA_IMAGE_SIZE

logger = logging.getLogger(__name__)

# Legacy 128x128 filename kept for backwards compatibility when size=128.
DOG_CAT_NPY_STEM = "dogcat_train"


def dogcat_npy_filename(size: int = CHIMERA_IMAGE_SIZE) -> str:
    """Processed dog+cat array filename for a given square resolution."""
    if size == 128:
        return f"{DOG_CAT_NPY_STEM}.npy"
    return f"{DOG_CAT_NPY_STEM}_{size}.npy"


def dogcat_npy_path(data_dir: str | Path, size: int = CHIMERA_IMAGE_SIZE) -> Path:
    return Path(data_dir) / dogcat_npy_filename(size)


def default_transform(image_size: int = CHIMERA_IMAGE_SIZE) -> transforms.Compose:
    """Resize shortest side, center-crop, normalize to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class DogsVsCatsDataset(Dataset[torch.Tensor]):
    """Mixed dogs + cats dataset from Kaggle competition.

    Loads from directory containing files like cat.0.jpg, dog.0.jpg, etc.
    Crops center square before resizing.
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: transforms.Compose | None = None,
        max_per_class: int | None = None,
        image_size: int = CHIMERA_IMAGE_SIZE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform or default_transform(image_size)

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        cat_paths = sorted(self.data_dir.glob("cat.*.jpg"))
        dog_paths = sorted(self.data_dir.glob("dog.*.jpg"))

        if max_per_class:
            cat_paths = cat_paths[:max_per_class]
            dog_paths = dog_paths[:max_per_class]

        for p in cat_paths:
            self.image_paths.append(p)
            self.labels.append(0)
        for p in dog_paths:
            self.image_paths.append(p)
            self.labels.append(1)

        logger.info(
            "DogsVsCats: %d cats, %d dogs from %s (%dx%d)",
            len(cat_paths),
            len(dog_paths),
            self.data_dir,
            image_size,
            image_size,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def process_dogcat_dataset(
    raw_dir: Path,
    output_dir: Path,
    size: int = CHIMERA_IMAGE_SIZE,
    max_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Process dogs+cats images → ``dogcat_train_{size}.npy`` (or ``dogcat_train.npy`` at 128)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = DogsVsCatsDataset(raw_dir, max_per_class=max_per_class, image_size=size)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))

    resize_crop = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ]
    )

    from tqdm import tqdm

    images: list[Any] = []
    for idx in tqdm(indices, desc=f"Processing dogs+cats {size}x{size}", unit="img"):
        img = Image.open(ds.image_paths[idx]).convert("RGB")
        img = resize_crop(img)
        images.append(np.array(img, dtype=np.uint8))

    arr = np.stack(images)
    out_path = output_dir / dogcat_npy_filename(size)
    np.save(out_path, arr)
    logger.info("Saved %d dog+cat images (%dx%d) → %s", len(arr), size, size, out_path)
    return {"total": len(arr), "size": size, "path": str(out_path)}
