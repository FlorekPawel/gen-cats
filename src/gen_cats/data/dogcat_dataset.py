"""Dogs vs Cats mixed dataset for chimera experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMG_SIZE = 64


def default_transform() -> transforms.Compose:
    """Resize shortest side to 64 px (preserving aspect ratio), then center-crop to 64x64."""
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class DogsVsCatsDataset(Dataset[torch.Tensor]):
    """Mixed dogs + cats dataset from Kaggle competition.

    Loads from directory containing files like cat.0.jpg, dog.0.jpg, etc.
    Crops center square before resizing to 64x64.
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: transforms.Compose | None = None,
        max_per_class: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform or default_transform()

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
            "DogsVsCats: %d cats, %d dogs from %s",
            len(cat_paths),
            len(dog_paths),
            self.data_dir,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def process_dogcat_dataset(
    raw_dir: Path,
    output_dir: Path,
    size: int = IMG_SIZE,
    max_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Process dogs+cats images → .npy for fast loading."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = DogsVsCatsDataset(raw_dir, max_per_class=max_per_class)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))

    resize_crop = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ]
    )

    images: list[Any] = []
    for idx in indices:
        img = Image.open(ds.image_paths[idx]).convert("RGB")
        img = resize_crop(img)
        images.append(np.array(img, dtype=np.uint8))

    arr = np.stack(images)
    np.save(output_dir / "dogcat_train.npy", arr)
    logger.info("Saved %d dog+cat images → %s", len(arr), output_dir)
    return {"total": len(arr)}
