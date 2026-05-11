"""PyTorch Dataset for pre-cropped cat face .npy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(augment: bool = False) -> transforms.Compose:
    """Normalize to [-1, 1]. Optional augmentation adds horizontal flip."""
    t: list[Any] = []
    if augment:
        t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(t)


class CatFaceDataset(Dataset[torch.Tensor]):
    """Dataset backed by a .npy file of shape (N, 64, 64, 3) uint8 images.

    Args:
        npy_path: path to .npy file
        transform: torchvision transform; defaults to normalize + optional augment
        augment: enable RandomHorizontalFlip (only used when transform is None)
    """

    def __init__(
        self,
        npy_path: str | Path,
        transform: transforms.Compose | None = None,
        augment: bool = False,
    ) -> None:
        self.data: NDArray[np.uint8] = np.load(npy_path, mmap_mode="r")
        self.transform = transform or default_transform(augment=augment)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.fromarray(np.array(self.data[idx]))
        return self.transform(img)
