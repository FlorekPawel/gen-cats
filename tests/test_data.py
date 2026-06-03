"""Tests for data loading, parsing, and dataset classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gen_cats.data.cat_dataset import CatFaceDataset, default_transform
from gen_cats.data.processing import (
    crop_cat_face,
    parse_cat_annotation,
    process_dataset,
)


@pytest.fixture
def sample_cat_file(tmp_path: Path) -> Path:
    """Create a minimal .cat annotation file."""
    cat_path = tmp_path / "test.jpg.cat"
    cat_path.write_text("9 50 60 150 60 100 120 30 30 20 10 50 5 170 30 180 10 150 5")
    return cat_path


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a 200x200 test image with matching .cat annotation."""
    img_path = tmp_path / "test.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    img.save(img_path)

    cat_path = tmp_path / "test.jpg.cat"
    cat_path.write_text("9 50 60 150 60 100 120 30 30 20 10 50 5 170 30 180 10 150 5")

    return img_path


class TestCatAnnotation:
    def test_parse_valid(self, sample_cat_file: Path) -> None:
        ann = parse_cat_annotation(sample_cat_file)
        assert ann.left_eye == (50, 60)
        assert ann.right_eye == (150, 60)
        assert ann.mouth == (100, 120)
        assert len(ann.all_points) == 9

    def test_parse_wrong_count(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jpg.cat"
        bad.write_text("5 10 20 30 40 50 60 70 80 90 100")
        with pytest.raises(ValueError, match="Expected 9"):
            parse_cat_annotation(bad)

    def test_annotation_immutable(self, sample_cat_file: Path) -> None:
        ann = parse_cat_annotation(sample_cat_file)
        with pytest.raises(AttributeError):
            ann.left_eye = (0, 0)  # type: ignore[misc]


class TestCropCatFace:
    def test_crop_output_size(self, sample_image: Path) -> None:
        img = Image.open(sample_image).convert("RGB")
        ann = parse_cat_annotation(sample_image.with_suffix(".jpg.cat"))
        cropped = crop_cat_face(img, ann, size=128)
        assert cropped.size == (128, 128)

    def test_crop_custom_size(self, sample_image: Path) -> None:
        img = Image.open(sample_image).convert("RGB")
        ann = parse_cat_annotation(sample_image.with_suffix(".jpg.cat"))
        cropped = crop_cat_face(img, ann, size=32)
        assert cropped.size == (32, 32)


class TestProcessDataset:
    def test_process_creates_npy(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        out = tmp_path / "processed"

        for i in range(20):
            img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
            img_path = raw / f"cat_{i:04d}.jpg"
            img.save(img_path)
            cat_path = raw / f"cat_{i:04d}.jpg.cat"
            cat_path.write_text("9 50 60 150 60 100 120 30 30 20 10 50 5 170 30 180 10 150 5")

        stats = process_dataset(raw, out, val_fraction=0.1, seed=42)
        assert (out / "train.npy").exists()
        assert (out / "val.npy").exists()
        assert stats["train"] + stats["val"] == 20
        assert stats["val"] >= 1

        train_data = np.load(out / "train.npy")
        assert train_data.shape[1:] == (128, 128, 3)
        assert train_data.dtype == np.uint8

    def test_process_empty_dir_raises(self, tmp_path: Path) -> None:
        raw = tmp_path / "empty"
        raw.mkdir()
        with pytest.raises(FileNotFoundError, match="No image"):
            process_dataset(raw, tmp_path / "out")


class TestCatFaceDataset:
    @pytest.fixture
    def npy_file(self, tmp_path: Path) -> Path:
        data = np.random.randint(0, 255, (50, 128, 128, 3), dtype=np.uint8)
        path = tmp_path / "train.npy"
        np.save(path, data)
        return path

    def test_len(self, npy_file: Path) -> None:
        ds = CatFaceDataset(npy_file)
        assert len(ds) == 50

    def test_getitem_shape(self, npy_file: Path) -> None:
        ds = CatFaceDataset(npy_file)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 128, 128)

    def test_getitem_range(self, npy_file: Path) -> None:
        ds = CatFaceDataset(npy_file)
        sample = ds[0]
        assert sample.min() >= -1.0
        assert sample.max() <= 1.0

    def test_augmented_dataset(self, npy_file: Path) -> None:
        ds = CatFaceDataset(npy_file, augment=True)
        sample = ds[0]
        assert sample.shape == (3, 128, 128)

    def test_custom_transform(self, npy_file: Path) -> None:
        t = default_transform(augment=False)
        ds = CatFaceDataset(npy_file, transform=t)
        assert ds[0].shape == (3, 128, 128)
