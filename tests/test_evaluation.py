"""Tests for evaluation modules: FID, interpolation, DogsVsCats dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gen_cats.evaluation.interpolation import (
    linear_interpolation,
    save_interpolation_grid,
)


class TestLinearInterpolation:
    def test_shape(self) -> None:
        z1 = torch.randn(128)
        z2 = torch.randn(128)
        result = linear_interpolation(z1, z2, n_steps=8)
        assert result.shape == (10, 128)

    def test_endpoints(self) -> None:
        z1 = torch.ones(64)
        z2 = torch.zeros(64)
        result = linear_interpolation(z1, z2, n_steps=8)
        assert torch.allclose(result[0], z1)
        assert torch.allclose(result[-1], z2)

    def test_midpoint(self) -> None:
        z1 = torch.zeros(32)
        z2 = torch.ones(32)
        result = linear_interpolation(z1, z2, n_steps=8)
        mid = result[5]
        expected = torch.full((32,), 5.0 / 9.0)
        assert torch.allclose(mid, expected, atol=1e-5)


class TestSaveGrid:
    def test_saves_png(self, tmp_path: Path) -> None:
        images = torch.randn(10, 3, 64, 64)
        path = save_interpolation_grid(images, tmp_path / "test_grid.png")
        assert path.exists()
        assert path.suffix == ".png"


class TestFIDCompute:
    def test_fid_zero_identical(self) -> None:
        from gen_cats.evaluation.fid import compute_fid

        features = np.random.randn(100, 64).astype(np.float64)
        fid = compute_fid(features, features)
        assert abs(fid) < 1e-6

    def test_fid_positive_different(self) -> None:
        from gen_cats.evaluation.fid import compute_fid

        real = np.random.randn(100, 64).astype(np.float64)
        fake = np.random.randn(100, 64).astype(np.float64) + 5.0
        fid = compute_fid(real, fake)
        assert fid > 0


class TestDogsVsCatsDataset:
    def test_dataset_from_files(self, tmp_path: Path) -> None:
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image

            Image.fromarray(img).save(tmp_path / f"cat.{i}.jpg")
            Image.fromarray(img).save(tmp_path / f"dog.{i}.jpg")

        from gen_cats.data.dogcat_dataset import DogsVsCatsDataset

        ds = DogsVsCatsDataset(tmp_path)
        assert len(ds) == 10
        sample = ds[0]
        assert sample.shape == (3, 64, 64)
        assert sample.min() >= -1.0
        assert sample.max() <= 1.0

    def test_max_per_class(self, tmp_path: Path) -> None:
        for i in range(10):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image

            Image.fromarray(img).save(tmp_path / f"cat.{i}.jpg")
            Image.fromarray(img).save(tmp_path / f"dog.{i}.jpg")

        from gen_cats.data.dogcat_dataset import DogsVsCatsDataset

        ds = DogsVsCatsDataset(tmp_path, max_per_class=3)
        assert len(ds) == 6
