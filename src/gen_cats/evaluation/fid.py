"""FID (Frechet Inception Distance) computation using pretrained InceptionV3."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_inception(device: torch.device) -> nn.Module:
    """Load InceptionV3 truncated at pool3 (2048-d features)."""
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(device)
    return inception


_INCEPTION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]."""
    return (x + 1) * 0.5


@torch.no_grad()
def extract_features(
    images: torch.Tensor,
    inception: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Extract InceptionV3 features from a tensor of images (N, 3, H, W) in [-1,1]."""
    inception.eval()
    features_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    n_batches = (len(images) + batch_size - 1) // batch_size
    batch_iter = tqdm(
        range(0, len(images), batch_size),
        total=n_batches,
        desc="Inception features",
        unit="batch",
    )
    for i in batch_iter:
        batch = images[i : i + batch_size].to(device)
        batch = _denorm(batch)
        batch = _INCEPTION_TRANSFORM(batch)
        feat = inception(batch)
        if isinstance(feat, tuple):
            feat = feat[0]
        features_list.append(feat.cpu().numpy().astype(np.float64))

    return np.concatenate(features_list, axis=0)


def compute_fid(
    real_features: np.ndarray[Any, np.dtype[np.float64]],
    fake_features: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """Compute FID between two sets of InceptionV3 features."""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean = linalg.sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))


@torch.no_grad()
def compute_fid_from_loaders(
    real_loader: DataLoader[Any],
    generator_fn: Any,
    n_samples: int,
    device: torch.device,
) -> float:
    """End-to-end FID: real data loader + generator function → FID score."""
    inception = _get_inception(device)

    real_imgs: list[torch.Tensor] = []
    for batch in real_loader:
        if isinstance(batch, list | tuple):
            batch = batch[0]
        real_imgs.append(batch)
        if sum(b.size(0) for b in real_imgs) >= n_samples:
            break
    real_tensor = torch.cat(real_imgs)[:n_samples]

    fake_tensor = generator_fn(n_samples)
    if fake_tensor.device != torch.device("cpu"):
        fake_tensor = fake_tensor.cpu()

    real_feat = extract_features(real_tensor, inception, device)
    fake_feat = extract_features(fake_tensor, inception, device)

    return compute_fid(real_feat, fake_feat)
