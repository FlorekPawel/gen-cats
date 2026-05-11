"""Linear interpolation in latent space for VAE and GAN models."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid


def linear_interpolation(z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 8) -> torch.Tensor:
    """Generate n_steps intermediate latent vectors between z1 and z2.

    Returns tensor of shape (n_steps + 2, *z1.shape) including endpoints.
    """
    alphas = torch.linspace(0, 1, n_steps + 2, device=z1.device)
    return torch.stack([z1 * (1 - a) + z2 * a for a in alphas])


@torch.no_grad()
def interpolation_strip(
    decoder_fn: object,
    latent_dim: int,
    device: torch.device,
    n_steps: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Generate interpolation strip: 10 images (2 endpoints + 8 intermediate).

    Args:
        decoder_fn: callable(z) → images (B, 3, H, W)
        latent_dim: dimension of latent space
        device: torch device
        n_steps: number of intermediate steps (default 8 → 10 total images)
        seed: for reproducible z1, z2

    Returns:
        Tensor of shape (n_steps+2, 3, H, W)
    """
    torch.manual_seed(seed)
    z1 = torch.randn(latent_dim, device=device)
    z2 = torch.randn(latent_dim, device=device)

    z_interp = linear_interpolation(z1, z2, n_steps)
    return decoder_fn(z_interp)  # type: ignore[operator]


def save_interpolation_grid(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int | None = None,
) -> Path:
    """Save interpolation images as a single-row grid."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if nrow is None:
        nrow = images.size(0)

    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(grid_np).save(path)
    return path
