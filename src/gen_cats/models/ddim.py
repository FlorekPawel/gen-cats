"""DDIM noise schedule and sampling for denoising diffusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear beta schedule from DDPM paper."""
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Improved DDPM."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0, 0.999).float()


class DDIMScheduler:
    """DDIM forward process + deterministic/stochastic reverse sampling.

    Args:
        timesteps: total diffusion steps (T)
        schedule: "linear" or "cosine"
    """

    def __init__(self, timesteps: int = 1000, schedule: str = "linear") -> None:
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer_dict = {}
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    def to(self, device: torch.device) -> DDIMScheduler:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM deterministic sampling (eta=0) or stochastic (eta>0)."""
        step_size = self.timesteps // ddim_steps
        timestep_seq = list(range(0, self.timesteps, step_size))
        timestep_seq_prev = [-1, *timestep_seq[:-1]]

        x = torch.randn(shape, device=device)

        for t_cur, t_prev in zip(reversed(timestep_seq), reversed(timestep_seq_prev), strict=True):
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha_t = self.alpha_cumprod[t_cur]
            alpha_prev = self.alpha_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt(1 - alpha_t / alpha_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise
            noise = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise

        return x
