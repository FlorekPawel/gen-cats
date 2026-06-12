"""Compare PixelCNN vs Tiny LDM generation speed and sample grids."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from gen_cats.config import TrainConfig, effective_vqvae_seed
from gen_cats.evaluation.checkpoint_resolve import load_trainer_for_eval

logger = logging.getLogger(__name__)


def _load_trainer(cfg: TrainConfig, seed: int) -> tuple[object, Path]:
    run_cfg = replace(cfg, seed=seed, vqvae_seed=None)
    loaded = load_trainer_for_eval(run_cfg)
    if loaded is None:
        loaded = load_trainer_for_eval(run_cfg, tag="latest")
    if loaded is None:
        msg = f"No {cfg.model_type} checkpoint under {cfg.checkpoint_dir}/{cfg.model_type}/"
        raise FileNotFoundError(msg)
    return loaded


def _timed_samples(trainer: object, n: int) -> tuple[torch.Tensor, float]:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        samples = trainer.generate_samples(n)  # type: ignore[attr-defined]
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return samples, elapsed


def compare_priors_for_seed(
    *,
    prior_cfg: TrainConfig,
    seed: int,
    n_samples: int,
    output_dir: Path,
) -> dict[str, object]:
    """Compare PixelCNN vs Tiny LDM for one seed; write grids and return metrics dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vqvae_seed = effective_vqvae_seed(replace(prior_cfg, seed=seed))

    logger.info(
        "Comparing priors (seed=%d, vqvae grid slug matches sweep-vae cell)",
        seed,
    )
    pixel_cfg = replace(prior_cfg, model_type="pixelcnn")
    ldm_cfg = replace(prior_cfg, model_type="tiny_ldm")

    pixel_trainer, pixel_ckpt = _load_trainer(pixel_cfg, seed)
    ldm_trainer, ldm_ckpt = _load_trainer(ldm_cfg, seed)

    pixel_samples, pixel_sec = _timed_samples(pixel_trainer, n_samples)
    ldm_samples, ldm_sec = _timed_samples(ldm_trainer, n_samples)

    nrow = 4
    for name, samples in [("pixelcnn", pixel_samples), ("tiny_ldm", ldm_samples)]:
        grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
        save_image(grid, output_dir / f"{name}_samples.png")

    combined = torch.cat([pixel_samples, ldm_samples], dim=0)
    combined_grid = make_grid(combined, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(combined_grid, output_dir / "combined_comparison.png")

    summary: dict[str, object] = {
        "seed": seed,
        "vqvae_seed": vqvae_seed,
        "num_embeddings": prior_cfg.num_embeddings,
        "feature_map_size": prior_cfg.feature_map_size,
        "recon_loss": prior_cfg.recon_loss,
        "n_samples": n_samples,
        "pixelcnn_checkpoint": str(pixel_ckpt),
        "tiny_ldm_checkpoint": str(ldm_ckpt),
        "pixelcnn_seconds": pixel_sec,
        "tiny_ldm_seconds": ldm_sec,
        "pixelcnn_seconds_per_image": pixel_sec / n_samples,
        "tiny_ldm_seconds_per_image": ldm_sec / n_samples,
        "speedup_ldm_over_pixelcnn": pixel_sec / max(ldm_sec, 1e-9),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "seed %d — PixelCNN: %.3fs/img | Tiny LDM: %.3fs/img",
        seed,
        summary["pixelcnn_seconds_per_image"],
        summary["tiny_ldm_seconds_per_image"],
    )
    return summary


def compare_priors_all_seeds(
    seeds: list[int],
    prior_cfg: TrainConfig,
    output_dir: Path,
) -> dict[str, object]:
    """Run comparison for each seed; write per-seed dirs and a combined summary."""
    per_seed: list[dict[str, object]] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        per_seed.append(
            compare_priors_for_seed(
                prior_cfg=prior_cfg,
                seed=seed,
                n_samples=prior_cfg.n_sample_images,
                output_dir=seed_dir,
            )
        )

    def _mean(key: str) -> float:
        vals = [float(s[key]) for s in per_seed]  # type: ignore[arg-type]
        return sum(vals) / len(vals)

    aggregate: dict[str, object] = {
        "seeds": seeds,
        "num_embeddings": prior_cfg.num_embeddings,
        "feature_map_size": prior_cfg.feature_map_size,
        "recon_loss": prior_cfg.recon_loss,
        "n_samples": prior_cfg.n_sample_images,
        "per_seed": per_seed,
        "mean_pixelcnn_seconds_per_image": _mean("pixelcnn_seconds_per_image"),
        "mean_tiny_ldm_seconds_per_image": _mean("tiny_ldm_seconds_per_image"),
        "mean_speedup_ldm_over_pixelcnn": _mean("speedup_ldm_over_pixelcnn"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    logger.info("Aggregate summary written to %s", output_dir / "summary.json")
    return aggregate
