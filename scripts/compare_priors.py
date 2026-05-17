"""Compare PixelCNN vs Tiny LDM generation speed and save sample grids."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from gen_cats.config import TrainConfig, checkpoint_run_slug
from gen_cats.factory import create_trainer
from torchvision.utils import make_grid, save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_ckpt(
    checkpoint_dir: Path,
    model_type: str,
    seed: int,
    run_name: str = "",
) -> Path:
    slug = checkpoint_run_slug(TrainConfig(model_type=model_type, run_name=run_name))
    path = checkpoint_dir / model_type / slug / f"best_seed{seed}.pt"
    if path.exists():
        return path
    root = checkpoint_dir / model_type
    candidates = sorted(
        root.glob(f"**/best_seed{seed}.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            root.glob("**/best_seed*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        msg = f"No {model_type} checkpoint under {root}/"
        raise FileNotFoundError(msg)
    return candidates[0]


def _load_trainer(
    model_type: str,
    seed: int,
    device: str,
    checkpoint_dir: str,
    vqvae_seed: int,
    vqvae_run_name: str,
) -> tuple[object, Path]:
    cfg = TrainConfig(
        model_type=model_type,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        vqvae_seed=vqvae_seed,
        vqvae_run_name=vqvae_run_name,
    )
    trainer = create_trainer(cfg)
    trainer.build_models()
    trainer.build_optimizers()
    ckpt_path = _resolve_ckpt(Path(checkpoint_dir), model_type, seed)
    if not trainer.load_checkpoint("best"):
        trainer.load_checkpoint("latest")
    return trainer, ckpt_path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PixelCNN and Tiny LDM priors")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/prior_comparison")
    parser.add_argument("--pixelcnn-seed", type=int, default=42)
    parser.add_argument("--ldm-seed", type=int, default=42)
    parser.add_argument("--vqvae-seed", type=int, default=42)
    parser.add_argument("--vqvae-run-name", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PixelCNN prior (seed=%d)", args.pixelcnn_seed)
    pixel_trainer, pixel_ckpt = _load_trainer(
        "pixelcnn",
        args.pixelcnn_seed,
        args.device,
        args.checkpoint_dir,
        args.vqvae_seed,
        args.vqvae_run_name,
    )
    logger.info("Loading Tiny LDM (seed=%d)", args.ldm_seed)
    ldm_trainer, ldm_ckpt = _load_trainer(
        "tiny_ldm",
        args.ldm_seed,
        args.device,
        args.checkpoint_dir,
        args.vqvae_seed,
        args.vqvae_run_name,
    )

    n = args.n_samples
    pixel_samples, pixel_sec = _timed_samples(pixel_trainer, n)
    ldm_samples, ldm_sec = _timed_samples(ldm_trainer, n)

    nrow = 4
    for name, samples in [("pixelcnn", pixel_samples), ("tiny_ldm", ldm_samples)]:
        grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
        save_image(grid, out_dir / f"{name}_samples.png")

    combined = torch.cat([pixel_samples, ldm_samples], dim=0)
    combined_grid = make_grid(combined, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(combined_grid, out_dir / "combined_comparison.png")

    summary = {
        "n_samples": n,
        "pixelcnn_checkpoint": str(pixel_ckpt),
        "tiny_ldm_checkpoint": str(ldm_ckpt),
        "pixelcnn_seconds": pixel_sec,
        "tiny_ldm_seconds": ldm_sec,
        "pixelcnn_seconds_per_image": pixel_sec / n,
        "tiny_ldm_seconds_per_image": ldm_sec / n,
        "speedup_ldm_over_pixelcnn": pixel_sec / max(ldm_sec, 1e-9),
    }
    summary_path = out_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "PixelCNN: %.2fs (%.3fs/img) | Tiny LDM: %.2fs (%.3fs/img)",
        pixel_sec,
        summary["pixelcnn_seconds_per_image"],
        ldm_sec,
        summary["tiny_ldm_seconds_per_image"],
    )
    logger.info("Saved grids and metrics to %s", out_dir)


if __name__ == "__main__":
    main()
