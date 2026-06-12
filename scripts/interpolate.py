"""Generate latent interpolation strips for best FID WGAN-GP and Beta-VAE runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from gen_cats.config import SEEDS, TrainConfig
from gen_cats.evaluation.checkpoint_resolve import load_trainer_from_checkpoint
from gen_cats.evaluation.fid_benchmark import best_slug_for_model, load_fid_score_results
from gen_cats.evaluation.interpolation import interpolation_strip, save_interpolation_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INTERPOLATION_MODELS = ("beta_vae", "wgan_gp")


def generate_interpolation(
    model_type: str,
    seed: int,
    *,
    slug: str,
    checkpoint_dir: str,
    output_dir: str,
    device: str = "mps",
) -> None:
    """Load best-FID grid cell checkpoint and generate interpolation strip for one seed."""
    ckpt_path = Path(checkpoint_dir) / model_type / slug / f"best_seed{seed}.pt"
    if not ckpt_path.is_file():
        logger.warning("No checkpoint at %s, skipping", ckpt_path)
        return

    cfg = TrainConfig(
        model_type=model_type,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    trainer, _ = load_trainer_from_checkpoint(ckpt_path, cfg)

    if model_type not in INTERPOLATION_MODELS:
        logger.info("Interpolation not supported for %s", model_type)
        return

    if hasattr(trainer, "model"):
        decoder_fn = lambda z: trainer.model.decoder(z)  # noqa: E731  # type: ignore[attr-defined]
        latent_dim = trainer.config.latent_dim
    elif hasattr(trainer, "generator"):
        decoder_fn = lambda z: trainer.generator(z)  # noqa: E731
        latent_dim = trainer.config.latent_dim
    else:
        logger.warning("Cannot find decoder for %s seed=%d", model_type, seed)
        return

    images = interpolation_strip(
        decoder_fn=decoder_fn,
        latent_dim=latent_dim,
        device=torch.device(device),
        n_steps=8,
        seed=seed,
    )

    out_path = Path(output_dir) / f"interpolation_{model_type}_seed{seed}.png"
    save_interpolation_grid(images, out_path)
    logger.info("Saved interpolation: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolation strips for best FID hyperparameter cell per model",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--fid-scores", type=str, default="results/fid_scores.json")
    parser.add_argument("--output-dir", type=str, default="results/interpolations")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed only (default: all SEEDS)",
    )
    args = parser.parse_args()

    fid_results = load_fid_score_results(args.fid_scores)
    seeds = [args.seed] if args.seed is not None else SEEDS

    for model_type in INTERPOLATION_MODELS:
        slug, hparams = best_slug_for_model(fid_results, model_type)
        logger.info(
            "Best FID cell for %s: slug=%s hyperparameters=%s",
            model_type,
            slug,
            hparams,
        )
        for seed in seeds:
            generate_interpolation(
                model_type,
                seed,
                slug=slug,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir,
                device=args.device,
            )


if __name__ == "__main__":
    main()
