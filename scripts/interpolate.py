"""Generate latent interpolation strips for WGAN-GP and Beta-VAE (all project seeds)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from gen_cats.config import SEEDS, TrainConfig
from gen_cats.evaluation.checkpoint_resolve import load_trainer_for_eval
from gen_cats.evaluation.interpolation import interpolation_strip, save_interpolation_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INTERPOLATION_MODELS = ("beta_vae", "wgan_gp")


def generate_interpolation(
    model_type: str,
    seed: int,
    *,
    checkpoint_dir: str,
    output_dir: str,
    device: str = "mps",
    run_name: str = "",
) -> None:
    """Load best checkpoint and generate interpolation strip for one seed."""
    cfg = TrainConfig(
        model_type=model_type,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
    )

    loaded = load_trainer_for_eval(cfg)
    if loaded is None:
        logger.warning("No best checkpoint for %s seed=%d, skipping", model_type, seed)
        return
    trainer, _ = loaded

    if model_type not in ("beta_vae", "wgan_gp", "sn_gan"):
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
    parser = argparse.ArgumentParser(description="Generate latent interpolation strips")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Match training run_name (or omit to use default-hparam fingerprint)",
    )
    parser.add_argument("--output-dir", type=str, default="results/interpolations")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed only (default: all SEEDS)",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS
    for seed in seeds:
        for model_type in INTERPOLATION_MODELS:
            generate_interpolation(
                model_type,
                seed,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir,
                device=args.device,
                run_name=args.run_name,
            )


if __name__ == "__main__":
    main()
