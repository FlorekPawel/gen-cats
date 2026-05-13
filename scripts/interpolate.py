"""Generate latent interpolation strips for WGAN-GP and Beta-VAE."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from gen_cats.config import TrainConfig
from gen_cats.evaluation.interpolation import interpolation_strip, save_interpolation_grid
from gen_cats.factory import create_trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_interpolation(
    model_type: str,
    checkpoint_dir: str,
    output_dir: str,
    device: str = "mps",
    seed: int = 42,
    run_name: str = "",
) -> None:
    """Load best checkpoint and generate interpolation strip."""
    cfg = TrainConfig(
        model_type=model_type,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
    )

    trainer = create_trainer(cfg)
    trainer.build_models()
    trainer.build_optimizers()

    if not trainer.load_checkpoint("best"):
        logger.warning("No best checkpoint for %s, skipping", model_type)
        return

    if model_type in ("beta_vae", "wgan_gp", "sn_gan"):
        if hasattr(trainer, "model"):
            decoder_fn = lambda z: trainer.model.decoder(z)  # noqa: E731
            latent_dim = cfg.latent_dim
        elif hasattr(trainer, "generator"):
            decoder_fn = lambda z: trainer.generator(z)  # noqa: E731
            latent_dim = cfg.latent_dim
        else:
            logger.warning("Cannot find decoder for %s", model_type)
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
    else:
        logger.info("Interpolation not supported for %s", model_type)


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for model_type in ["beta_vae", "wgan_gp"]:
        generate_interpolation(
            model_type=model_type,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
