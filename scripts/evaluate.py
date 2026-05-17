"""Compute FID for best models across 3 seeds."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from gen_cats.config import SEEDS, TrainConfig
from gen_cats.evaluation.fid import compute_fid_from_loaders
from gen_cats.factory import create_dataloaders, create_trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_TYPES = ["beta_vae", "vqvae", "wgan_gp", "sn_gan", "ddim"]


def evaluate_model(
    model_type: str,
    args: argparse.Namespace,
    seeds: list[int],
) -> dict[str, float | str | dict[str, float]]:
    """Compute FID for a model type across the given seeds."""
    per_seed: dict[str, float] = {}

    for seed in seeds:
        cfg = TrainConfig(
            model_type=model_type,
            seed=seed,
            device=args.device,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
        )

        try:
            trainer = create_trainer(cfg)
            trainer.build_models()
            trainer.build_optimizers()

            if not trainer.load_checkpoint("best"):
                logger.warning("No best checkpoint for %s seed=%d, skipping", model_type, seed)
                continue

            _train_loader, val_loader = create_dataloaders(cfg)

            def gen_fn(n: int, _t: object = trainer) -> torch.Tensor:
                return _t.generate_samples(n).cpu()

            fid = compute_fid_from_loaders(
                val_loader, gen_fn, n_samples=args.n_samples, device=torch.device(cfg.device)
            )
            per_seed[str(seed)] = fid
            logger.info("FID %s seed=%d: %.2f", model_type, seed, fid)

        except Exception:
            logger.exception("Failed to evaluate %s seed=%d", model_type, seed)

    if not per_seed:
        return {"model": model_type, "mean_fid": float("nan"), "std_fid": float("nan")}

    import numpy as np

    fid_scores = list(per_seed.values())
    return {
        "model": model_type,
        "seeds": list(per_seed.keys()),
        "per_seed": per_seed,
        "mean_fid": float(np.mean(fid_scores)),
        "std_fid": float(np.std(fid_scores)),
        "fid_scores": fid_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FID for trained models")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Match training run_name (or omit to use default-hparam fingerprint)",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/fid_scores.json")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed only (default: all SEEDS)",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS

    results = []
    for model_type in MODEL_TYPES:
        result = evaluate_model(model_type, args, seeds)
        results.append(result)
        logger.info(
            "%s: FID = %.2f +/- %.2f", result["model"], result["mean_fid"], result["std_fid"]
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
