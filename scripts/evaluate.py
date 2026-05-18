"""CLI: compute FID for all trained model families."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from gen_cats.config import SEEDS
from gen_cats.evaluation.fid_benchmark import MODEL_TYPES, VQVAE_GRID_FIELDS, evaluate_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _vqvae_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: getattr(args, name) for name in VQVAE_GRID_FIELDS if getattr(args, name) is not None
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FID for all trained models")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Match training run_name (or default hyperparameter fingerprint)",
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
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated subset of model types (default: all)",
    )
    for name in VQVAE_GRID_FIELDS:
        flag = f"--{name.replace('_', '-')}"
        ftype = int if name != "recon_loss" else str
        choices = ["l1", "mse"] if name == "recon_loss" else None
        parser.add_argument(flag, type=ftype, default=None, choices=choices)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else SEEDS

    if args.models.strip():
        model_types = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = set(model_types) - set(MODEL_TYPES)
        if unknown:
            msg = f"Unknown model types: {unknown}. Available: {MODEL_TYPES}"
            raise ValueError(msg)
    else:
        model_types = list(MODEL_TYPES)

    overrides = _vqvae_overrides(args)
    results = evaluate_all(
        model_types,
        seeds,
        device=args.device,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        n_samples=args.n_samples,
        vqvae_overrides=overrides or None,
    )

    for result in results:
        logger.info(
            "%s: FID = %.2f +/- %.2f",
            result["model"],
            result["mean_fid"],
            result["std_fid"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
