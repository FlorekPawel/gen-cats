"""Build per-seed VQ-VAE winner manifest for PixelCNN / Tiny LDM (lowest val recon)."""

from __future__ import annotations

import argparse
import json
import logging

from gen_cats.config import SEEDS
from gen_cats.models.vqvae_prior_selection import save_vqvae_prior_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best VQ-VAE checkpoint per seed across the sweep grid",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"seeds to include (default: project SEEDS {SEEDS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else list(SEEDS)
    path = save_vqvae_prior_manifest(args.checkpoint_dir, seeds)
    payload_seeds = len(json.loads(path.read_text(encoding="utf-8")).get("seeds", {}))
    if payload_seeds == 0:
        raise SystemExit(
            f"No VQ-VAE checkpoints found under {args.checkpoint_dir}/vqvae/. "
            "Run `make sweep-vae` first."
        )
    print(f"Wrote {path} ({payload_seeds} seeds)")


if __name__ == "__main__":
    main()
