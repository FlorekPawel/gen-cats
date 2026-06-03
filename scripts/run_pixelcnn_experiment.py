"""Train PixelCNN for all project seeds, then compare each to Tiny LDM."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields, replace
from pathlib import Path

from gen_cats.config import SEEDS, TrainConfig
from gen_cats.evaluation.prior_comparison import compare_priors_all_seeds
from gen_cats.factory import create_dataloaders, create_trainer
from gen_cats.models.vqvae_checkpoint import resolve_vqvae_checkpoint
from gen_cats.models.vqvae_prior_selection import (
    load_vqvae_prior_manifest,
    manifest_path,
    save_vqvae_prior_manifest,
)
from gen_cats.training.experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VQVAE_GRID_FIELDS = (
    "num_embeddings",
    "feature_map_size",
    "recon_loss",
    "embedding_dim",
    "commitment_cost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelCNN baseline: train 3 seeds + compare vs Tiny LDM",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/prior_comparison")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=None, help="default: TrainConfig (1000)")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--vqvae-run-name", type=str, default="")
    parser.add_argument(
        "--vqvae-selection",
        type=str,
        choices=["auto", "manifest", "slug"],
        default=None,
        help="how to pick frozen VQ-VAE (default: auto, or slug if VQ hparams passed)",
    )
    parser.add_argument(
        "--rebuild-vqvae-manifest",
        action="store_true",
        help="rescan checkpoints/vqvae and rewrite prior_best_by_seed.json",
    )
    parser.add_argument(
        "--skip-vqvae-manifest",
        action="store_true",
        help="do not build manifest (use existing or slug/fallback)",
    )
    parser.add_argument(
        "--allow-vqvae-fallback",
        action="store_true",
        help="if slug/manifest miss, use newest VQ checkpoint (not recommended)",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ldm", action="store_true", help="skip Tiny LDM training")
    parser.add_argument("--skip-compare", action="store_true")
    for name in VQVAE_GRID_FIELDS:
        flag = f"--{name.replace('_', '-')}"
        ftype = int if name != "recon_loss" else str
        choices = ["l1", "mse"] if name == "recon_loss" else None
        parser.add_argument(flag, type=ftype, default=None, choices=choices)
    return parser.parse_args()


def _vqvae_grid_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: getattr(args, name) for name in VQVAE_GRID_FIELDS if getattr(args, name) is not None
    }


def _prior_base_config(args: argparse.Namespace) -> TrainConfig:
    overrides = _vqvae_grid_overrides(args)
    pinned_slug = bool(overrides) or bool(args.vqvae_run_name.strip())

    selection = args.vqvae_selection
    if selection is None:
        selection = "slug" if pinned_slug else "auto"

    cfg = TrainConfig(
        model_type="pixelcnn",
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        vqvae_run_name=args.vqvae_run_name,
        vqvae_selection=selection,
        require_vqvae_slug=selection == "slug" and not args.allow_vqvae_fallback,
        sample_interval=10,
        n_sample_images=args.n_samples,
    )
    if args.max_epochs is not None:
        cfg = replace(cfg, max_epochs=args.max_epochs)
    if args.patience is not None:
        cfg = replace(cfg, patience=args.patience)

    if overrides:
        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        cfg_dict.update(overrides)
        cfg = TrainConfig(**cfg_dict)

    return cfg


def _ensure_vqvae_manifest(args: argparse.Namespace, cfg: TrainConfig) -> None:
    if args.skip_vqvae_manifest or cfg.vqvae_selection == "slug":
        return
    path = manifest_path(cfg.checkpoint_dir)
    if (
        args.rebuild_vqvae_manifest
        or not path.is_file()
        or load_vqvae_prior_manifest(cfg.checkpoint_dir) is None
    ):
        save_vqvae_prior_manifest(cfg.checkpoint_dir, list(SEEDS))


def _log_vqvae_resolution(cfg: TrainConfig) -> None:
    path = resolve_vqvae_checkpoint(cfg.checkpoint_dir, cfg, strict=cfg.require_vqvae_slug)
    logger.info(
        "VQ-VAE resolved: %s (selection=%s, seed=%d)",
        path,
        cfg.vqvae_selection,
        cfg.seed if cfg.vqvae_seed is None else cfg.vqvae_seed,
    )


def main() -> None:
    args = parse_args()
    base = _prior_base_config(args)
    _ensure_vqvae_manifest(args, base)

    _log_vqvae_resolution(replace(base, seed=SEEDS[0]))

    logger.info(
        "Prior experiment: max_epochs=%d, vqvae_selection=%s, require_vqvae_slug=%s",
        base.max_epochs,
        base.vqvae_selection,
        base.require_vqvae_slug,
    )

    if not args.skip_train:
        runner = ExperimentRunner(
            base_config=base,
            grid={},
            trainer_factory=create_trainer,
            seeds=SEEDS,
        )
        logger.info(
            "Training PixelCNN for seeds %s (vqvae_selection=%s)",
            SEEDS,
            base.vqvae_selection,
        )
        train_loader, val_loader = create_dataloaders(base)
        results = runner.run_all(train_loader, val_loader)
        n_ok = sum(1 for r in results if r["status"] == "success")
        logger.info("PixelCNN training: %d/%d successful", n_ok, len(results))
        if n_ok == 0:
            raise RuntimeError("All PixelCNN training runs failed")

        if not args.skip_ldm:
            ldm_base = replace(
                base,
                model_type="tiny_ldm",
                use_ema=True,
                ddim_steps=100,
            )
            ldm_runner = ExperimentRunner(
                base_config=ldm_base,
                grid={},
                trainer_factory=create_trainer,
                seeds=SEEDS,
            )
            logger.info(
                "Training Tiny LDM for seeds %s (vqvae_selection=%s, EMA on)",
                SEEDS,
                ldm_base.vqvae_selection,
            )
            ldm_results = ldm_runner.run_all(train_loader, val_loader)
            ldm_ok = sum(1 for r in ldm_results if r["status"] == "success")
            logger.info("Tiny LDM training: %d/%d successful", ldm_ok, len(ldm_results))
            if ldm_ok == 0:
                raise RuntimeError("All Tiny LDM training runs failed")

    if not args.skip_compare:
        compare_priors_all_seeds(SEEDS, base, Path(args.output_dir))


if __name__ == "__main__":
    main()
