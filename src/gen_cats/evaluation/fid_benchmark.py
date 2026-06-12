"""FID evaluation across all trained model families."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gen_cats.config import TrainConfig
from gen_cats.evaluation.checkpoint_resolve import (
    discover_checkpoints,
    load_trainer_from_checkpoint,
    run_hyperparameters,
)
from gen_cats.evaluation.fid import compute_fid_from_loaders
from gen_cats.factory import create_dataloaders

logger = logging.getLogger(__name__)

MODEL_TYPES: list[str] = [
    "beta_vae",
    "vqvae",
    "pixelcnn",
    "tiny_ldm",
    "wgan_gp",
    "sn_gan",
    "ddim",
]

VQVAE_LINKED_MODELS = frozenset({"vqvae", "pixelcnn", "tiny_ldm"})

VQVAE_GRID_FIELDS = (
    "num_embeddings",
    "feature_map_size",
    "recon_loss",
    "embedding_dim",
    "commitment_cost",
)


def build_eval_config(
    model_type: str,
    seed: int,
    *,
    device: str,
    data_dir: str,
    checkpoint_dir: str,
    run_name: str = "",
    vqvae_overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Build ``TrainConfig`` for FID evaluation of one model and seed."""
    cfg = TrainConfig(
        model_type=model_type,
        seed=seed,
        device=device,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        vqvae_seed=None,
        vqvae_selection="auto",
    )
    if vqvae_overrides and model_type in VQVAE_LINKED_MODELS:
        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        cfg_dict.update(vqvae_overrides)
        return TrainConfig(**cfg_dict)
    return cfg


def _matches_vqvae_filter(cfg: TrainConfig, vqvae_overrides: dict[str, Any] | None) -> bool:
    if not vqvae_overrides or cfg.model_type not in VQVAE_LINKED_MODELS:
        return True
    return all(getattr(cfg, key, None) == value for key, value in vqvae_overrides.items())


def evaluate_model(
    model_type: str,
    seeds: list[int],
    *,
    device: str,
    data_dir: str,
    checkpoint_dir: str,
    run_name: str = "",
    n_samples: int = 1000,
    vqvae_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute FID for every sweep grid cell and seed with a ``best`` checkpoint."""
    all_paths = discover_checkpoints(
        checkpoint_dir,
        model_type,
        seeds,
        run_name=run_name,
        tag="best",
    )

    by_slug: dict[str, list[Any]] = defaultdict(list)
    for path in all_paths:
        by_slug[path.parent.name].append(path)

    runs: list[dict[str, Any]] = []

    for slug in sorted(by_slug):
        per_seed: dict[str, float] = {}
        hyperparameters: dict[str, Any] | None = None

        for ckpt_path in sorted(by_slug[slug], key=lambda p: p.name):
            seed = int(ckpt_path.stem.rsplit("_seed", 1)[1])
            seed_cfg = build_eval_config(
                model_type,
                seed,
                device=device,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                vqvae_overrides=vqvae_overrides,
            )

            try:
                trainer, _ = load_trainer_from_checkpoint(ckpt_path, seed_cfg)
                if not _matches_vqvae_filter(trainer.config, vqvae_overrides):
                    logger.info(
                        "Skipping %s slug=%s seed=%d (VQ filter)",
                        model_type,
                        slug,
                        seed,
                    )
                    continue

                if hyperparameters is None:
                    hyperparameters = run_hyperparameters(trainer.config)

                _train_loader, val_loader = create_dataloaders(trainer.config)

                def gen_fn(n: int, _t: object = trainer) -> torch.Tensor:
                    return _t.generate_samples(n).cpu()  # type: ignore[attr-defined]

                fid = compute_fid_from_loaders(
                    val_loader,
                    gen_fn,
                    n_samples=n_samples,
                    device=torch.device(device),
                )
                per_seed[str(seed)] = fid
                logger.info("FID %s slug=%s seed=%d: %.2f", model_type, slug, seed, fid)

            except Exception:
                logger.exception(
                    "Failed to evaluate %s slug=%s seed=%d",
                    model_type,
                    slug,
                    seed,
                )

        if not per_seed:
            continue

        scores = list(per_seed.values())
        runs.append(
            {
                "slug": slug,
                "hyperparameters": hyperparameters or {},
                "per_seed": per_seed,
                "mean_fid": float(np.mean(scores)),
                "std_fid": float(np.std(scores)),
            }
        )

    if not runs:
        return {
            "model": model_type,
            "n_runs": 0,
            "runs": [],
            "mean_fid": float("nan"),
            "std_fid": float("nan"),
        }

    best = min(runs, key=lambda r: r["mean_fid"])
    all_scores = [fid for r in runs for fid in r["per_seed"].values()]

    result: dict[str, Any] = {
        "model": model_type,
        "n_runs": len(runs),
        "runs": runs,
        "best_run": {
            "slug": best["slug"],
            "hyperparameters": best["hyperparameters"],
            "mean_fid": best["mean_fid"],
        },
        "mean_fid": float(np.mean(all_scores)),
        "std_fid": float(np.std(all_scores)),
    }
    if model_type == "vqvae":
        result["note"] = "Uses random codebook indices in generate_samples; not a learned prior."
    return result


def evaluate_all(
    model_types: list[str],
    seeds: list[int],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run FID for each model type (all grid cells per type)."""
    return [evaluate_model(mt, seeds, **kwargs) for mt in model_types]


def load_fid_score_results(path: str | Path) -> list[dict[str, Any]]:
    """Load ``results/fid_scores.json`` produced by ``scripts/evaluate.py``."""
    fid_path = Path(path)
    if not fid_path.is_file():
        msg = f"FID results not found at {fid_path}. Run `make eval-fid` first."
        raise FileNotFoundError(msg)
    data = json.loads(fid_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Expected a JSON list in {fid_path}"
        raise ValueError(msg)
    return data


def best_slug_for_model(
    fid_results: list[dict[str, Any]],
    model_type: str,
) -> tuple[str, dict[str, Any]]:
    """Return (slug, hyperparameters) for the lowest-mean-FID grid cell of ``model_type``."""
    for entry in fid_results:
        if entry.get("model") != model_type:
            continue
        best_run = entry.get("best_run")
        if isinstance(best_run, dict) and best_run.get("slug"):
            return str(best_run["slug"]), dict(best_run.get("hyperparameters") or {})
        runs = entry.get("runs") or []
        if runs:
            best = min(runs, key=lambda r: float(r["mean_fid"]))
            return str(best["slug"]), dict(best.get("hyperparameters") or {})
        msg = f"No FID runs recorded for {model_type!r} in results file"
        raise ValueError(msg)
    msg = f"Model {model_type!r} not found in FID results"
    raise ValueError(msg)
