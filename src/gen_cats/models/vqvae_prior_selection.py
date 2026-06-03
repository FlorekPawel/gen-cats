"""Pick the best VQ-VAE checkpoint per seed across the sweep grid for PixelCNN / Tiny LDM."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from gen_cats.config import TrainConfig, effective_vqvae_seed

logger = logging.getLogger(__name__)

VQVAE_PRIOR_MANIFEST_NAME = "prior_best_by_seed.json"
VqvaeSelection = Literal["slug", "manifest", "auto"]


@dataclass(frozen=True)
class VqvaePriorEntry:
    """One seed's winning VQ-VAE cell from the sweep grid."""

    path: str
    slug: str
    seed: int
    best_metric: float
    num_embeddings: int
    feature_map_size: int
    recon_loss: str
    embedding_dim: int
    commitment_cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "vqvae" / VQVAE_PRIOR_MANIFEST_NAME


def _read_vqvae_checkpoint_meta(path: Path) -> tuple[float, dict[str, Any], str] | None:
    """Return (best_metric, vqvae config dict, slug) or None if not a VQ-VAE run."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        logger.warning("Skipping unreadable VQ-VAE checkpoint %s", path, exc_info=True)
        return None

    cfg = ckpt.get("config") or {}
    if cfg.get("model_type") != "vqvae":
        return None

    ts = ckpt.get("train_state") or {}
    metric = float(ts.get("best_metric", float("inf")))
    slug = path.parent.name
    return metric, cfg, slug


def _iter_vqvae_best_for_seed(vqvae_root: Path, seed: int) -> list[Path]:
    pattern = f"best_seed{seed}.pt"
    return sorted(
        (p for p in vqvae_root.rglob(pattern) if p.name == pattern),
        key=lambda p: p.parent.name,
    )


def select_best_vqvae_for_seed(
    checkpoint_dir: str | Path,
    seed: int,
) -> VqvaePriorEntry | None:
    """Among all ``checkpoints/vqvae/<slug>/best_seed{seed}.pt``, pick lowest val ``recon``."""
    vqvae_root = Path(checkpoint_dir) / "vqvae"
    if not vqvae_root.is_dir():
        return None

    manifest = manifest_path(checkpoint_dir)
    best: VqvaePriorEntry | None = None
    for path in _iter_vqvae_best_for_seed(vqvae_root, seed):
        if path == manifest:
            continue
        meta = _read_vqvae_checkpoint_meta(path)
        if meta is None:
            continue
        metric, cfg, slug = meta
        entry = VqvaePriorEntry(
            path=str(path.resolve()),
            slug=slug,
            seed=seed,
            best_metric=metric,
            num_embeddings=int(cfg.get("num_embeddings", 512)),
            feature_map_size=int(cfg.get("feature_map_size", 16)),
            recon_loss=str(cfg.get("recon_loss", "mse")),
            embedding_dim=int(cfg.get("embedding_dim", 64)),
            commitment_cost=float(cfg.get("commitment_cost", 0.25)),
        )
        if (
            best is None
            or entry.best_metric < best.best_metric
            or (entry.best_metric == best.best_metric and entry.slug < best.slug)
        ):
            best = entry

    return best


def build_vqvae_prior_manifest(
    checkpoint_dir: str | Path,
    seeds: list[int],
    *,
    metric: str = "val_recon",
) -> dict[str, Any]:
    """Scan the VQ-VAE grid and return manifest JSON payload."""
    entries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        picked = select_best_vqvae_for_seed(checkpoint_dir, seed)
        if picked is not None:
            entries[str(seed)] = picked.to_dict()

    return {
        "version": 1,
        "metric": metric,
        "lower_is_better": True,
        "seeds": entries,
    }


def save_vqvae_prior_manifest(
    checkpoint_dir: str | Path,
    seeds: list[int],
    *,
    metric: str = "val_recon",
) -> Path:
    """Write ``checkpoints/vqvae/prior_best_by_seed.json``."""
    payload = build_vqvae_prior_manifest(checkpoint_dir, seeds, metric=metric)
    out = manifest_path(checkpoint_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "VQ-VAE prior manifest: %s (%d/%d seeds)",
        out,
        len(payload["seeds"]),
        len(seeds),
    )
    for seed_str, row in sorted(payload["seeds"].items(), key=lambda x: int(x[0])):
        logger.info(
            "  seed %s → %s (val_recon=%.6f, K=%d, map=%d, loss=%s)",
            seed_str,
            row["slug"],
            row["best_metric"],
            row["num_embeddings"],
            row["feature_map_size"],
            row["recon_loss"],
        )
    return out


def load_vqvae_prior_manifest(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    path = manifest_path(checkpoint_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_vqvae_from_manifest(
    checkpoint_dir: str | Path,
    cfg: TrainConfig,
) -> Path | None:
    """Return manifest path for ``effective_vqvae_seed(cfg)`` if present and file exists."""
    payload = load_vqvae_prior_manifest(checkpoint_dir)
    if not payload:
        return None

    seed = effective_vqvae_seed(cfg)
    row = (payload.get("seeds") or {}).get(str(seed))
    if not row:
        logger.warning("VQ-VAE prior manifest has no entry for seed %d", seed)
        return None

    path = Path(row["path"])
    if not path.is_file():
        logger.warning("VQ-VAE manifest path missing for seed %d: %s", seed, path)
        return None

    logger.info(
        "Resolved VQ-VAE from prior manifest (seed=%d, slug=%s, val_recon=%.6f): %s",
        seed,
        row.get("slug"),
        row.get("best_metric"),
        path,
    )
    return path
