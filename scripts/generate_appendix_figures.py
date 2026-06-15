"""Build appendix sample panels from checkpoint PNG grids."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from gen_cats.evaluation.report_analysis import (
    REPO_ROOT,
    ensure_plots_dir,
    load_fid_scores,
    resolve_sample_image,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS = REPO_ROOT / "checkpoints"
PLOTS = ensure_plots_dir("results")

SAMPLE_ROWS = 4
SAMPLES_PER_ROW = 4


def split_sample_grid(path: Path, grid: int = 4) -> list[np.ndarray]:
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    crop = (min(h, w) // grid) * grid
    off_y = (h - crop) // 2
    off_x = (w - crop) // 2
    img = img[off_y : off_y + crop, off_x : off_x + crop]
    tile_h, tile_w = crop // grid, crop // grid
    tiles: list[np.ndarray] = []
    for row in range(grid):
        for col in range(grid):
            tiles.append(img[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w])
    return tiles


def stitch_tiles(path: Path, sample_rows: int, samples_per_row: int, gap: int = 2) -> np.ndarray:
    tiles = split_sample_grid(path)[: sample_rows * samples_per_row]
    if not tiles:
        msg = f"No tiles extracted from {path}"
        raise ValueError(msg)
    tile_h, tile_w = tiles[0].shape[:2]
    channels = tiles[0].shape[2] if tiles[0].ndim == 3 else 1
    canvas = np.ones(
        (
            sample_rows * tile_h + (sample_rows - 1) * gap,
            samples_per_row * tile_w + (samples_per_row - 1) * gap,
            channels,
        ),
        dtype=tiles[0].dtype,
    )
    for row in range(sample_rows):
        for col in range(samples_per_row):
            tile_i = row * samples_per_row + col
            if tile_i >= len(tiles):
                continue
            y0 = row * (tile_h + gap)
            x0 = col * (tile_w + gap)
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tiles[tile_i]
    return canvas


def panel_title(label: str, mean_fid: float, rank: int, slug: str) -> str:
    return f"{label} (rank {rank}, FID {mean_fid:.1f}) — {slug}"


def ranked_run(entry: dict[str, Any], rank: int) -> dict[str, Any]:
    runs = sorted(entry.get("runs", []), key=lambda row: row["mean_fid"])
    if rank < 1 or rank > len(runs):
        msg = f"Rank {rank} out of range for {entry['model']}"
        raise IndexError(msg)
    run = runs[rank - 1]
    slug = run["slug"]
    image = resolve_sample_image(CHECKPOINTS, entry["model"], slug)
    if image is None:
        msg = f"No sample PNG for {entry['model']}/{slug}"
        raise FileNotFoundError(msg)
    return {
        "model": entry["model"],
        "label": entry.get("label", entry["model"]),
        "mean_fid": float(run["mean_fid"]),
        "slug": slug,
        "rank": rank,
        "image": image,
    }


def make_panel_figure(
    items: list[dict[str, Any]],
    *,
    width: float = 10.5,
    model_cols: int = 2,
) -> plt.Figure:
    plt.rcParams["axes.grid"] = False
    n_models = len(items)
    n_model_rows = (n_models + model_cols - 1) // model_cols
    panel_width = width / model_cols
    panel_height = panel_width
    title_space = 0.1
    fig_height = n_model_rows * (panel_height + title_space)

    fig = plt.figure(figsize=(width, fig_height))
    gs = fig.add_gridspec(n_model_rows, model_cols, wspace=0.03, hspace=0.08)

    for i, item in enumerate(items):
        row, col = divmod(i, model_cols)
        remainder = n_models % model_cols
        if remainder == 1 and i == n_models - 1:
            ax = fig.add_subplot(gs[row, :])
        else:
            ax = fig.add_subplot(gs[row, col])
        ax.imshow(
            stitch_tiles(Path(item["image"]), SAMPLE_ROWS, SAMPLES_PER_ROW, gap=1),
            aspect="equal",
        )
        ax.set_title(
            panel_title(item["label"], item["mean_fid"], item["rank"], item["slug"]),
            loc="left",
            fontsize=8,
            pad=2,
        )
        ax.axis("off")

    for j in range(n_models, n_model_rows * model_cols):
        row, col = divmod(j, model_cols)
        fig.add_subplot(gs[row, col]).axis("off")

    fig.subplots_adjust(left=0.002, right=0.998, top=0.97, bottom=0.002)
    return fig


def chimera_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for seed_dir in sorted((CHECKPOINTS / "chimera" / "wgan_gp").glob("chimera_64_seed*")):
        image = None
        for name in ("additional_samples.png", "samples_best.png"):
            candidate = seed_dir / name
            if candidate.is_file():
                image = candidate
                break
        if image is None:
            continue
        seed = seed_dir.name.rsplit("seed", maxsplit=1)[-1]
        items.append(
            {
                "label": f"Chimera WGAN-GP (seed {seed})",
                "mean_fid": float("nan"),
                "rank": 0,
                "slug": seed_dir.name,
                "image": image,
            }
        )
    if not items:
        msg = "No chimera sample grids found"
        raise FileNotFoundError(msg)
    return items


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    fid_by_model = {entry["model"]: entry for entry in load_fid_scores()}
    label_map = {
        "wgan_gp": "WGAN-GP",
        "sn_gan": "SN-GAN",
        "beta_vae": r"$\beta$-VAE",
        "vqvae": "VQ-VAE-1",
        "pixelcnn": "PixelCNN",
    }

    runners: list[dict[str, Any]] = []
    for model, rank in (
        ("sn_gan", 2),
        ("sn_gan", 3),
        ("wgan_gp", 5),
        ("beta_vae", 2),
    ):
        entry = fid_by_model[model]
        item = ranked_run(entry, rank)
        item["label"] = label_map[model]
        runners.append(item)

    fig = make_panel_figure(runners, model_cols=2)
    save_figure(fig, PLOTS / "appendix_promising_runners_up.png")

    chimera = chimera_items()
    fig = make_panel_figure(chimera, model_cols=3, width=10.5)
    for ax, item in zip(fig.axes, chimera, strict=True):
        ax.set_title(f"{item['label']} — {item['slug']}", loc="left", fontsize=8, pad=2)
    save_figure(fig, PLOTS / "appendix_chimera_samples.png")


if __name__ == "__main__":
    main()
