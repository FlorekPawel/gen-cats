"""Tests for report analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

from gen_cats.evaluation.report_analysis import (
    best_runs_table,
    format_hyperparameters,
    latex_fid_table,
    list_interpolation_images,
    load_fid_scores,
)


def test_load_fid_scores_has_seven_families() -> None:
    data = load_fid_scores()
    models = {entry["model"] for entry in data}
    assert "wgan_gp" in models
    assert "beta_vae" in models
    assert len(data) == 7


def test_best_runs_table_wgan_gp_is_best() -> None:
    data = load_fid_scores()
    rows = best_runs_table(data)
    scored = sorted(
        [row for row in rows if row["mean_fid"] == row["mean_fid"]],
        key=lambda row: row["mean_fid"],
    )
    assert scored[0]["model"] == "wgan_gp"
    assert scored[0]["mean_fid"] < scored[1]["mean_fid"]


def test_format_hyperparameters_beta_vae() -> None:
    text = format_hyperparameters("beta_vae", {"latent_dim": 128, "beta": 1.0})
    assert "128" in text
    assert "beta" in text


def test_latex_fid_table_contains_booktabs() -> None:
    rows = best_runs_table(load_fid_scores())
    latex = latex_fid_table(rows)
    assert r"\toprule" in latex
    assert "WGAN-GP" in latex


def test_interpolation_images_exist() -> None:
    images = list_interpolation_images()
    assert len(images) >= 1
    assert images[0].suffix == ".png"


def test_prior_summary_on_disk() -> None:
    path = Path("results/prior_comparison/summary.json")
    assert path.is_file()
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["mean_speedup_ldm_over_pixelcnn"] > 1.0
