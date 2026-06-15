"""Tests for MLflow / checkpoint artifact loaders."""

from __future__ import annotations

from gen_cats.evaluation.experiment_artifacts import (
    catalog_checkpoints,
    join_fid_checkpoints,
    load_mlflow_runs,
    match_mlflow_to_fid_cell,
    training_summary_by_model,
)
from gen_cats.evaluation.report_analysis import best_runs_table, load_fid_scores


def test_load_mlflow_runs_non_empty() -> None:
    df = load_mlflow_runs()
    assert len(df) >= 200
    assert "model_type" in df.columns
    assert set(df["status"].unique()) <= {"FINISHED", "FAILED", "RUNNING", "KILLED"}


def test_checkpoint_catalog_lists_epoch_samples() -> None:
    cat = catalog_checkpoints()
    assert len(cat) >= 100
    assert cat["n_epoch_samples"].sum() > 1000


def test_join_fid_checkpoints_links_samples() -> None:
    joined = join_fid_checkpoints()
    assert not joined.empty
    assert joined["has_samples_best"].any()


def test_match_mlflow_to_fid_best_beta_vae() -> None:
    ml = load_mlflow_runs()
    best = best_runs_table(load_fid_scores())
    row = next(r for r in best if r["model"] == "beta_vae")
    matched = match_mlflow_to_fid_cell(ml, "beta_vae", row["hyperparameters"])
    assert len(matched) >= 1


def test_join_fid_mlflow_metrics_has_best_metric() -> None:
    from gen_cats.evaluation.experiment_artifacts import join_fid_mlflow_metrics

    frame = join_fid_mlflow_metrics()
    assert not frame.empty
    missing = frame["best_metric"].isna().sum()
    assert missing == 0, f"{missing} rows missing best_metric"


def test_training_summary_counts() -> None:
    summary = training_summary_by_model(load_mlflow_runs())
    assert int(summary["runs_finished"].sum()) > 100
    assert "median_duration_min" in summary.columns
    assert summary["median_sec_per_epoch"].notna().any()
