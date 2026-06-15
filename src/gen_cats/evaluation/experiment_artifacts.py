"""MLflow SQLite and checkpoint artifact loaders for report analysis."""

# ruff: noqa: S608

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen_cats.config import GRIDS, TrainConfig
from gen_cats.evaluation.report_analysis import MODEL_LABELS, REPO_ROOT, load_fid_scores

MLFLOW_DB = REPO_ROOT / "mlflow.db"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

EPOCH_SAMPLE_RE = re.compile(r"samples_epoch(\d+)\.png$")

SKIP_CHECKPOINT_ROOTS = frozenset({"chimera", ".DS_Store"})

MONITOR_METRIC: dict[str, str] = {
    "beta_vae": "val/recon",
    "vqvae": "val/recon",
    "wgan_gp": "val/g_loss",
    "sn_gan": "val/g_loss",
    "ddim": "val/val_loss",
    "tiny_ldm": "val/val_loss",
    "pixelcnn": "val/loss",
}


def _coerce_param_value(field_name: str, raw: str) -> Any:
    for field in fields(TrainConfig):
        if field.name != field_name:
            continue
        annotation = str(field.type)
        if raw in {"True", "False"}:
            return raw == "True"
        if "int" in annotation and raw not in {"", "None"}:
            try:
                return int(float(raw))
            except ValueError:
                return raw
        if "float" in annotation and raw not in {"", "None"}:
            try:
                return float(raw)
            except ValueError:
                return raw
        if raw == "None":
            return None
        return raw
    return raw


def _params_to_dict(rows: list[tuple[str, str]]) -> dict[str, Any]:
    return {key: _coerce_param_value(key, value) for key, value in rows}


def load_mlflow_runs(
    db_path: Path | str = MLFLOW_DB,
    *,
    experiments: tuple[str, ...] = ("gen-cats", "chimera-dogcat"),
) -> pd.DataFrame:
    """Wide dataframe of MLflow runs with pivoted params and final metrics."""
    path = Path(db_path)
    if not path.is_file():
        msg = f"MLflow database not found: {path}"
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(path)
    try:
        exp_placeholders = ",".join("?" for _ in experiments)
        runs = pd.read_sql_query(
            f"""
            SELECT r.run_uuid, r.name AS run_name, r.status, r.start_time, r.end_time,
                   e.name AS experiment
            FROM runs r
            JOIN experiments e ON e.experiment_id = r.experiment_id
            WHERE e.name IN ({exp_placeholders})
            """,
            conn,
            params=list(experiments),
        )
        if runs.empty:
            return runs

        params = pd.read_sql_query(
            """
            SELECT run_uuid, key, value
            FROM params
            WHERE run_uuid IN ({})
            """.format(",".join("?" for _ in runs["run_uuid"])),
            conn,
            params=runs["run_uuid"].tolist(),
        )
        param_wide = params.pivot_table(
            index="run_uuid", columns="key", values="value", aggfunc="first"
        )
        if "run_name" in param_wide.columns:
            param_wide = param_wide.drop(columns=["run_name"])
        for column in param_wide.columns:
            param_wide[column] = [
                _coerce_param_value(column, str(value) if pd.notna(value) else "")
                for value in param_wide[column]
            ]

        finals = pd.read_sql_query(
            """
            SELECT m.run_uuid, m.key, m.value, m.step
            FROM metrics m
            WHERE m.run_uuid IN ({})
              AND m.key IN ('best_metric', 'final_epoch', 'early_stopped')
              AND m.is_nan = 0
            """.format(",".join("?" for _ in runs["run_uuid"])),
            conn,
            params=runs["run_uuid"].tolist(),
        )
        if not finals.empty:
            finals = (
                finals.sort_values("step")
                .groupby(["run_uuid", "key"], as_index=False)
                .last()
                .pivot(index="run_uuid", columns="key", values="value")
                .reset_index()
            )
        else:
            finals = pd.DataFrame(columns=["run_uuid"])

        df = runs.merge(param_wide.reset_index(), on="run_uuid", how="left")
        if "run_uuid" in finals.columns:
            df = df.merge(finals, on="run_uuid", how="left")
        if "model_type" in df.columns:
            df["model_label"] = df["model_type"].map(MODEL_LABELS).fillna(df["model_type"])
        if "seed" in df.columns:
            df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
        if "start_time" in df.columns and "end_time" in df.columns:
            df["duration_s"] = np.where(
                df["end_time"].notna() & df["start_time"].notna(),
                (df["end_time"] - df["start_time"]) / 1000.0,
                np.nan,
            )
        else:
            df["duration_s"] = np.nan
        if "final_epoch" in df.columns:
            epochs = pd.to_numeric(df["final_epoch"], errors="coerce")
            df["sec_per_epoch"] = np.where(
                epochs > 0,
                df["duration_s"] / epochs,
                np.nan,
            )
        return df
    finally:
        conn.close()


def load_metric_history(
    run_uuid: str,
    metric_key: str,
    *,
    db_path: Path | str = MLFLOW_DB,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            """
            SELECT step AS epoch, value, timestamp
            FROM metrics
            WHERE run_uuid = ? AND key = ? AND is_nan = 0
            ORDER BY step
            """,
            conn,
            params=(run_uuid, metric_key),
        )
    finally:
        conn.close()


def monitor_metric_for_model(model_type: str) -> str:
    return MONITOR_METRIC.get(model_type, "val/val_loss")


def catalog_checkpoints(root: Path | str = CHECKPOINTS_DIR) -> pd.DataFrame:
    """Index every checkpoint slug directory and its PNG artifacts."""
    base = Path(root)
    rows: list[dict[str, Any]] = []
    if not base.is_dir():
        return pd.DataFrame(rows)

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_CHECKPOINT_ROOTS:
            continue
        if model_dir.name.startswith("old_"):
            legacy = True
            model_type = model_dir.name.removeprefix("old_")
        else:
            legacy = False
            model_type = model_dir.name

        for slug_dir in sorted(model_dir.iterdir()):
            if not slug_dir.is_dir() or slug_dir.name.startswith("."):
                continue
            pngs = sorted(slug_dir.glob("*.png"))
            epoch_paths = sorted(
                (p for p in pngs if EPOCH_SAMPLE_RE.match(p.name)),
                key=lambda p: int(EPOCH_SAMPLE_RE.match(p.name).group(1)),  # type: ignore[union-attr]
            )
            epoch_nums = [
                int(EPOCH_SAMPLE_RE.match(p.name).group(1))  # type: ignore[union-attr]
                for p in epoch_paths
            ]
            rows.append(
                {
                    "model_type": model_type,
                    "slug": slug_dir.name,
                    "legacy": legacy,
                    "checkpoint_root": model_dir.name,
                    "dir": slug_dir,
                    "n_epoch_samples": len(epoch_paths),
                    "epochs_sampled": epoch_nums,
                    "has_samples_best": (slug_dir / "samples_best.png").is_file(),
                    "has_additional_samples": (slug_dir / "additional_samples.png").is_file(),
                    "samples_best": slug_dir / "samples_best.png",
                    "additional_samples": slug_dir / "additional_samples.png",
                    "first_epoch_sample": epoch_paths[0] if epoch_paths else None,
                    "last_epoch_sample": epoch_paths[-1] if epoch_paths else None,
                }
            )
    return pd.DataFrame(rows)


def fid_runs_long(fid_data: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Flatten ``fid_scores.json`` to one row per (model, slug, seed)."""
    if fid_data is None:
        fid_data = load_fid_scores()
    rows: list[dict[str, Any]] = []
    for entry in fid_data:
        model = entry["model"]
        for run in entry.get("runs", []):
            slug = run.get("slug", "")
            hp = run.get("hyperparameters", {})
            for seed, fid in run.get("per_seed", {}).items():
                rows.append(
                    {
                        "model": model,
                        "slug": slug,
                        "seed": int(seed),
                        "fid": float(fid),
                        **{f"hp_{k}": v for k, v in hp.items()},
                    }
                )
        best = entry.get("best_run")
        if best:
            rows.append(
                {
                    "model": model,
                    "slug": best.get("slug", ""),
                    "seed": np.nan,
                    "fid": float(best.get("mean_fid", np.nan)),
                    "is_best_cell": True,
                    **{f"hp_{k}": v for k, v in best.get("hyperparameters", {}).items()},
                }
            )
    return pd.DataFrame(rows)


def _hp_match(run_params: dict[str, Any], hp: dict[str, Any]) -> bool:
    if not hp:
        return False
    for key, expected in hp.items():
        if key not in run_params:
            return False
        actual = run_params[key]
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-9):
                    return False
                continue
            except (TypeError, ValueError):
                pass
        if str(actual) != str(expected):
            return False
    return True


def hyperparameter_grid_keys(model_type: str) -> list[str]:
    return list(GRIDS.get(model_type, {}).keys())


def _fid_hyperparameters(model: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Keep only swept grid keys for MLflow matching."""
    keys = hyperparameter_grid_keys(model)
    if model == "pixelcnn":
        keys = ["num_embeddings", "feature_map_size", "recon_loss"]
    if not keys:
        return dict(hyperparameters)
    return {key: hyperparameters[key] for key in keys if key in hyperparameters}


def pick_mlflow_run(matches: pd.DataFrame) -> pd.Series | None:
    """Prefer a finished run with logged ``best_metric`` (latest ``end_time``)."""
    if matches.empty:
        return None
    frame = matches.copy()
    if "status" in frame.columns:
        finished = frame[frame["status"] == "FINISHED"]
        if not finished.empty:
            frame = finished
    if "best_metric" in frame.columns:
        with_metric = frame[frame["best_metric"].notna()]
        if not with_metric.empty:
            frame = with_metric
    if "end_time" in frame.columns:
        frame = frame.sort_values("end_time", ascending=False)
    return frame.iloc[0]


def match_mlflow_to_fid_cell(
    mlflow_df: pd.DataFrame,
    model: str,
    hyperparameters: dict[str, Any],
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return MLflow runs whose grid params match a FID cell."""
    subset = mlflow_df[mlflow_df["model_type"] == model].copy()
    if seed is not None and "seed" in subset.columns:
        subset = subset[subset["seed"] == seed]
    if subset.empty:
        return subset

    hp = _fid_hyperparameters(model, hyperparameters)
    matches = []
    for _, row in subset.iterrows():
        if _hp_match(row.to_dict(), hp):
            matches.append(row)
    return pd.DataFrame(matches)


def join_fid_checkpoints(
    fid_data: list[dict[str, Any]] | None = None,
    checkpoint_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach checkpoint artifact paths to each FID evaluation row."""
    fid_long = fid_runs_long(fid_data)
    fid_long = fid_long[fid_long["seed"].notna()].copy()
    if checkpoint_catalog is None:
        checkpoint_catalog = catalog_checkpoints()

    if checkpoint_catalog.empty:
        return fid_long

    catalog = checkpoint_catalog[~checkpoint_catalog["legacy"]].copy()
    return fid_long.merge(
        catalog,
        left_on=["model", "slug"],
        right_on=["model_type", "slug"],
        how="left",
    )


def training_summary_by_model(
    mlflow_df: pd.DataFrame,
    *,
    experiment: str | None = "gen-cats",
) -> pd.DataFrame:
    """Aggregate MLflow training outcomes per model family."""
    if mlflow_df.empty:
        return pd.DataFrame()

    frame = mlflow_df
    if experiment is not None and "experiment" in frame.columns:
        frame = frame[frame["experiment"] == experiment]

    rows = []
    for model_type, group in frame.groupby("model_type"):
        finished = group[group["status"] == "FINISHED"]
        timed = finished
        if "duration_s" in finished.columns and "final_epoch" in finished.columns:
            epochs = pd.to_numeric(finished["final_epoch"], errors="coerce")
            timed = finished[
                finished["duration_s"].notna() & epochs.notna() & (epochs > 0)
            ].copy()
        rows.append(
            {
                "model_type": model_type,
                "model_label": MODEL_LABELS.get(model_type, model_type),
                "runs_total": len(group),
                "runs_finished": int((group["status"] == "FINISHED").sum()),
                "runs_failed": int((group["status"] == "FAILED").sum()),
                "mean_final_epoch": float(finished["final_epoch"].mean())
                if "final_epoch" in finished.columns and not finished.empty
                else np.nan,
                "median_final_epoch": float(finished["final_epoch"].median())
                if "final_epoch" in finished.columns and not finished.empty
                else np.nan,
                "early_stop_rate": float(finished["early_stopped"].mean())
                if "early_stopped" in finished.columns and not finished.empty
                else np.nan,
                "mean_best_metric": float(finished["best_metric"].mean())
                if "best_metric" in finished.columns and not finished.empty
                else np.nan,
                "median_duration_min": float(timed["duration_s"].median() / 60.0)
                if "duration_s" in timed.columns and not timed.empty
                else np.nan,
                "mean_duration_min": float(timed["duration_s"].mean() / 60.0)
                if "duration_s" in timed.columns and not timed.empty
                else np.nan,
                "median_sec_per_epoch": float(timed["sec_per_epoch"].median())
                if "sec_per_epoch" in timed.columns and not timed.empty
                else np.nan,
                "mean_sec_per_epoch": float(timed["sec_per_epoch"].mean())
                if "sec_per_epoch" in timed.columns and not timed.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("runs_total", ascending=False)


def join_fid_mlflow_metrics(
    fid_data: list[dict[str, Any]] | None = None,
    mlflow_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per FID eval with the best matching MLflow run metrics."""
    if fid_data is None:
        fid_data = load_fid_scores()
    if mlflow_df is None:
        mlflow_df = load_mlflow_runs()

    rows: list[dict[str, Any]] = []
    for entry in fid_data:
        model = entry["model"]
        for run in entry.get("runs", []):
            hp = run.get("hyperparameters", {})
            slug = run.get("slug", "")
            for seed, fid in run.get("per_seed", {}).items():
                matched = match_mlflow_to_fid_cell(
                    mlflow_df,
                    model,
                    hp,
                    seed=int(seed),
                )
                picked = pick_mlflow_run(matched)
                rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "fid": float(fid),
                        "slug": slug,
                        "mlflow_status": None if picked is None else picked.get("status"),
                        "best_metric": None if picked is None else picked.get("best_metric"),
                        "final_epoch": None if picked is None else picked.get("final_epoch"),
                        "run_uuid": None if picked is None else picked.get("run_uuid"),
                    }
                )
    return pd.DataFrame(rows)


def export_json_summary(path: Path | str, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
