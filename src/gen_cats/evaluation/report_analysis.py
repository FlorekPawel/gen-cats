"""Load experiment results and build report tables / figure paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"
SNIPPETS_DIR = REPO_ROOT / "notebooks" / "report_snippets"

FID_PATH = RESULTS_DIR / "fid_scores.json"
INTERPOLATIONS_DIR = RESULTS_DIR / "interpolations"
PRIOR_DIR = RESULTS_DIR / "prior_comparison"
CHIMERA_DIR = REPO_ROOT / "checkpoints" / "chimera"

MODEL_LABELS: dict[str, str] = {
    "beta_vae": r"$\beta$-VAE",
    "vqvae": "VQ-VAE-1",
    "wgan_gp": "WGAN-GP",
    "sn_gan": "SN-GAN",
    "ddim": "DDIM",
    "tiny_ldm": "Tiny LDM",
    "pixelcnn": "PixelCNN",
}

FAMILY_ORDER: list[str] = [
    "beta_vae",
    "vqvae",
    "pixelcnn",
    "tiny_ldm",
    "wgan_gp",
    "sn_gan",
    "ddim",
]

SAMPLE_PREFERENCE = ("additional_samples.png", "samples_best.png")


def load_fid_scores(path: Path | str = FID_PATH) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def fid_by_model(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["model"]: entry for entry in data}


def best_runs_table(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model family: best grid cell from ``best_run``."""
    rows: list[dict[str, Any]] = []
    for model in FAMILY_ORDER:
        entry = fid_by_model(data).get(model, {})
        best = entry.get("best_run")
        label = MODEL_LABELS.get(model, model)
        if not best:
            rows.append(
                {
                    "model": model,
                    "label": label,
                    "mean_fid": np.nan,
                    "slug": "",
                    "hyperparameters": {},
                    "n_runs": entry.get("n_runs", 0),
                }
            )
            continue
        rows.append(
            {
                "model": model,
                "label": label,
                "mean_fid": float(best["mean_fid"]),
                "slug": best.get("slug", ""),
                "hyperparameters": best.get("hyperparameters", {}),
                "n_runs": entry.get("n_runs", 0),
            }
        )
    return rows


def format_hyperparameters(model: str, hyperparameters: dict[str, Any]) -> str:
    if not hyperparameters:
        return "---"
    if model == "beta_vae":
        return f"z={hyperparameters.get('latent_dim')}, $\\beta$={hyperparameters.get('beta')}"
    if model == "vqvae":
        return (
            f"K={hyperparameters.get('num_embeddings')}, "
            f"{hyperparameters.get('feature_map_size')}$\\times$"
            f"{hyperparameters.get('feature_map_size')}, "
            f"{hyperparameters.get('recon_loss', '').upper()}"
        )
    if model in {"wgan_gp", "sn_gan"}:
        lr_d = hyperparameters.get("lr_d")
        lr_g = hyperparameters.get("lr_g", hyperparameters.get("lr"))
        sym = "sym" if lr_d == lr_g else "TTUR"
        aug = ""
        if model == "sn_gan":
            aug = ", aug" if hyperparameters.get("d_augment") else ", no aug"
        return (
            f"n$_c$={hyperparameters.get('n_critic')}, "
            f"batch={hyperparameters.get('batch_size')}, {sym}{aug}"
        )
    if model in {"ddim", "tiny_ldm"}:
        return (
            f"{hyperparameters.get('noise_schedule')}, "
            f"ch={hyperparameters.get('base_channels')}, "
            f"steps={hyperparameters.get('ddim_steps')}"
        )
    if model == "pixelcnn":
        return (
            f"VQ {hyperparameters.get('feature_map_size')}$\\times$"
            f"{hyperparameters.get('feature_map_size')}, "
            f"K={hyperparameters.get('num_embeddings')}"
        )
    parts = [f"{key}={value}" for key, value in sorted(hyperparameters.items())]
    return ", ".join(parts[:4])


def latex_fid_table(rows: list[dict[str, Any]]) -> str:
    """Booktabs-style LaTeX rows for the best-run FID summary."""
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Best hyperparameter cell per model family (lowest mean FID).}",
        r"    \label{tab:fid_best}",
        r"    \begin{tabular}{lrl}",
        r"        \toprule",
        r"        \textbf{Model} & \textbf{FID} $\downarrow$ & \textbf{Best config} \\",
        r"        \midrule",
    ]
    ranked = sorted(
        [row for row in rows if not np.isnan(row["mean_fid"])],
        key=lambda row: row["mean_fid"],
    )
    for row in ranked:
        fid = f"{row['mean_fid']:.1f}"
        config = format_hyperparameters(row["model"], row["hyperparameters"])
        lines.append(f"        {row['label']} & {fid} & {config} \\\\")
    missing = [row["label"] for row in rows if np.isnan(row["mean_fid"])]
    if missing:
        lines.append(r"        \midrule")
        lines.append(
            "        \\multicolumn{3}{l}{\\textit{No FID eval: " + ", ".join(missing) + "}} \\\\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def resolve_sample_image(
    checkpoint_dir: Path | str,
    model_type: str,
    slug: str,
) -> Path | None:
    """Return ``additional_samples.png`` or ``samples_best.png`` for a run slug."""
    base = Path(checkpoint_dir) / model_type / slug
    for name in SAMPLE_PREFERENCE:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def best_run_sample_path(
    row: dict[str, Any],
    checkpoint_dir: Path | str = REPO_ROOT / "checkpoints",
) -> Path | None:
    slug = row.get("slug", "")
    model = row.get("model", "")
    if not slug or not model:
        return None
    return resolve_sample_image(checkpoint_dir, model, slug)


def list_interpolation_images(directory: Path | str = INTERPOLATIONS_DIR) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("interpolation_*.png"))


def load_prior_summary(path: Path | str = PRIOR_DIR / "summary.json") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def prior_seed_assets(seed: int, directory: Path | str = PRIOR_DIR) -> dict[str, Path]:
    root = Path(directory) / f"seed_{seed}"
    return {
        "combined": root / "combined_comparison.png",
        "pixelcnn": root / "pixelcnn_samples.png",
        "tiny_ldm": root / "tiny_ldm_samples.png",
        "json": root / "comparison.json",
    }


def chimera_sample_paths(checkpoint_dir: Path | str = CHIMERA_DIR) -> list[Path]:
    root = Path(checkpoint_dir)
    paths: list[Path] = []
    for seed_dir in sorted(root.glob("wgan_gp/chimera_64_seed*")):
        for name in SAMPLE_PREFERENCE:
            candidate = seed_dir / name
            if candidate.is_file():
                paths.append(candidate)
                break
    return paths


def write_snippet(name: str, content: str, directory: Path | str = SNIPPETS_DIR) -> Path:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def write_stats_json(name: str, payload: Any, directory: Path | str = SNIPPETS_DIR) -> Path:
    return write_snippet(name, json.dumps(payload, indent=2, default=str) + "\n", directory)


def write_stats_csv(
    frame: Any,
    name: str,
    *,
    directory: Path | str = SNIPPETS_DIR,
) -> Path:
    import pandas as pd

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if not isinstance(frame, pd.DataFrame):
        msg = "write_stats_csv expects a pandas DataFrame"
        raise TypeError(msg)
    frame.to_csv(path, index=False)
    return path


def ensure_plots_dir(subdir: str = "results") -> Path:
    out = PLOTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out
