# gen-cats

Comparative study of variational, adversarial, and diffusion models for unconditional cat face synthesis at 128×128. Full report available in [`report/report.pdf`](report/report.pdf); code and instructions below.

**Dataset:** [Kaggle Cat Dataset](https://www.kaggle.com/datasets/crawford/cat-dataset) — faces cropped from `.cat` landmark annotations, resized with Lanczos, normalized to `[-1, 1]`.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Kaggle API credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`)

## Setup

```bash
git clone <repo-url> && cd gen-cats
make setup
cp .env.example .env   # fill in Kaggle credentials
```

All commands below assume you are in the repo root. Run `make help` for the full target list.

## Experiment workflow

Cat-face experiments follow a fixed order. Training is long; `make mlflow` (port 5050) is useful for monitoring.

```bash
# 1. Data
make download-data
make process-data          # → data/processed/train.npy, val.npy

# 2. Training (132 grid jobs + 3 PixelCNN + 3 fixed Tiny LDM for prior comparison)
make run-all

# 3. Evaluation
make eval                  # FID → results/fid_scores.json; interpolations → results/interpolations/
make additional-samples    # sample grids from best checkpoints

# 4. Report figures (re-run after process-data for real crops)
make report-figures        # → notebooks/plots/
```

**Chimera extension** (optional, 64×64 WGAN-GP on mixed dogs and cats):

```bash
make download-dogcat
make process-dogcat-chimera
make chimera               # +3 WGAN-GP runs → checkpoints/chimera/
```

### Run counts

| Stage | Jobs |
|-------|-----:|
| Grid sweeps (VAE, GAN, diffusion families) | 132 |
| PixelCNN prior (3 seeds, fixed architecture) | 3 |
| Tiny LDM for prior comparison (fixed: EMA on, 100 DDIM steps) | 3 |
| Chimera WGAN-GP | 3 |
| **Total** | **141** |

Each hyperparameter cell is trained with seeds `{42, 0, 3407}`. Early stopping: `patience=15`, `min_epochs=20`, `max_epochs=1000`. Default device is Apple MPS (`TrainConfig.device`).

### Single runs

For debugging one configuration instead of a full sweep:

```bash
make train-vae MODEL=beta_vae SEED=42
make train-gan MODEL=wgan_gp SEED=42
make train-dm MODEL=ddim SEED=42
make train-pixelcnn SEED=42
```

Pin a frozen VQ-VAE for PixelCNN or Tiny LDM with `NUM_EMBEDDINGS`, `FEATURE_MAP_SIZE`, and `RECON_LOSS`, or pass `--vqvae-selection slug` via the underlying script.

## Models

Seven generators are compared on the same preprocessed cat faces. Hyperparameter grids are defined in [`src/gen_cats/config.py`](src/gen_cats/config.py) (`GRIDS`).

| Family | Model | What varies in the sweep |
|--------|-------|--------------------------|
| VAE | β-VAE | `latent_dim` ∈ {64, 128}, `beta` ∈ {1.0, 4.0} |
| VAE | VQ-VAE-1 | codebook ∈ {512, 1024}, map ∈ {16², 8²}, recon ∈ {L1, MSE} |
| GAN | WGAN-GP | `n_critic` ∈ {3, 5}, batch ∈ {64, 128}, symmetric vs TTUR LR |
| GAN | SN-GAN | batch, LR schedule, optional D-side hflip augmentation |
| Diffusion | DDIM | schedule, `base_channels`, `ddim_steps`; U-Net bottleneck 8×8; T=1000 |
| Diffusion | Tiny LDM | same grid as DDIM; denoises continuous VQ encoder latents |
| Prior | PixelCNN | 128 channels, 10 masked layers (fixed); AR prior over VQ code indices |

**Sampling at evaluation time**

| Model | Generation |
|-------|------------|
| β-VAE, GAN | `z ~ N(0, I)` → decoder / generator |
| DDIM | DDIM in pixel space (EMA weights when enabled) |
| Tiny LDM | DDIM in latent space → frozen VQ decoder |
| PixelCNN | autoregressive code indices → `decode_indices` |
| VQ-VAE alone | random codebook indices (debug only; not a learned prior) |

For VQ-based synthesis, use PixelCNN or Tiny LDM—not the random-index path in `VAETrainer.generate_samples`.

### VQ-VAE prior selection

PixelCNN and Tiny LDM need a frozen VQ-VAE per seed. After `make sweep-vae`, `select-vqvae-priors` (also invoked by `run-all`) writes `checkpoints/vqvae/prior_best_by_seed.json`: for each seed, the grid cell with lowest validation reconstruction loss. `pixelcnn-experiment` then trains PixelCNN on that manifest and, separately, three fixed-config Tiny LDM runs for wall-clock comparison (`results/prior_comparison/`).

## Repository layout

```
src/gen_cats/
  config.py          # TrainConfig, GRIDS, SEEDS
  factory.py           # model / trainer / dataloader construction
  data/                # parsing, crop, .npy I/O
  models/              # architectures and checkpoint loading
  training/            # trainers and ExperimentRunner
  evaluation/          # FID, interpolation, prior comparison
scripts/               # CLI entry points (called by Makefile)
data/raw/              # Kaggle downloads (gitignored)
data/processed/        # train.npy, val.npy (gitignored)
checkpoints/           # per-run weights (gitignored)
results/               # metrics and figures from eval
report/                # LaTeX source and bibliography
notebooks/             # EDA; plots consumed by the report
```

Checkpoints are stored as `checkpoints/<model_type>/<slug>/best_seed{seed}.pt`. Slugs come from `run_name` or a hash of hyperparameters so runs do not overwrite each other.

**Results analysis.** Open and run `notebooks/00_results_overview.ipynb` through `09_*.ipynb` (figures → `notebooks/plots/results/`, numeric exports → `notebooks/report_snippets/`).

**Key outputs**

| Path | Contents |
|------|----------|
| `results/fid_scores.json` | FID per model family and hyperparameter cell |
| `results/interpolations/` | β-VAE and WGAN-GP latent interpolation strips |
| `results/prior_comparison/` | PixelCNN vs Tiny LDM timings and sample grids |
| `checkpoints/chimera/` | 64×64 mixed-species WGAN-GP |
| `notebooks/plots/` | Figures for the report (`make report-figures`) |

Finished runs are marked in checkpoint metadata and are not resumed on restart. MLflow logs hyperparameters, train/val curves, periodic sample grids, and a final `samples_best.png` from the best checkpoint.

## Report

```bash
cd report
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
```

Regenerate figure PNGs after processing data: `make report-figures`.

## Development

```bash
make test
make lint
make format
```

Pre-commit hooks install with `make setup`.
