# Gen-Cats: Generative Models for Cat Image Synthesis

Academic project comparing **VAE**, **GAN**, and **Diffusion** models for generating 128×128 cat face images. Trained on Apple M4 (`mps` backend).

## Architecture

```
src/gen_cats/
├── config.py          # Dataclass configs with grid search support
├── factory.py         # Model / optimizer / dataset factories
├── data/              # Dataset loading, .cat parsing, face cropping, .npy I/O
├── models/            # β-VAE, VQ-VAE-1, WGAN-GP, SN-GAN, DDIM, Tiny LDM
├── training/          # BaseTrainer (Template Method) + specialized trainers
└── evaluation/        # FID (InceptionV3), latent interpolation
```

**Design patterns:** Factory (model init), Template Method (BaseTrainer → train_step), Strategy (swappable loss/scheduler). All configs via `dataclasses` with grid iteration.

## Models

| Family | Model | Grid Params |
|--------|-------|-------------|
| VAE | β-VAE | latent_dim ∈ {64,128}, β ∈ {1.0,4.0} |
| VAE | VQ-VAE-1 | codebook ∈ {512,1024}, feat_map ∈ {16×16,8×8}, loss ∈ {L1,L2} |
| GAN | WGAN-GP | n_critic ∈ {3,5}, batch ∈ {64,128}, lr ∈ {sym,TTUR} |
| GAN | SN-GAN | batch ∈ {64,128}, lr ∈ {sym,TTUR}, aug ∈ {none,hflip} |
| DM | DDIM | schedule ∈ {linear,cosine}, base_ch ∈ {32,64} |
| DM | Tiny LDM | frozen VQ-VAE encoder, DDIM steps ∈ {50,100,200} |

Each config × 3 seeds. Budget: 100 epochs/run, early stopping (patience=15).

## Quick Start

```bash
# Setup
make setup

# Download & process data
make download-data
make process-data

# Train single model
make train-vae MODEL=beta_vae
make train-gan MODEL=wgan_gp
make train-dm  MODEL=ddim

# Run full grid sweep
make sweep-vae
make sweep-gan
make sweep-dm
make run-all

# Evaluate
make eval-fid
make interpolate
make chimera

# MLflow UI
make mlflow
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd gen-cats
make setup          # creates venv, installs deps
cp .env.example .env  # set KAGGLE_USERNAME, KAGGLE_KEY
```

## Project Structure

- `configs/model/` — YAML model configs
- `scripts/` — CLI entry points (train, sweep, evaluate, interpolate, chimera)
- `data/raw/` — original Kaggle images (gitignored)
- `data/processed/` — cropped 128×128 `.npy` files (gitignored)
- `report/` — LaTeX report

## Reproducibility

Every run seeds `random`, `numpy`, `torch` before training. Results logged to local MLflow with hyperparams, losses, and sample grids (4×4).

## Hardware

Optimized for Apple M4 unified memory: `.npy` pre-loading, `mps` device, batch sizes tuned for ~16GB RAM.
