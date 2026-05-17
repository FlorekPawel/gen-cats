# Gen-Cats: Generative Models for Cat Image Synthesis

Academic project comparing **VAE**, **GAN** and **diffusion** for generating 128×128 cat face images. Trained on Apple M4 (`mps` backend).

Run `make help` for all targets.

## Architecture

```
src/gen_cats/
├── config.py              # TrainConfig dataclass + grid search helpers
├── factory.py             # Model / optimizer / dataloader / trainer factories
├── data/                  # CatDataset, .cat parsing, face crop, .npy I/O
├── models/                # β-VAE, VQ-VAE, GANs, U-Net/DDIM, PixelCNN, checkpoint loaders
├── training/              # BaseTrainer + VAE / GAN / DM / PixelCNN trainers
└── evaluation/            # FID, latent interpolation
scripts/                   # invoked by Makefile targets
```

**Design patterns:** Factory (model init), Template Method (`BaseTrainer` → `train_step`), Strategy (per-family losses). Hyperparameters live in `TrainConfig`; sweeps iterate grids × seeds via `ExperimentRunner`.

## Models

| Family | Model | Role | Grid / notes |
|--------|-------|------|----------------|
| VAE | β-VAE | Continuous latent | `latent_dim` ∈ {64, 128}, `beta` ∈ {1.0, 4.0} |
| VAE | VQ-VAE-1 | Discrete codes + decoder | codebook ∈ {512, 1024}, map ∈ {16×16, 8×8}, recon ∈ {L1, L2} |
| GAN | WGAN-GP | Adversarial | `n_critic` ∈ {3, 5}, batch, symmetric vs TTUR LR |
| GAN | SN-GAN | Adversarial + spectral norm | batch, LR, optional D augmentation |
| DM | DDIM | Pixel-space diffusion | schedule ∈ {linear, cosine}, `base_channels` ∈ {32, 64} |
| DM | Tiny LDM | Diffusion in **frozen VQ-VAE** latent space | Uses best VQ-VAE checkpoint; DDIM steps at inference |
| Prior | **PixelCNN** | Autoregressive prior over VQ **code indices** | Lightweight baseline vs Tiny LDM (Milestone 7) |

Grid sweeps run each config × **3 seeds**. Training uses early stopping (`patience=15`, `min_epochs=20`) with a high epoch cap (`max_epochs=1000`). Checkpoints are namespaced per run: `checkpoints/<model_type>/<slug>/best_seed{seed}.pt`.

### Unconditional sampling (what each model uses at generation time)

| Model | Sampling |
|-------|----------|
| β-VAE | `z ~ N(0, I)` → decoder |
| VQ-VAE (trainer grids only) | Random codebook indices (debug / not a learned prior) |
| **PixelCNN** | Autoregressive code indices on the VQ grid → `decode_indices` |
| **Tiny LDM** | DDIM in latent space → frozen VQ decoder |
| GAN | `z ~ N(0, I)` → generator |

For fair VQ-VAE generation comparisons, use **PixelCNN** or **Tiny LDM**, not the random-index path in `VAETrainer.generate_samples`.

## Quick Start

```bash
make setup
cp .env.example .env   # KAGGLE_USERNAME, KAGGLE_KEY

make download-data
make process-data

# Single runs (MODEL / SEED / EPOCHS / VQVAE_SEED where applicable)
make train-vae MODEL=beta_vae SEED=42
make train-vae MODEL=vqvae SEED=42
make train-gan MODEL=wgan_gp SEED=42
make train-dm MODEL=ddim SEED=42
make train-dm MODEL=tiny_ldm SEED=42 VQVAE_SEED=42

make train-pixelcnn VQVAE_SEED=42 SEED=42 EPOCHS=80
make compare-priors PIXELCNN_SEED=42 LDM_SEED=42 VQVAE_SEED=42

make sweep-vae
make sweep-gan
make sweep-dm
make run-all

make eval-fid
make interpolate
make chimera
make mlflow
```

### Dogs vs Cats (chimera experiment)

```bash
make download-dogcat
make process-dogcat
make train-gan MODEL=sn_gan SEED=42
make chimera
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd gen-cats
make setup
cp .env.example .env
```

## Project structure

| Path | Description |
|------|-------------|
| `Makefile` | Primary CLI (`make help`) |
| `scripts/` | Training, sweeps, FID, interpolation, chimera, prior comparison |
| `data/raw/` | Kaggle downloads (gitignored) |
| `data/processed/` | Cropped 128×128 `.npy` tensors (gitignored) |
| `checkpoints/` | Per-model, per-run checkpoints (gitignored) |
| `results/` | FID outputs, prior comparison, etc. |
| `report/` | LaTeX report |

## Reproducibility

Each run resets `random`, `numpy`, and `torch` seeds. MLflow logs hyperparameters, train/val metrics, and 4×4 sample grids. Finished runs (including early-stopped) are marked `finished` in checkpoints and are **not** resumed on restart.

## Development

```bash
make test
make lint
make format
```
