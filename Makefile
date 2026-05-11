PYTHON := uv run python
MLFLOW_PORT := 5050

.PHONY: help setup download-data download-dogcat process-data process-dogcat \
        train-vae train-gan train-dm \
        sweep-vae sweep-gan sweep-dm run-all \
        eval-fid interpolate chimera \
        mlflow test lint format pre-commit pre-commit-all clean

help:
	@echo "Available targets:"
	@echo "  make setup            - install deps and pre-commit hooks"
	@echo "  make download-data    - download Cat Dataset from Kaggle"
	@echo "  make download-dogcat  - download Dogs vs Cats from Kaggle"
	@echo "  make process-data     - process cat images into .npy files"
	@echo "  make process-dogcat   - process dogs+cats images into .npy"
	@echo "  make train-vae        - train VAE model (MODEL=beta_vae|vqvae)"
	@echo "  make train-gan        - train GAN model (MODEL=wgan_gp|sn_gan)"
	@echo "  make train-dm         - train diffusion model (MODEL=ddim|tiny_ldm)"
	@echo "  make sweep-vae        - run VAE grid sweep"
	@echo "  make sweep-gan        - run GAN grid sweep"
	@echo "  make sweep-dm         - run diffusion grid sweep"
	@echo "  make run-all          - run all sweeps"
	@echo "  make eval-fid         - compute FID metrics"
	@echo "  make interpolate      - generate interpolation strips"
	@echo "  make chimera          - run Dogs vs Cats chimera experiment"
	@echo "  make mlflow           - start local MLflow UI"
	@echo "  make test             - run tests"
	@echo "  make lint             - run lint checks"
	@echo "  make format           - format and autofix lint issues"
	@echo "  make pre-commit       - run pre-commit on staged files"
	@echo "  make pre-commit-all   - run pre-commit on all files"
	@echo "  make clean            - remove generated artifacts"

# ─── Environment ──────────────────────────────────────────────
setup:
	uv sync --all-groups
	uv run pre-commit install

# ─── Data Pipeline ────────────────────────────────────────────
download-data:
	$(PYTHON) scripts/download_data.py --dataset cats

download-dogcat:
	$(PYTHON) scripts/download_data.py --dataset dogcat

process-data:
	$(PYTHON) scripts/process_data.py --dataset cats

process-dogcat:
	$(PYTHON) scripts/process_data.py --dataset dogcat

# ─── Training (single model) ─────────────────────────────────
train-vae:
	$(PYTHON) scripts/train.py --config configs/model/$(or $(MODEL),beta_vae).yaml

train-gan:
	$(PYTHON) scripts/train.py --config configs/model/$(or $(MODEL),wgan_gp).yaml

train-dm:
	$(PYTHON) scripts/train.py --config configs/model/$(or $(MODEL),ddim).yaml

# ─── Grid Sweeps (all configs × 3 seeds) ─────────────────────
sweep-vae:
	$(PYTHON) scripts/run_sweep.py --family vae

sweep-gan:
	$(PYTHON) scripts/run_sweep.py --family gan

sweep-dm:
	$(PYTHON) scripts/run_sweep.py --family dm

run-all: sweep-vae sweep-gan sweep-dm

# ─── Evaluation ───────────────────────────────────────────────
eval-fid:
	$(PYTHON) scripts/evaluate.py

interpolate:
	$(PYTHON) scripts/interpolate.py

chimera:
	$(PYTHON) scripts/chimera.py

# ─── MLflow ───────────────────────────────────────────────────
mlflow:
	uv run mlflow ui --port $(MLFLOW_PORT) --backend-store-uri sqlite:///mlflow.db

# ─── Dev Tools ────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ scripts/ tests/

format:
	uv run ruff format src/ scripts/ tests/
	uv run ruff check --fix src/ scripts/ tests/

pre-commit:
	uv run pre-commit run

pre-commit-all:
	uv run pre-commit run --all-files

clean:
	rm -rf data/processed/*.npy
	rm -rf mlruns/ mlflow.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
