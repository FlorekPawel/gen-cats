"""Tests for generative model architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from gen_cats.config import TrainConfig
from gen_cats.models.beta_vae import BetaVAE
from gen_cats.models.ddim import DDIMScheduler, cosine_beta_schedule, linear_beta_schedule
from gen_cats.models.discriminator import Discriminator, compute_gradient_penalty
from gen_cats.models.generator import Generator
from gen_cats.models.pixelcnn import PixelCNN
from gen_cats.models.unet import UNet
from gen_cats.models.vqvae import VQVAE
from gen_cats.training.dm_trainer import DiffusionTrainer
from gen_cats.training.gan_trainer import GANTrainer
from gen_cats.training.pixelcnn_trainer import PixelCNNTrainer
from gen_cats.training.vae_trainer import VAETrainer
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"
B = 4


def _dummy_loaders() -> tuple[DataLoader[Any], DataLoader[Any]]:
    data = torch.randn(16, 3, 128, 128)
    loader: DataLoader[Any] = DataLoader(TensorDataset(data), batch_size=B)
    return loader, loader


class TestBetaVAE:
    def test_forward_shape(self) -> None:
        model = BetaVAE(latent_dim=64, beta=1.0)
        x = torch.randn(B, 3, 128, 128)
        recon, mu, logvar = model(x)
        assert recon.shape == (B, 3, 128, 128)
        assert mu.shape == (B, 64)
        assert logvar.shape == (B, 64)

    def test_sample_shape(self) -> None:
        model = BetaVAE(latent_dim=128)
        samples = model.sample(8, torch.device(DEVICE))
        assert samples.shape == (8, 3, 128, 128)

    def test_loss_keys(self) -> None:
        model = BetaVAE(latent_dim=64, beta=2.0)
        x = torch.randn(B, 3, 128, 128)
        recon, mu, logvar = model(x)
        losses = model.loss(x, recon, mu, logvar)
        assert "total" in losses
        assert "recon" in losses
        assert "kl" in losses

    def test_loss_l1(self) -> None:
        model = BetaVAE(latent_dim=64)
        x = torch.randn(B, 3, 128, 128)
        recon, mu, logvar = model(x)
        losses = model.loss(x, recon, mu, logvar, recon_type="l1")
        assert losses["recon"].item() > 0

    def test_latent_dims(self) -> None:
        for dim in [64, 128]:
            model = BetaVAE(latent_dim=dim)
            x = torch.randn(2, 3, 128, 128)
            _, mu, _ = model(x)
            assert mu.shape[1] == dim


class TestVQVAE:
    def test_forward_shape_16x16(self) -> None:
        model = VQVAE(num_embeddings=512, embedding_dim=64, feature_map_size=16)
        x = torch.randn(B, 3, 128, 128)
        recon, vq_loss, indices = model(x)
        assert recon.shape == (B, 3, 128, 128)
        assert vq_loss.dim() == 0
        assert indices.shape[0] == B * 16 * 16

    def test_forward_shape_8x8(self) -> None:
        model = VQVAE(num_embeddings=1024, embedding_dim=64, feature_map_size=8)
        x = torch.randn(B, 3, 128, 128)
        recon, _vq_loss, indices = model(x)
        assert recon.shape == (B, 3, 128, 128)
        assert indices.shape[0] == B * 8 * 8

    def test_encode_decode(self) -> None:
        model = VQVAE(feature_map_size=16, embedding_dim=64)
        x = torch.randn(B, 3, 128, 128)
        z_q = model.encode(x)
        assert z_q.shape == (B, 64, 16, 16)
        recon = model.decode(z_q)
        assert recon.shape == (B, 3, 128, 128)

    def test_loss_keys(self) -> None:
        model = VQVAE()
        x = torch.randn(B, 3, 128, 128)
        recon, vq_loss, _ = model(x)
        losses = model.loss(x, recon, vq_loss)
        assert "total" in losses
        assert "recon" in losses
        assert "vq" in losses

    def test_encode_decode_indices(self) -> None:
        model = VQVAE(num_embeddings=64, embedding_dim=32, feature_map_size=8)
        x = torch.randn(B, 3, 128, 128)
        indices = model.encode_indices(x)
        assert indices.shape == (B, 8, 8)
        recon = model.decode_indices(indices)
        assert recon.shape == (B, 3, 128, 128)


class TestPixelCNN:
    def test_forward_shape(self) -> None:
        k, h = 64, 8
        model = PixelCNN(num_embeddings=k, hidden_channels=32, n_layers=2)
        indices = torch.randint(0, k, (B, h, h))
        logits = model(indices)
        assert logits.shape == (B, k, h, h)

    def test_sample_shape(self) -> None:
        model = PixelCNN(num_embeddings=32, hidden_channels=16, n_layers=2)
        out = model.sample(3, spatial_size=4, device=torch.device(DEVICE))
        assert out.shape == (3, 4, 4)


class TestVAETrainer:
    @patch("gen_cats.training.base_trainer.mlflow")
    def test_beta_vae_train(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="beta_vae",
            device=DEVICE,
            max_epochs=2,
            latent_dim=64,
            beta=1.0,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
            batch_size=B,
        )
        trainer = VAETrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 2

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_vqvae_train(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="vqvae",
            device=DEVICE,
            max_epochs=2,
            num_embeddings=64,
            embedding_dim=32,
            feature_map_size=16,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
            batch_size=B,
        )
        trainer = VAETrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 2

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_vae_generates_samples(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="beta_vae",
            device=DEVICE,
            latent_dim=64,
            checkpoint_dir=str(tmp_path),
        )
        trainer = VAETrainer(cfg)
        trainer.build_models()
        samples = trainer.generate_samples(8)
        assert samples.shape == (8, 3, 128, 128)


# ── GAN Tests ────────────────────────────────────────────────────


class TestGenerator:
    def test_output_shape(self) -> None:
        g = Generator(latent_dim=128)
        z = torch.randn(B, 128)
        out = g(z)
        assert out.shape == (B, 3, 128, 128)

    def test_output_range(self) -> None:
        g = Generator(latent_dim=64)
        z = torch.randn(B, 64)
        out = g(z)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestDiscriminator:
    def test_output_shape(self) -> None:
        d = Discriminator()
        x = torch.randn(B, 3, 128, 128)
        out = d(x)
        assert out.shape == (B,)

    def test_spectral_norm(self) -> None:
        d = Discriminator(use_spectral_norm=True)
        x = torch.randn(B, 3, 128, 128)
        out = d(x)
        assert out.shape == (B,)

    def test_gradient_penalty(self) -> None:
        d = Discriminator()
        real = torch.randn(B, 3, 128, 128, requires_grad=True)
        fake = torch.randn(B, 3, 128, 128)
        gp = compute_gradient_penalty(d, real, fake, torch.device(DEVICE))
        assert gp.dim() == 0
        assert gp.item() >= 0


class TestGANTrainer:
    @patch("gen_cats.training.base_trainer.mlflow")
    def test_wgan_gp_train(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="wgan_gp",
            device=DEVICE,
            max_epochs=2,
            latent_dim=64,
            n_critic=2,
            batch_size=B,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
        )
        trainer = GANTrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 2

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_sn_gan_train(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="sn_gan",
            device=DEVICE,
            max_epochs=2,
            latent_dim=64,
            n_critic=1,
            batch_size=B,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
        )
        trainer = GANTrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 2

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_gan_generates_samples(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="wgan_gp",
            device=DEVICE,
            latent_dim=64,
            checkpoint_dir=str(tmp_path),
        )
        trainer = GANTrainer(cfg)
        trainer.build_models()
        samples = trainer.generate_samples(8)
        assert samples.shape == (8, 3, 128, 128)


# ── Diffusion Tests ──────────────────────────────────────────────


class TestUNet:
    def test_output_shape(self) -> None:
        model = UNet(in_ch=3, base_ch=32)
        x = torch.randn(B, 3, 128, 128)
        t = torch.randint(0, 100, (B,))
        out = model(x, t)
        assert out.shape == (B, 3, 128, 128)

    def test_latent_input(self) -> None:
        model = UNet(in_ch=64, base_ch=32)
        x = torch.randn(B, 64, 16, 16)
        t = torch.randint(0, 100, (B,))
        out = model(x, t)
        assert out.shape == (B, 64, 16, 16)


class TestDDIMScheduler:
    def test_linear_schedule(self) -> None:
        betas = linear_beta_schedule(100)
        assert betas.shape == (100,)
        assert betas[0] < betas[-1]

    def test_cosine_schedule(self) -> None:
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)
        assert (betas >= 0).all()
        assert (betas < 1).all()

    def test_q_sample_shape(self) -> None:
        scheduler = DDIMScheduler(timesteps=100, schedule="linear")
        x0 = torch.randn(B, 3, 128, 128)
        t = torch.randint(0, 100, (B,))
        noisy, noise = scheduler.q_sample(x0, t)
        assert noisy.shape == x0.shape
        assert noise.shape == x0.shape

    def test_ddim_sample_shape(self) -> None:
        model = UNet(in_ch=3, base_ch=16)
        scheduler = DDIMScheduler(timesteps=100, schedule="linear")
        samples = scheduler.ddim_sample(model, (2, 3, 128, 128), torch.device("cpu"), ddim_steps=5)
        assert samples.shape == (2, 3, 128, 128)


class TestDiffusionTrainer:
    @patch("gen_cats.training.base_trainer.mlflow")
    def test_ddim_train(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="ddim",
            device=DEVICE,
            max_epochs=2,
            base_channels=16,
            timesteps=50,
            ddim_steps=5,
            noise_schedule="linear",
            batch_size=B,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
        )
        trainer = DiffusionTrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 2

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_ddim_generates_samples(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="ddim",
            device=DEVICE,
            base_channels=16,
            timesteps=50,
            ddim_steps=5,
            checkpoint_dir=str(tmp_path),
        )
        trainer = DiffusionTrainer(cfg)
        trainer.build_models()
        samples = trainer.generate_samples(4)
        assert samples.shape == (4, 3, 128, 128)

    @patch("gen_cats.training.base_trainer.mlflow")
    def test_ddim_cosine_schedule(self, _mock_mlflow: Any, tmp_path: Any) -> None:
        cfg = TrainConfig(
            model_type="ddim",
            device=DEVICE,
            max_epochs=1,
            base_channels=16,
            timesteps=50,
            ddim_steps=5,
            noise_schedule="cosine",
            batch_size=B,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
        )
        trainer = DiffusionTrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 1


class TestPixelCNNTrainer:
    @patch("gen_cats.training.pixelcnn_trainer.load_frozen_vqvae")
    @patch("gen_cats.training.base_trainer.mlflow")
    def test_pixelcnn_train(self, _mock_mlflow: Any, mock_vqvae: Any, tmp_path: Any) -> None:
        vqvae = VQVAE(num_embeddings=64, embedding_dim=32, feature_map_size=8)
        mock_vqvae.return_value = (
            vqvae,
            {"num_embeddings": 64, "feature_map_size": 8},
            Path("fake.pt"),
        )
        cfg = TrainConfig(
            model_type="pixelcnn",
            device=DEVICE,
            max_epochs=2,
            prior_hidden_channels=32,
            prior_n_layers=2,
            batch_size=B,
            checkpoint_dir=str(tmp_path),
            patience=50,
            sample_interval=100,
            min_epochs=0,
        )
        trainer = PixelCNNTrainer(cfg)
        train_loader, val_loader = _dummy_loaders()
        results = trainer.fit(train_loader, val_loader)
        assert results["final_epoch"] == 2

    @patch("gen_cats.training.pixelcnn_trainer.load_frozen_vqvae")
    def test_pixelcnn_generate(self, mock_vqvae: Any, tmp_path: Any) -> None:
        vqvae = VQVAE(num_embeddings=64, embedding_dim=32, feature_map_size=8)
        mock_vqvae.return_value = (vqvae, {}, Path("fake.pt"))
        cfg = TrainConfig(
            model_type="pixelcnn",
            device=DEVICE,
            prior_hidden_channels=32,
            prior_n_layers=2,
            checkpoint_dir=str(tmp_path),
        )
        trainer = PixelCNNTrainer(cfg)
        trainer.build_models()
        samples = trainer.generate_samples(2)
        assert samples.shape == (2, 3, 128, 128)
