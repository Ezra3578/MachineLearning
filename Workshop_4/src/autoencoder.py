"""Autoencoder and Variational Autoencoder architectures for Workshop 4.

Implements:
- ``ConvAE``          : Standard convolutional autoencoder (z_dim in {32, 64, 128}).
- ``ConvVAE``         : Variational autoencoder with reparameterization trick & ELBO.
- ``DenoisingConvAE`` : Denoising autoencoder corrupting inputs during training.

Run ``python -m src.autoencoder`` to verify parameter counts and forward shapes.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    """Select accelerator: CUDA -> MPS -> DirectML -> CPU."""
    req = requested.lower()
    if req == "cpu":
        return torch.device("cpu")
    if req in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if req in ("mps", "auto") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if req in ("dml", "directml", "auto"):
        try:
            import torch_directml  # type: ignore

            return torch_directml.device()
        except Exception:
            if req in ("dml", "directml"):
                raise
    return torch.device("cpu")


class ConvEncoder(nn.Module):
    """Shared convolutional encoder trunk: 1x28x28 -> 32x14x14 -> 64x7x7 (3,136 flat)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
        )
        self.flat_dim = 64 * 7 * 7
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Shared transpose-convolutional decoder: z_dim -> 64x7x7 -> 32x14x14 -> 16x28x28 -> 1x28x28."""

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 64 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class ConvAE(nn.Module):
    """Standard Convolutional Autoencoder."""

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.model_type = "ae"
        self.encoder_trunk = ConvEncoder()
        self.fc_z = nn.Linear(self.encoder_trunk.flat_dim, z_dim)
        self.decoder = ConvDecoder(z_dim=z_dim)
        nn.init.xavier_uniform_(self.fc_z.weight)
        nn.init.zeros_(self.fc_z.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_z(self.encoder_trunk(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class DenoisingConvAE(ConvAE):
    """Denoising Convolutional Autoencoder (Vincent et al. 2010).

    Adds isotropic Gaussian noise during training and reconstructs clean targets.
    """

    def __init__(self, z_dim: int = 64, noise_std: float = 0.25) -> None:
        super().__init__(z_dim=z_dim)
        self.model_type = "dae"
        self.noise_std = noise_std

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0.0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def forward(self, x: torch.Tensor, add_noise: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        use_noise = self.training if add_noise is None else add_noise
        x_in = self.corrupt(x) if use_noise else x
        z = self.encode(x_in)
        x_hat = self.decode(z)
        return x_hat, z


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder (Kingma & Welling 2013; Higgins et al. 2017)."""

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.model_type = "vae"
        self.encoder_trunk = ConvEncoder()
        self.fc_mu = nn.Linear(self.encoder_trunk.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.encoder_trunk.flat_dim, z_dim)
        self.decoder = ConvDecoder(z_dim=z_dim)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def vae_elbo_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-sample summed ELBO = Recon_MSE + beta * KL, averaged over batch."""
    recon_per_sample = F.mse_loss(x_hat, x, reduction="none").view(x.size(0), -1).sum(dim=1)
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    recon = recon_per_sample.mean()
    kl = kl_per_sample.mean()
    total = recon + beta * kl
    return total, recon, kl


def build_autoencoder(model_type: str, z_dim: int = 64, noise_std: float = 0.25) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "ae":
        return ConvAE(z_dim=z_dim)
    if model_type == "vae":
        return ConvVAE(z_dim=z_dim)
    if model_type == "dae":
        return DenoisingConvAE(z_dim=z_dim, noise_std=noise_std)
    raise ValueError(f"Unknown model_type={model_type!r}; expected 'ae', 'vae', or 'dae'")


def load_autoencoder_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load an AE/VAE/DAE checkpoint with weights_only=False compatibility."""
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model", ckpt.get("model_type", "ae"))
    z_dim = int(cfg.get("z_dim", ckpt.get("z_dim", 64)))
    noise_std = float(cfg.get("noise_std", 0.25))
    model = build_autoencoder(model_type=model_type, z_dim=z_dim, noise_std=noise_std)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    sample = torch.rand(4, 1, 28, 28)
    for z_dim in (32, 64, 128):
        ae = ConvAE(z_dim=z_dim)
        vae = ConvVAE(z_dim=z_dim)
        dae = DenoisingConvAE(z_dim=z_dim)
        x_ae, z_ae = ae(sample)
        x_vae, mu, logvar, z_vae = vae(sample)
        x_dae, z_dae = dae(sample)
        print(
            f"z_dim={z_dim:>3} | ConvAE params={count_parameters(ae)[0]:,} out={tuple(x_ae.shape)} | "
            f"ConvVAE params={count_parameters(vae)[0]:,} out={tuple(x_vae.shape)}"
        )
    print("forward-pass check passed [OK]")
