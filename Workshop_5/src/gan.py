"""Generator, Discriminator, and WGAN-GP Critic architectures for Workshop 5.

Implements:
- ``Generator``                : DCGAN-style transpose-conv generator (Tanh output in [-1, 1]).
- ``Discriminator``            : Vanilla GAN discriminator with Sigmoid output.
- ``Critic``                   : WGAN-GP critic without Sigmoid and using InstanceNorm2d.
- ``compute_gradient_penalty`` : WGAN-GP gradient penalty (Gulrajani et al. 2017, lambda=10).

Run ``python -m src.gan`` to verify parameter counts, output shapes, and gradient penalty.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch import autograd


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


def _weights_init_dcgan(m: nn.Module) -> None:
    """DCGAN weight initialization (Radford et al., 2015)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """DCGAN Generator mapping z in R^{z_dim} (and optional class label y) to [-1, 1]^{1x28x28}."""

    def __init__(self, z_dim: int = 100, num_classes: int = 0, embed_dim: int = 16) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        in_dim = z_dim + (embed_dim if num_classes > 0 else 0)
        self.label_emb = nn.Embedding(num_classes, embed_dim) if num_classes > 0 else None

        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128 * 7 * 7, bias=False),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=False),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),   # 14 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.apply(_weights_init_dcgan)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_emb is not None and labels is not None:
            z = torch.cat([z, self.label_emb(labels)], dim=1)
        h = self.proj(z).view(-1, 128, 7, 7)
        return self.net(h)


class Discriminator(nn.Module):
    """Vanilla GAN Discriminator outputting probabilities in (0, 1)."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__()
        self.num_classes = num_classes
        in_ch = 1 + (1 if num_classes > 0 else 0)
        self.label_emb = nn.Embedding(num_classes, 28 * 28) if num_classes > 0 else None

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=4, stride=2, padding=1),                    # 28 -> 14
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),           # 14 -> 7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),          # 7 -> 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid(),
        )
        self.apply(_weights_init_dcgan)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_emb is not None and labels is not None:
            y_map = self.label_emb(labels).view(-1, 1, 28, 28)
            x = torch.cat([x, y_map], dim=1)
        return self.features(x).squeeze(1)


class Critic(nn.Module):
    """WGAN-GP Critic outputting unconstrained real scores (no Sigmoid, InstanceNorm2d)."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__()
        self.num_classes = num_classes
        in_ch = 1 + (1 if num_classes > 0 else 0)
        self.label_emb = nn.Embedding(num_classes, 28 * 28) if num_classes > 0 else None

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=4, stride=2, padding=1),                    # 28 -> 14
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),           # 14 -> 7
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),          # 7 -> 4
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )
        self.apply(_weights_init_dcgan)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_emb is not None and labels is not None:
            y_map = self.label_emb(labels).view(-1, 1, 28, 28)
            x = torch.cat([x, y_map], dim=1)
        return self.features(x).squeeze(1)


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    labels: torch.Tensor | None = None,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute WGAN-GP penalty: lambda * E[(||grad_{x_hat} D(x_hat)||_2 - 1)^2]."""
    batch_size = real_samples.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
    x_hat = (eps * real_samples + (1.0 - eps) * fake_samples.detach()).requires_grad_(True)
    scores = critic(x_hat, labels)
    grads = autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp


def load_generator_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[Generator, dict]:
    """Load a saved Generator checkpoint with weights_only=False compatibility."""
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt.get("config", {})
    z_dim = int(cfg.get("z_dim", ckpt.get("z_dim", 100)))
    num_classes = int(cfg.get("num_classes", ckpt.get("num_classes", 0)))
    gen = Generator(z_dim=z_dim, num_classes=num_classes)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    return gen, ckpt


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    z = torch.randn(4, 100)
    G = Generator(z_dim=100)
    D = Discriminator()
    C = Critic()
    fake = G(z)
    prob = D(fake)
    score = C(fake)
    gp = compute_gradient_penalty(C, torch.randn_like(fake), fake)
    print(f"Generator     : params={count_parameters(G)[0]:,} | out={tuple(fake.shape)}")
    print(f"Discriminator : params={count_parameters(D)[0]:,} | out={tuple(prob.shape)}")
    print(f"Critic        : params={count_parameters(C)[0]:,} | out={tuple(score.shape)} | GP={gp.item():.4f}")
    print("forward-pass check passed [OK]")
