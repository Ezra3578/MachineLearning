"""Synthetic sample generation and grid visualization script for Workshop 5.

Deliverable: ``src/generate_samples.py``
Generates 1,000-10,000 synthetic examples from both Vanilla GAN and WGAN-GP,
saves them to disk, and renders 4x4 and 8x8 side-by-side comparison grids
(Real vs Vanilla GAN vs WGAN-GP).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import get_full_dataset, get_splits
from src.gan import Generator, load_generator_checkpoint, pick_device, set_seed


def sample_from_generator(
    gen: Generator,
    num_samples: int,
    device: torch.device,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``num_samples`` images in [0, 1] of shape (N, 28, 28)."""
    set_seed(seed)
    gen.eval().to(device)
    imgs: list[np.ndarray] = []
    lbls: list[np.ndarray] = []
    rem = num_samples
    idx = 0
    with torch.no_grad():
        while rem > 0:
            bsz = min(batch_size, rem)
            z = torch.randn(bsz, gen.z_dim, device=device)
            if gen.num_classes > 0:
                y = (torch.arange(idx, idx + bsz, device=device) % gen.num_classes).long()
                out = gen(z, y)
                lbls.append(y.cpu().numpy())
            else:
                out = gen(z, None)
                lbls.append(np.full(bsz, -1, dtype=np.int64))
            arr = np.clip((out.cpu().numpy()[:, 0] + 1.0) * 0.5, 0.0, 1.0)
            imgs.append(arr)
            rem -= bsz
            idx += bsz
    return np.concatenate(imgs, axis=0), np.concatenate(lbls, axis=0)


def plot_side_by_side_grid(
    real_imgs: np.ndarray,
    vanilla_imgs: np.ndarray,
    wgan_imgs: np.ndarray,
    grid_size: int,
    out_path: Path,
) -> None:
    """Plot side-by-side grid_size x grid_size panels for Real, Vanilla GAN, and WGAN-GP."""
    n = grid_size * grid_size
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    panels = [
        ("Real Fashion-MNIST (Test Split)", real_imgs[:n]),
        ("Vanilla DCGAN (BCE + Label Smoothing 0.9)", vanilla_imgs[:n]),
        ("WGAN-GP (Wasserstein + Gradient Penalty lambda=10)", wgan_imgs[:n]),
    ]

    for ax, (title, arr) in zip(axes, panels):
        canvas = np.ones((grid_size * 28 + (grid_size - 1) * 2, grid_size * 28 + (grid_size - 1) * 2), dtype=np.float32)
        for i in range(n):
            r, c = divmod(i, grid_size)
            r0 = r * 30
            c0 = c * 30
            canvas[r0 : r0 + 28, c0 : c0 + 28] = arr[i]
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Workshop 5: {grid_size}x{grid_size} Side-by-Side Comparison — Real vs Vanilla GAN vs WGAN-GP",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[generate_samples] saved {grid_size}x{grid_size} grid -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic samples and 4x4 / 8x8 grids")
    parser.add_argument("--vanilla-ckpt", type=str, default="checkpoints/gan_vanilla_best.pt")
    parser.add_argument("--wgan-ckpt", type=str, default="checkpoints/gan_wgan_best.pt")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--runs-dir", type=str, default="runs/workshop5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    images, labels = get_full_dataset()
    splits = get_splits(labels, seed=args.seed)
    real_test = images[splits["test"]].astype(np.float32) / 255.0

    gen_van, _ = load_generator_checkpoint(args.vanilla_ckpt, map_location="cpu")
    gen_wgan, _ = load_generator_checkpoint(args.wgan_ckpt, map_location="cpu")

    van_imgs, van_lbls = sample_from_generator(gen_van, args.num_samples, device, seed=args.seed)
    wgan_imgs, wgan_lbls = sample_from_generator(gen_wgan, args.num_samples, device, seed=args.seed + 1)

    np.savez_compressed(runs_dir / "generated_vanilla.npz", images=van_imgs, labels=van_lbls)
    np.savez_compressed(runs_dir / "generated_wgan_gp.npz", images=wgan_imgs, labels=wgan_lbls)

    plot_side_by_side_grid(real_test, van_imgs, wgan_imgs, grid_size=4, out_path=runs_dir / "sample_grid_4x4.png")
    plot_side_by_side_grid(real_test, van_imgs, wgan_imgs, grid_size=8, out_path=runs_dir / "sample_grid_8x8.png")
    print(f"[generate_samples] generated {args.num_samples} samples per model [OK]")


if __name__ == "__main__":
    main()
