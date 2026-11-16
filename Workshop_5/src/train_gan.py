"""Training loop for Workshop 5 (Vanilla GAN & WGAN-GP).

Deliverable: ``src/train_gan.py``
Saves best checkpoints to ``checkpoints/gan_vanilla_best.pt`` and ``checkpoints/gan_wgan_best.pt``,
logs per-epoch metrics to ``runs/workshop5/<model>/train_log.csv``, and saves generated sample
grids every 10 epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data import get_full_dataset, get_splits
from src.gan import (
    Critic,
    Discriminator,
    Generator,
    compute_gradient_penalty,
    count_parameters,
    pick_device,
    set_seed,
)

warnings.filterwarnings("ignore", message=".*lerp.*")


def _dml_safe_bce(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """DirectML-native Binary Cross-Entropy avoiding CPU fallback."""
    p = torch.clamp(prob, eps, 1.0 - eps)
    return -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p)).mean()


def _safe_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            torch.save(payload, path)
            return
        except RuntimeError:
            time.sleep(0.5 * (attempt + 1))


def _load_gan_train_tensors(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Load contiguous float32 [-1, 1] train images (49000, 1, 28, 28) and int64 labels."""
    images, labels = get_full_dataset()
    splits = get_splits(labels, seed=seed)
    tr_idx = np.asarray(splits["train"], dtype=np.int64)
    x_train = torch.from_numpy(images[tr_idx]).unsqueeze(1).float().div_(127.5).sub_(1.0).contiguous()
    y_train = torch.from_numpy(labels[tr_idx]).long().contiguous()
    return x_train, y_train


def save_sample_grid(
    generator: Generator,
    fixed_z: torch.Tensor,
    fixed_labels: torch.Tensor | None,
    out_path: Path,
    title: str,
    nrow: int = 8,
) -> None:
    """Render an 8x8 grid of generated samples mapped from [-1, 1] to [0, 1]."""
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_z, fixed_labels).cpu().numpy()[:, 0]
    samples = np.clip((samples + 1.0) * 0.5, 0.0, 1.0)

    n = len(samples)
    ncol = nrow
    nrow_grid = max(1, n // ncol)
    canvas = np.ones((nrow_grid * 30 - 2, ncol * 30 - 2), dtype=np.float32)
    for i in range(min(n, nrow_grid * ncol)):
        r, c = divmod(i, ncol)
        canvas[r * 30 : r * 30 + 28, c * 30 : c * 30 + 28] = samples[i]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_vanilla_gan(
    epochs: int = 100,
    batch_size: int = 512,
    z_dim: int = 100,
    lr: float = 2e-4,
    label_smoothing: float = 0.9,
    conditional: bool = False,
    sample_interval: int = 10,
    max_batches_per_epoch: int | None = None,
    seed: int = 42,
    device_name: str = "auto",
    checkpoint_dir: str | Path = "checkpoints",
    runs_dir: str | Path = "runs/workshop5",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Train Vanilla DCGAN with BCE loss and one-sided label smoothing."""
    set_seed(seed)
    device = pick_device(device_name)
    num_classes = 10 if conditional else 0
    run_name = "vanilla_cgan" if conditional else "vanilla_gan"

    ckpt_root = Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(runs_dir) / run_name
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    x_train_cpu, y_train_cpu = cached_data if cached_data is not None else _load_gan_train_tensors(seed=seed)
    n_train = x_train_cpu.size(0)

    G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    D = Discriminator(num_classes=num_classes).to(device)
    criterion = nn.BCELoss()

    try:
        opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999), foreach=False)
        opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999), foreach=False)
    except TypeError:
        opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_z = torch.randn(64, z_dim, device=device)
    fixed_y = torch.arange(64, device=device) % 10 if conditional else None

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    csv_path = run_dir / "train_log.csv"
    history: list[dict] = []
    t_start = time.time()
    rng = torch.Generator().manual_seed(seed)

    print(
        f"[train_gan] model={run_name} | epochs={epochs} batch={batch_size} lr={lr} "
        f"smooth={label_smoothing} | G_params={count_parameters(G)[0]:,} D_params={count_parameters(D)[0]:,} | device={device}"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["epoch", "d_loss", "g_loss", "d_acc_real", "d_acc_fake", "d_prob_real", "d_prob_fake", "epoch_sec"]
        csv_writer = csv.DictWriter(fh, fieldnames=fieldnames)
        csv_writer.writeheader()

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            G.train()
            D.train()

            d_loss_sum = 0.0
            g_loss_sum = 0.0
            d_acc_real_sum = 0.0
            d_acc_fake_sum = 0.0
            d_prob_real_sum = 0.0
            d_prob_fake_sum = 0.0
            n_seen = 0

            perm = torch.randperm(n_train, generator=rng)
            n_batches = n_train // batch_size
            if max_batches_per_epoch is not None:
                n_batches = min(n_batches, max_batches_per_epoch)

            for b_idx in range(n_batches):
                idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
                real_x = x_train_cpu[idx].to(device)
                bsz = real_x.size(0)
                y_cond = y_train_cpu[idx].to(device) if conditional else None

                real_targets = torch.full((bsz,), float(label_smoothing), dtype=torch.float32, device=device)
                fake_targets = torch.zeros(bsz, dtype=torch.float32, device=device)
                gen_targets = torch.ones(bsz, dtype=torch.float32, device=device)

                # 1. Train Discriminator
                opt_d.zero_grad()
                out_real = D(real_x, y_cond)
                loss_d_real = _dml_safe_bce(out_real, real_targets)

                z = torch.randn(bsz, z_dim, device=device)
                fake_x = G(z, y_cond)
                out_fake = D(fake_x.detach(), y_cond)
                loss_d_fake = _dml_safe_bce(out_fake, fake_targets)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                loss_d.backward()
                opt_d.step()

                # 2. Train Generator
                opt_g.zero_grad()
                out_fake_for_g = D(fake_x, y_cond)
                loss_g = _dml_safe_bce(out_fake_for_g, gen_targets)
                loss_g.backward()
                opt_g.step()

                d_loss_sum += float(loss_d.item()) * bsz
                g_loss_sum += float(loss_g.item()) * bsz
                d_acc_real_sum += float((out_real >= 0.5).float().mean().item()) * bsz
                d_acc_fake_sum += float((out_fake < 0.5).float().mean().item()) * bsz
                d_prob_real_sum += float(out_real.mean().item()) * bsz
                d_prob_fake_sum += float(out_fake.mean().item()) * bsz
                n_seen += bsz

            row = {
                "epoch": epoch,
                "d_loss": d_loss_sum / max(1, n_seen),
                "g_loss": g_loss_sum / max(1, n_seen),
                "d_acc_real": d_acc_real_sum / max(1, n_seen),
                "d_acc_fake": d_acc_fake_sum / max(1, n_seen),
                "d_prob_real": d_prob_real_sum / max(1, n_seen),
                "d_prob_fake": d_prob_fake_sum / max(1, n_seen),
                "epoch_sec": round(time.time() - t_ep, 2),
            }
            history.append(row)
            csv_writer.writerow(row)
            fh.flush()

            writer.add_scalar("loss/discriminator", row["d_loss"], epoch)
            writer.add_scalar("loss/generator", row["g_loss"], epoch)
            writer.add_scalar("accuracy/real", row["d_acc_real"], epoch)
            writer.add_scalar("accuracy/fake", row["d_acc_fake"], epoch)

            if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
                save_sample_grid(
                    G,
                    fixed_z,
                    fixed_y,
                    samples_dir / f"samples_epoch_{epoch:03d}.png",
                    f"Vanilla GAN Generated Samples — Epoch {epoch}",
                )
                print(
                    f"  ep {epoch:03d}/{epochs:03d} | d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
                    f"| d_acc_real={row['d_acc_real']*100:.1f}% d_acc_fake={row['d_acc_fake']*100:.1f}% "
                    f"({row['epoch_sec']:.1f}s)"
                )

    writer.close()
    ckpt_path = ckpt_root / ("gan_vanilla_cond_best.pt" if conditional else "gan_vanilla_best.pt")
    payload = {
        "generator_state_dict": {k: v.detach().cpu().clone() for k, v in G.state_dict().items()},
        "discriminator_state_dict": {k: v.detach().cpu().clone() for k, v in D.state_dict().items()},
        "model_type": "vanilla",
        "z_dim": z_dim,
        "num_classes": num_classes,
        "epochs": epochs,
        "config": {
            "model": "vanilla",
            "z_dim": z_dim,
            "num_classes": num_classes,
            "conditional": conditional,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "label_smoothing": label_smoothing,
            "seed": seed,
        },
    }
    _safe_torch_save(payload, ckpt_path)
    summary = {
        "run_name": run_name,
        "model": "vanilla",
        "conditional": conditional,
        "epochs": epochs,
        "total_train_sec": round(time.time() - t_start, 2),
        "checkpoint": str(ckpt_path),
        "final_metrics": history[-1],
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_wgan_gp(
    epochs: int = 100,
    batch_size: int = 512,
    z_dim: int = 100,
    lr: float = 2e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    conditional: bool = False,
    sample_interval: int = 10,
    max_batches_per_epoch: int | None = None,
    seed: int = 42,
    device_name: str = "auto",
    checkpoint_dir: str | Path = "checkpoints",
    runs_dir: str | Path = "runs/workshop5",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Train Wasserstein GAN with Gradient Penalty (Gulrajani et al. 2017)."""
    set_seed(seed)
    device = pick_device(device_name)
    num_classes = 10 if conditional else 0
    run_name = "wgan_gp_cgan" if conditional else "wgan_gp"

    ckpt_root = Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(runs_dir) / run_name
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    x_train_cpu, y_train_cpu = cached_data if cached_data is not None else _load_gan_train_tensors(seed=seed)
    n_train = x_train_cpu.size(0)

    G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
    C = Critic(num_classes=num_classes).to(device)

    try:
        opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9), foreach=False)
        opt_c = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9), foreach=False)
    except TypeError:
        opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
        opt_c = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(64, z_dim, device=device)
    fixed_y = torch.arange(64, device=device) % 10 if conditional else None

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    csv_path = run_dir / "train_log.csv"
    history: list[dict] = []
    t_start = time.time()
    rng = torch.Generator().manual_seed(seed)

    print(
        f"[train_gan] model={run_name} | epochs={epochs} batch={batch_size} lr={lr} "
        f"n_critic={n_critic} lambda_gp={lambda_gp} | G_params={count_parameters(G)[0]:,} C_params={count_parameters(C)[0]:,} | device={device}"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["epoch", "critic_loss", "wasserstein_dist", "gp_loss", "g_loss", "epoch_sec"]
        csv_writer = csv.DictWriter(fh, fieldnames=fieldnames)
        csv_writer.writeheader()

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            G.train()
            C.train()

            c_loss_sum = 0.0
            w_dist_sum = 0.0
            gp_sum = 0.0
            g_loss_sum = 0.0
            n_c_steps = 0
            n_g_steps = 0

            perm = torch.randperm(n_train, generator=rng)
            n_batches = n_train // batch_size
            if max_batches_per_epoch is not None:
                n_batches = min(n_batches, max_batches_per_epoch)

            for b_idx in range(n_batches):
                idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
                real_x = x_train_cpu[idx].to(device)
                bsz = real_x.size(0)
                y_cond = y_train_cpu[idx].to(device) if conditional else None

                # 1. Train Critic
                opt_c.zero_grad()
                z = torch.randn(bsz, z_dim, device=device)
                fake_x = G(z, y_cond)

                c_real = C(real_x, y_cond).mean()
                c_fake = C(fake_x.detach(), y_cond).mean()
                gp = compute_gradient_penalty(C, real_x, fake_x.detach(), labels=y_cond, lambda_gp=lambda_gp)

                w_dist = c_real - c_fake
                loss_c = -w_dist + gp
                loss_c.backward()
                opt_c.step()

                c_loss_sum += float(loss_c.item())
                w_dist_sum += float(w_dist.item())
                gp_sum += float(gp.item())
                n_c_steps += 1

                # 2. Train Generator every n_critic steps
                if (b_idx + 1) % n_critic == 0:
                    opt_g.zero_grad()
                    z_g = torch.randn(bsz, z_dim, device=device)
                    gen_x = G(z_g, y_cond)
                    loss_g = -C(gen_x, y_cond).mean()
                    loss_g.backward()
                    opt_g.step()
                    g_loss_sum += float(loss_g.item())
                    n_g_steps += 1

            row = {
                "epoch": epoch,
                "critic_loss": c_loss_sum / max(1, n_c_steps),
                "wasserstein_dist": w_dist_sum / max(1, n_c_steps),
                "gp_loss": gp_sum / max(1, n_c_steps),
                "g_loss": g_loss_sum / max(1, n_g_steps),
                "epoch_sec": round(time.time() - t_ep, 2),
            }
            history.append(row)
            csv_writer.writerow(row)
            fh.flush()

            writer.add_scalar("loss/critic", row["critic_loss"], epoch)
            writer.add_scalar("loss/wasserstein_dist", row["wasserstein_dist"], epoch)
            writer.add_scalar("loss/gradient_penalty", row["gp_loss"], epoch)
            writer.add_scalar("loss/generator", row["g_loss"], epoch)

            if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
                save_sample_grid(
                    G,
                    fixed_z,
                    fixed_y,
                    samples_dir / f"samples_epoch_{epoch:03d}.png",
                    f"WGAN-GP Generated Samples — Epoch {epoch}",
                )
                print(
                    f"  ep {epoch:03d}/{epochs:03d} | W_dist={row['wasserstein_dist']:.4f} "
                    f"critic_loss={row['critic_loss']:.4f} gp={row['gp_loss']:.4f} g_loss={row['g_loss']:.4f} "
                    f"({row['epoch_sec']:.1f}s)"
                )

    writer.close()
    ckpt_path = ckpt_root / ("gan_wgan_cond_best.pt" if conditional else "gan_wgan_best.pt")
    payload = {
        "generator_state_dict": {k: v.detach().cpu().clone() for k, v in G.state_dict().items()},
        "critic_state_dict": {k: v.detach().cpu().clone() for k, v in C.state_dict().items()},
        "model_type": "wgan_gp",
        "z_dim": z_dim,
        "num_classes": num_classes,
        "epochs": epochs,
        "config": {
            "model": "wgan_gp",
            "z_dim": z_dim,
            "num_classes": num_classes,
            "conditional": conditional,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_critic": n_critic,
            "lambda_gp": lambda_gp,
            "seed": seed,
        },
    }
    _safe_torch_save(payload, ckpt_path)
    summary = {
        "run_name": run_name,
        "model": "wgan_gp",
        "conditional": conditional,
        "epochs": epochs,
        "total_train_sec": round(time.time() - t_start, 2),
        "checkpoint": str(ckpt_path),
        "final_metrics": history[-1],
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_gan_training_curves(vanilla_summary: dict, wgan_summary: dict, out_path: Path) -> None:
    """Plot 3-panel comparison of Vanilla GAN vs WGAN-GP learning dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    v_hist = vanilla_summary["history"]
    w_hist = wgan_summary["history"]

    ax = axes[0]
    eps_v = [r["epoch"] for r in v_hist]
    ax.plot(eps_v, [r["d_loss"] for r in v_hist], label="Discriminator BCE Loss", linewidth=2.0, color="#1f77b4")
    ax.plot(eps_v, [r["g_loss"] for r in v_hist], label="Generator BCE Loss", linewidth=2.0, color="#ff7f0e")
    ax.set_title("Vanilla GAN: Generator & Discriminator Loss\n(Label Smoothing = 0.9)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(eps_v, [r["d_acc_real"] * 100 for r in v_hist], label="D Accuracy on Real (%)", linewidth=2.0, color="#2ca02c")
    ax.plot(eps_v, [r["d_acc_fake"] * 100 for r in v_hist], label="D Accuracy on Fake (%)", linewidth=2.0, color="#d62728")
    ax.axhline(50.0, color="black", linestyle="--", alpha=0.5, label="50% Nash Equilibrium")
    ax.set_title("Vanilla GAN: Discriminator Real vs Fake Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax = axes[2]
    eps_w = [r["epoch"] for r in w_hist]
    ax.plot(
        eps_w,
        [r["wasserstein_dist"] for r in w_hist],
        label="Wasserstein Distance E[D(x)] - E[D(G(z))]",
        linewidth=2.2,
        color="#9467bd",
    )
    ax.plot(
        eps_w,
        [r["gp_loss"] for r in w_hist],
        label="Gradient Penalty (lambda=10)",
        linewidth=1.8,
        linestyle="--",
        color="#8c564b",
    )
    ax.set_title("WGAN-GP: Wasserstein Critic Distance & GP\n(n_critic = 5, Monotonic Convergence)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score / Penalty")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[train_gan] saved training stability comparison curves -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vanilla GAN or WGAN-GP on Fashion-MNIST")
    parser.add_argument("--model", choices=["vanilla", "wgan_gp"], default="vanilla")
    parser.add_argument("--run-all", action="store_true", help="Train both Vanilla GAN and WGAN-GP (100+ epochs)")
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.9)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--runs-dir", type=str, default="runs/workshop5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached = _load_gan_train_tensors(seed=args.seed)
    if args.run_all:
        s_van = train_vanilla_gan(
            epochs=args.epochs,
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            lr=args.lr,
            label_smoothing=args.label_smoothing,
            conditional=args.conditional,
            sample_interval=args.sample_interval,
            max_batches_per_epoch=args.max_batches_per_epoch,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=args.checkpoint_dir,
            runs_dir=args.runs_dir,
            cached_data=cached,
        )
        s_wgan = train_wgan_gp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            lr=args.lr,
            n_critic=args.n_critic,
            lambda_gp=args.lambda_gp,
            conditional=args.conditional,
            sample_interval=args.sample_interval,
            max_batches_per_epoch=args.max_batches_per_epoch,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=args.checkpoint_dir,
            runs_dir=args.runs_dir,
            cached_data=cached,
        )
        plot_gan_training_curves(s_van, s_wgan, Path(args.runs_dir) / "gan_training_curves.png")
        print("[train_gan] completed both Vanilla GAN and WGAN-GP runs [OK]")
    elif args.model == "vanilla":
        train_vanilla_gan(
            epochs=args.epochs,
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            lr=args.lr,
            label_smoothing=args.label_smoothing,
            conditional=args.conditional,
            sample_interval=args.sample_interval,
            max_batches_per_epoch=args.max_batches_per_epoch,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=args.checkpoint_dir,
            runs_dir=args.runs_dir,
            cached_data=cached,
        )
    else:
        train_wgan_gp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            lr=args.lr,
            n_critic=args.n_critic,
            lambda_gp=args.lambda_gp,
            conditional=args.conditional,
            sample_interval=args.sample_interval,
            max_batches_per_epoch=args.max_batches_per_epoch,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=args.checkpoint_dir,
            runs_dir=args.runs_dir,
            cached_data=cached,
        )


if __name__ == "__main__":
    main()
