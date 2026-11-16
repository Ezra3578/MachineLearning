"""Training script for Workshop 4 (Standard AE, beta-VAE, Denoising AE, Anomaly AE/VAE).

Deliverable: ``src/train_ae.py``
Saves best checkpoints to ``checkpoints/ae_best.pt`` and ``checkpoints/vae_best.pt``
and logs per-epoch metrics and curves to ``runs/workshop4/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.autoencoder import (
    ConvVAE,
    DenoisingConvAE,
    build_autoencoder,
    count_parameters,
    pick_device,
    set_seed,
    vae_elbo_loss,
)
from src.data import DEFAULT_NORMAL_CLASSES, get_full_dataset, get_splits

warnings.filterwarnings("ignore", message=".*lerp.*")


def _load_split_tensors(seed: int = 42, normal_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load contiguous float32 [0, 1] tensors (N, 1, 28, 28) for train and val splits."""
    images, labels = get_full_dataset()
    splits = get_splits(labels, seed=seed)
    tr_idx = np.asarray(splits["train"], dtype=np.int64)
    va_idx = np.asarray(splits["val"], dtype=np.int64)

    if normal_only:
        norm_set = set(int(c) for c in DEFAULT_NORMAL_CLASSES)
        tr_idx = np.array([i for i in tr_idx if int(labels[i]) in norm_set], dtype=np.int64)
        va_idx = np.array([i for i in va_idx if int(labels[i]) in norm_set], dtype=np.int64)

    x_train = torch.from_numpy(images[tr_idx]).unsqueeze(1).float().div_(255.0).contiguous()
    x_val = torch.from_numpy(images[va_idx]).unsqueeze(1).float().div_(255.0).contiguous()
    return x_train, x_val


def train_one_run(
    model_type: str = "ae",
    z_dim: int = 64,
    beta: float = 1.0,
    kl_warmup_epochs: int = 10,
    noise_std: float = 0.25,
    anomaly: bool = False,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device_name: str = "auto",
    run_name: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    runs_dir: str | Path = "runs/workshop4",
    cached_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Train a single AE, VAE, or DAE configuration and save checkpoints + logs."""
    set_seed(seed)
    device = pick_device(device_name)

    if run_name is None:
        suffix = "_anomaly" if anomaly else ""
        if model_type == "vae":
            run_name = f"vae_z{z_dim}_beta{beta}{suffix}"
        elif model_type == "dae":
            run_name = f"dae_z{z_dim}_noise{noise_std}{suffix}"
        else:
            run_name = f"ae_z{z_dim}{suffix}"

    ckpt_root = Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if cached_tensors is not None:
        x_train_cpu, x_val_cpu = cached_tensors
    else:
        x_train_cpu, x_val_cpu = _load_split_tensors(seed=seed, normal_only=anomaly)

    n_train = x_train_cpu.size(0)
    n_val = x_val_cpu.size(0)

    model = build_autoencoder(model_type=model_type, z_dim=z_dim, noise_std=noise_std).to(device)
    trainable_params, total_params = count_parameters(model)

    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
    except TypeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    print(
        f"[train_ae] run={run_name} | model={model_type} z_dim={z_dim} beta={beta} "
        f"anomaly={anomaly} | params={trainable_params:,} | device={device}"
    )

    csv_path = run_dir / "train_log.csv"
    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_payload: dict | None = None
    ckpt_run_best = ckpt_root / f"{run_name}_best.pt"
    t_start = time.time()
    rng = torch.Generator().manual_seed(seed)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "epoch",
            "beta_eff",
            "train_loss",
            "train_mse_pixel",
            "train_kl",
            "val_loss",
            "val_mse_pixel",
            "val_kl",
            "lr",
            "epoch_sec",
        ]
        csv_writer = csv.DictWriter(fh, fieldnames=fieldnames)
        csv_writer.writeheader()

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            model.train()

            if model_type == "vae" and kl_warmup_epochs > 0:
                warmup_factor = min(1.0, epoch / float(kl_warmup_epochs))
                beta_eff = beta * warmup_factor
            else:
                beta_eff = beta if model_type == "vae" else 0.0

            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_kl_sum = 0.0

            perm = torch.randperm(n_train, generator=rng)
            for start in range(0, n_train, batch_size):
                batch_idx = perm[start : start + batch_size]
                x = x_train_cpu[batch_idx].to(device)
                bsz = x.size(0)
                optimizer.zero_grad()

                if isinstance(model, ConvVAE):
                    x_hat, mu, logvar, _ = model(x)
                    loss, recon_sum, kl_sum = vae_elbo_loss(x, x_hat, mu, logvar, beta=beta_eff)
                    mse_pixel = recon_sum / 784.0
                    kl_val = kl_sum
                elif isinstance(model, DenoisingConvAE):
                    x_hat, _ = model(x, add_noise=True)
                    recon_sum = F.mse_loss(x_hat, x, reduction="none").view(bsz, -1).sum(dim=1).mean()
                    loss = recon_sum
                    mse_pixel = recon_sum / 784.0
                    kl_val = torch.zeros((), device=device)
                else:
                    x_hat, _ = model(x)
                    recon_sum = F.mse_loss(x_hat, x, reduction="none").view(bsz, -1).sum(dim=1).mean()
                    loss = recon_sum
                    mse_pixel = recon_sum / 784.0
                    kl_val = torch.zeros((), device=device)

                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item()) * bsz
                train_mse_sum += float(mse_pixel.item()) * bsz
                train_kl_sum += float(kl_val.item()) * bsz

            scheduler.step()

            model.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_kl_sum = 0.0

            with torch.no_grad():
                for start in range(0, n_val, batch_size):
                    x = x_val_cpu[start : start + batch_size].to(device)
                    bsz = x.size(0)

                    if isinstance(model, ConvVAE):
                        x_hat, mu, logvar, _ = model(x)
                        loss, recon_sum, kl_sum = vae_elbo_loss(x, x_hat, mu, logvar, beta=beta)
                        mse_pixel = recon_sum / 784.0
                        kl_val = kl_sum
                    elif isinstance(model, DenoisingConvAE):
                        x_hat, _ = model(x, add_noise=True)
                        recon_sum = F.mse_loss(x_hat, x, reduction="none").view(bsz, -1).sum(dim=1).mean()
                        loss = recon_sum
                        mse_pixel = recon_sum / 784.0
                        kl_val = torch.zeros((), device=device)
                    else:
                        x_hat, _ = model(x)
                        recon_sum = F.mse_loss(x_hat, x, reduction="none").view(bsz, -1).sum(dim=1).mean()
                        loss = recon_sum
                        mse_pixel = recon_sum / 784.0
                        kl_val = torch.zeros((), device=device)

                    val_loss_sum += float(loss.item()) * bsz
                    val_mse_sum += float(mse_pixel.item()) * bsz
                    val_kl_sum += float(kl_val.item()) * bsz

            row = {
                "epoch": epoch,
                "beta_eff": round(float(beta_eff), 4),
                "train_loss": train_loss_sum / max(1, n_train),
                "train_mse_pixel": train_mse_sum / max(1, n_train),
                "train_kl": train_kl_sum / max(1, n_train),
                "val_loss": val_loss_sum / max(1, n_val),
                "val_mse_pixel": val_mse_sum / max(1, n_val),
                "val_kl": val_kl_sum / max(1, n_val),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_sec": round(time.time() - t_ep, 2),
            }
            history.append(row)
            csv_writer.writerow(row)
            fh.flush()

            writer.add_scalar("loss/train", row["train_loss"], epoch)
            writer.add_scalar("loss/val", row["val_loss"], epoch)
            writer.add_scalar("mse_pixel/train", row["train_mse_pixel"], epoch)
            writer.add_scalar("mse_pixel/val", row["val_mse_pixel"], epoch)
            if model_type == "vae":
                writer.add_scalar("kl/train", row["train_kl"], epoch)
                writer.add_scalar("kl/val", row["val_kl"], epoch)

            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                best_epoch = epoch
                best_payload = {
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "model_type": model_type,
                    "z_dim": z_dim,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_mse_pixel": row["val_mse_pixel"],
                    "best_val_kl": row["val_kl"],
                    "config": {
                        "model": model_type,
                        "z_dim": z_dim,
                        "beta": beta,
                        "kl_warmup_epochs": kl_warmup_epochs,
                        "noise_std": noise_std,
                        "anomaly": anomaly,
                        "normal_classes": list(DEFAULT_NORMAL_CLASSES) if anomaly else list(range(10)),
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "seed": seed,
                        "run_name": run_name,
                    },
                }

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"  ep {epoch:02d}/{epochs:02d} | train_mse={row['train_mse_pixel']:.5f} "
                    f"val_mse={row['val_mse_pixel']:.5f} val_kl={row['val_kl']:.2f} "
                    f"({row['epoch_sec']:.1f}s)"
                )

    writer.close()
    if best_payload is not None:
        for attempt in range(5):
            try:
                torch.save(best_payload, ckpt_run_best)
                break
            except RuntimeError:
                time.sleep(0.5 * (attempt + 1))
    total_sec = round(time.time() - t_start, 2)
    summary = {
        "run_name": run_name,
        "model": model_type,
        "z_dim": z_dim,
        "beta": beta,
        "anomaly": anomaly,
        "epochs": epochs,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mse_pixel": history[best_epoch - 1]["val_mse_pixel"],
        "best_val_kl": history[best_epoch - 1]["val_kl"],
        "total_train_sec": total_sec,
        "checkpoint": str(ckpt_run_best),
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_training_curves(summaries: list[dict], out_path: Path) -> None:
    """Plot AE z_dim sweep and VAE beta sweep learning curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for s in summaries:
        if s["model"] == "ae" and not s["anomaly"]:
            eps = [r["epoch"] for r in s["history"]]
            ax.plot(eps, [r["train_mse_pixel"] for r in s["history"]], linestyle="--", alpha=0.55)
            ax.plot(
                eps,
                [r["val_mse_pixel"] for r in s["history"]],
                linewidth=2.0,
                label=f"AE z={s['z_dim']} (val={s['best_val_mse_pixel']:.4f})",
            )
    ax.set_title("Standard AE: Reconstruction MSE vs Epoch\n(dashed=train, solid=val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-Pixel MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for s in summaries:
        if s["model"] in ("vae", "dae") and not s["anomaly"]:
            eps = [r["epoch"] for r in s["history"]]
            lbl = (
                f"VAE beta={s['beta']} (val MSE={s['best_val_mse_pixel']:.4f})"
                if s["model"] == "vae"
                else f"Denoising AE z={s['z_dim']} (val MSE={s['best_val_mse_pixel']:.4f})"
            )
            ax.plot(eps, [r["val_mse_pixel"] for r in s["history"]], linewidth=2.0, label=lbl)
    ax.set_title("VAE (beta sweep) & DAE: Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-Pixel MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for s in summaries:
        if s["model"] == "vae" and not s["anomaly"]:
            eps = [r["epoch"] for r in s["history"]]
            ax.plot(
                eps,
                [r["val_kl"] for r in s["history"]],
                linewidth=2.0,
                label=f"VAE beta={s['beta']} (KL={s['best_val_kl']:.1f})",
            )
    ax.set_title("VAE KL Divergence D_KL(q(z|x) || N(0,I))\n(with 10-epoch KL warmup)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[train_ae] saved training curves -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Autoencoder / beta-VAE on Fashion-MNIST")
    parser.add_argument("--model", choices=["ae", "vae", "dae"], default="ae")
    parser.add_argument("--z-dim", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--run-all", action="store_true", help="Run full Workshop 4 experiment suite")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--runs-dir", type=str, default="runs/workshop4")
    return parser.parse_args()


def _safe_copy2(src: Path, dst: Path) -> None:
    for attempt in range(5):
        try:
            shutil.copy2(src, dst)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))


def main() -> None:
    args = parse_args()
    ckpt_root = Path(args.checkpoint_dir)
    runs_root = Path(args.runs_dir)

    if args.run_all:
        full_tensors = _load_split_tensors(seed=args.seed, normal_only=False)
        anom_tensors = _load_split_tensors(seed=args.seed, normal_only=True)
        summaries: list[dict] = []

        # 1. Standard AE across z_dim in {32, 64, 128}
        for z_dim in (32, 64, 128):
            z32_summary_path = runs_root / "ae_z32" / "summary.json"
            if z_dim == 32 and z32_summary_path.exists() and (ckpt_root / "ae_z32_best.pt").exists():
                existing = json.loads(z32_summary_path.read_text(encoding="utf-8"))
                if existing.get("trainable_params") == 263985:
                    print(f"[train_ae] reusing completed ae_z32 (best_val_mse={existing['best_val_mse_pixel']:.5f})")
                    summaries.append(existing)
                    continue
            s = train_one_run(
                model_type="ae",
                z_dim=z_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device_name=args.device,
                checkpoint_dir=ckpt_root,
                runs_dir=runs_root,
                cached_tensors=full_tensors,
            )
            summaries.append(s)

        _safe_copy2(ckpt_root / "ae_z64_best.pt", ckpt_root / "ae_best.pt")

        # 2. Variational Autoencoder across beta in {0.5, 1.0, 4.0} at z_dim=64
        for beta in (0.5, 1.0, 4.0):
            s = train_one_run(
                model_type="vae",
                z_dim=64,
                beta=beta,
                kl_warmup_epochs=args.kl_warmup_epochs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device_name=args.device,
                checkpoint_dir=ckpt_root,
                runs_dir=runs_root,
                cached_tensors=full_tensors,
            )
            summaries.append(s)

        _safe_copy2(ckpt_root / "vae_z64_beta1.0_best.pt", ckpt_root / "vae_best.pt")

        # 3. Denoising Autoencoder (optional extension)
        s_dae = train_one_run(
            model_type="dae",
            z_dim=64,
            noise_std=args.noise_std,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=ckpt_root,
            runs_dir=runs_root,
            cached_tensors=full_tensors,
        )
        summaries.append(s_dae)
        _safe_copy2(Path(s_dae["checkpoint"]), ckpt_root / "dae_best.pt")

        # 4. Anomaly Detection models (trained on Normal classes only, excluding Bag [8])
        s_ae_anom = train_one_run(
            model_type="ae",
            z_dim=64,
            anomaly=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=ckpt_root,
            runs_dir=runs_root,
            cached_tensors=anom_tensors,
        )
        summaries.append(s_ae_anom)
        _safe_copy2(Path(s_ae_anom["checkpoint"]), ckpt_root / "ae_anomaly_best.pt")

        s_vae_anom = train_one_run(
            model_type="vae",
            z_dim=64,
            beta=1.0,
            kl_warmup_epochs=args.kl_warmup_epochs,
            anomaly=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=ckpt_root,
            runs_dir=runs_root,
            cached_tensors=anom_tensors,
        )
        summaries.append(s_vae_anom)
        _safe_copy2(Path(s_vae_anom["checkpoint"]), ckpt_root / "vae_anomaly_best.pt")

        plot_training_curves(summaries, runs_root / "training_curves.png")
        (runs_root / "all_train_summaries.json").write_text(
            json.dumps([{k: v for k, v in s.items() if k != "history"} for s in summaries], indent=2),
            encoding="utf-8",
        )
        print("[train_ae] completed all Workshop 4 training runs [OK]")
    else:
        s = train_one_run(
            model_type=args.model,
            z_dim=args.z_dim,
            beta=args.beta,
            kl_warmup_epochs=args.kl_warmup_epochs,
            noise_std=args.noise_std,
            anomaly=args.anomaly,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device_name=args.device,
            checkpoint_dir=ckpt_root,
            runs_dir=runs_root,
        )
        target_name = "vae_best.pt" if args.model == "vae" else ("dae_best.pt" if args.model == "dae" else "ae_best.pt")
        if not args.anomaly:
            shutil.copy2(Path(s["checkpoint"]), ckpt_root / target_name)


if __name__ == "__main__":
    main()
