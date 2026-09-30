"""Training script for Workshop 2.

Design: cross-entropy (optional label smoothing), Adam or SGD + momentum, cosine
annealing schedule, weight decay, early stopping on validation macro-F1, optional
mixup augmentation. Best checkpoint (highest validation macro-F1) is written to
``checkpoints/model_best.pt``; every epoch is logged to ``runs/train_log.csv`` and
(incrementally) to TensorBoard under ``runs/tensorboard/``.

Usage:
    python -m src.train --model smallcnn --epochs 30
    python -m src.train --model resnet18 --epochs 20 --freeze-epochs 5 --lr 3e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import CLASS_NAMES, FMNIST_MEAN, FMNIST_STD, IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from src.model import build_model, count_parameters, set_backbone_trainable


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def evaluate(
    model: nn.Module, loader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            preds_all.append(logits.argmax(dim=1).cpu())
            labels_all.append(labels.cpu())
    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return {
        "loss": total_loss / n,
        "accuracy": float((preds == labels).mean()),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_loss, n, correct = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
            logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return {"loss": total_loss / n, "accuracy": correct / n}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on Fashion-MNIST (Workshop 2)")
    parser.add_argument("--model", choices=["smallcnn", "resnet18"], default="smallcnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (Adam default 1e-3; try 3e-4 for resnet18)")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 disables)")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="epochs with frozen backbone (resnet18 two-stage fine-tuning)")
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience on validation macro-F1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    print(f"[train] device={device} | model={args.model} | seed={args.seed}")

    checkpoint_dir = Path(args.checkpoint_dir)
    runs_dir = Path(args.runs_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        args.model,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        seed=args.seed,
    )

    model = build_model(args.model).to(device)
    trainable, total = count_parameters(model)
    print(f"[train] parameters: trainable={trainable:,} total={total:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == "cosine"
        else None
    )

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(runs_dir / "tensorboard"))
    except Exception:
        writer = None

    log_path = runs_dir / "train_log.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
         "val_precision_macro", "val_recall_macro", "val_f1_macro", "epoch_time_sec"]
    )

    best_f1, best_epoch, epochs_without_improvement = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        if args.model == "resnet18":
            frozen = epoch <= args.freeze_epochs
            set_backbone_trainable(model, not frozen)
        start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha=args.mixup
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]
        log_writer.writerow(
            [
                epoch, f"{current_lr:.6g}", f"{train_metrics['loss']:.4f}", f"{train_metrics['accuracy']:.4f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['accuracy']:.4f}",
                f"{val_metrics['precision_macro']:.4f}", f"{val_metrics['recall_macro']:.4f}",
                f"{val_metrics['f1_macro']:.4f}", f"{elapsed:.1f}",
            ]
        )
        log_file.flush()

        if writer is not None:
            writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("f1_macro/val", val_metrics["f1_macro"], epoch)
            writer.add_scalar("lr", current_lr, epoch)

        frozen_tag = ""
        if args.model == "resnet18":
            frozen_tag = " [backbone frozen]" if epoch <= args.freeze_epochs else " [fine-tuning]"
        print(
            f"[train] epoch {epoch:03d}/{args.epochs} | lr {current_lr:.2e} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"f1 {val_metrics['f1_macro']:.4f}{frozen_tag}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = val_metrics["f1_macro"], epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "classes": CLASS_NAMES,
                    "normalization": {
                        "mean": list(IMAGENET_MEAN) if args.model == "resnet18" else [FMNIST_MEAN],
                        "std": list(IMAGENET_STD) if args.model == "resnet18" else [FMNIST_STD],
                    },
                    "input_size": 224 if args.model == "resnet18" else 28,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                checkpoint_dir / "model_best.pt",
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "classes": CLASS_NAMES,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            },
            checkpoint_dir / "model_last.pt",
        )

        if scheduler is not None:
            scheduler.step()

        if epochs_without_improvement >= args.patience:
            print(f"[train] early stopping (no improvement for {args.patience} epochs)")
            break

    log_file.close()
    if writer is not None:
        writer.close()

    summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "epochs_ran": epoch,
        "args": vars(args),
    }
    (runs_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. best val macro-F1 {best_f1:.4f} at epoch {best_epoch} — saved {checkpoint_dir / 'model_best.pt'}")


if __name__ == "__main__":
    main()
