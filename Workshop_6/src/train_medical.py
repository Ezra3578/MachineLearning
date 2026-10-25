"""Training script for Workshop 6 medical adaptation (PneumoniaMNIST).

Trains the domain-transferred SmallCNN (see src/transfer.py) under different
configurations for the ablation study:

    python -m src.train_medical --name scratch    --mode scratch
    python -m src.train_medical --name frozen     --mode frozen
    python -m src.train_medical --name finetune   --mode finetune
    python -m src.train_medical --name finetune_clahe   --mode finetune --preprocess clahe
    python -m src.train_medical --name finetune_weights --mode finetune --class-weights balanced
    python -m src.train_medical --name full_recipe --mode finetune --preprocess clahe \
        --class-weights balanced --augment medical

Binary task with one logit + BCEWithLogitsLoss. Early stopping on validation
AUC-ROC (the metric used by MedMNIST benchmarks, for comparability).
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
from sklearn.metrics import roc_auc_score

from src import medical_preprocessing as mp
from src.transfer import build_transferred_model


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_proba(model, x: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    """Sigmoid probabilities for normalized input arrays (N,1,28,28)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i : i + batch]).to(device)
            out.append(torch.sigmoid(model(t)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def build_arrays(preprocess: str, data_root: str):
    """Load all splits as normalized arrays; optionally CLAHE first."""
    mean, std = mp.get_stats(data_root, use_clahe=(preprocess == "clahe"))
    arrays, labels = {}, {}
    for split in ("train", "val", "test"):
        imgs, y = mp.load_split_arrays(split, data_root)
        if preprocess == "clahe":
            imgs = mp.apply_clahe(imgs)
        arrays[split] = mp.make_tensors(imgs, mean, std)
        labels[split] = y
    return arrays, labels, {"mean": mean, "std": std, "preprocess": preprocess}


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Translation (up to 2 px per axis) and brightness jitter (0.9-1.1x).

    Mimics patient positioning and exposure variability; no horizontal flip
    (chest anatomy is lateralized)."""
    shifts = torch.randint(-2, 3, (x.size(0), 2))
    for i in range(x.size(0)):
        x[i] = torch.roll(x[i], shifts=(int(shifts[i, 0]), int(shifts[i, 1])), dims=(1, 2))
    gain = torch.empty(x.size(0), 1, 1, 1).uniform_(0.9, 1.1)
    return x * gain.to(x.device)


def rotate_batch_torch(x: torch.Tensor, degrees: float = 10.0) -> torch.Tensor:
    """Random rotation up to +-`degrees` using torch's affine grid."""
    import torch.nn.functional as F

    angle = (torch.rand(x.size(0)) * 2 - 1) * (degrees * 3.14159265 / 180.0)
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(x.size(0), 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def train_one_epoch(model, x_tr, y_tr, criterion, optimizer, device, augment: bool, batch_size: int):
    model.train()
    perm = torch.randperm(len(x_tr))
    total_loss, n = 0.0, 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i : i + batch_size]
        xb = x_tr[idx].clone()
        yb = y_tr[idx]
        if augment:
            xb = rotate_batch_torch(xb)
            xb = augment_batch(xb)
        xb = xb.to(device)
        yb_t = yb.to(device).float()
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(idx)
        n += len(idx)
    return total_loss / n


def main() -> None:
    p = argparse.ArgumentParser(description="Train the medical-adapted model (Workshop 6)")
    p.add_argument("--name", required=True, help="run name (used for checkpoints/ and runs/ dirs)")
    p.add_argument("--mode", choices=["scratch", "frozen", "finetune"], default="finetune")
    p.add_argument("--preprocess", choices=["none", "clahe"], default="none")
    p.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    p.add_argument("--augment", choices=["none", "medical"], default="none")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7, help="early stop patience on val AUC")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    arrays, labels, stats = build_arrays(args.preprocess, args.data_root)
    x_tr = torch.from_numpy(arrays["train"])
    y_tr = torch.from_numpy(labels["train"]).long()
    counts = mp.class_counts(labels["train"])

    model, transfer_report = build_transferred_model(args.mode, seed=args.seed)
    model.to(device)

    pos_weight = None
    if args.class_weights == "balanced":
        # weight positives so both classes contribute equally to the loss
        n_pos = counts["pneumonia"]
        n_neg = counts["normal"]
        pos_weight = torch.tensor(n_neg / n_pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path("runs/workshop6") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints") / f"{args.name}_best.pt"

    log_path = run_dir / "train_log.csv"
    with log_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(
            ["epoch", "lr", "train_loss", "val_auc", "val_acc", "val_sens", "val_spec"]
        )

    best_auc, best_epoch, no_improve = -1.0, -1, 0
    t0 = time.time()
    print(f"[train:{args.name}] mode={args.mode} preprocess={args.preprocess} "
          f"weights={args.class_weights} augment={args.augment} "
          f"transfer_loaded={transfer_report['loaded']} "
          f"trainable={transfer_report['trainable_params']:,}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, x_tr, y_tr, criterion, optimizer, device,
                               augment=(args.augment == "medical"), batch_size=args.batch_size)
        scheduler.step()
        proba_val = predict_proba(model, arrays["val"], device)
        y_val = labels["val"]
        auc = roc_auc_score(y_val, proba_val)
        pred = (proba_val >= 0.5).astype(int)
        acc = float((pred == y_val).mean())
        tp = int(((pred == 1) & (y_val == 1)).sum()); fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum()); fp = int(((pred == 1) & (y_val == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        with log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch, f"{scheduler.get_last_lr()[0]:.2e}", f"{loss:.4f}",
                 f"{auc:.4f}", f"{acc:.4f}", f"{sens:.4f}", f"{spec:.4f}"]
            )
        marker = ""
        if auc > best_auc:
            best_auc, best_epoch, no_improve = auc, epoch, 0
            torch.save(
                {
                    "model_name": "smallcnn-transfer",
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "transfer": transfer_report,
                    "normalization": stats,
                    "val_auc": float(auc),
                    "best_epoch": epoch,
                },
                ckpt_path,
            )
            marker = " *"
        else:
            no_improve += 1
        print(f"[train:{args.name}] epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | "
              f"val AUC {auc:.4f} acc {acc:.4f} sens {sens:.3f} spec {spec:.3f}{marker}")
        if no_improve >= args.patience:
            print(f"[train:{args.name}] early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    summary = {
        "name": args.name,
        "config": vars(args),
        "transfer": transfer_report,
        "normalization": stats,
        "class_counts": counts,
        "best_val_auc": round(float(best_auc), 4),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_seconds": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train:{args.name}] done. best val AUC {best_auc:.4f} at epoch {best_epoch} "
          f"({summary['train_seconds']}s) -> {ckpt_path}")


if __name__ == "__main__":
    main()
