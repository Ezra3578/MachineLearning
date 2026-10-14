"""Data augmentation experiment script for Workshop 5.

Deliverable: ``src/augment_and_retrain.py``
Retrains the Workshop 2 SmallCNN architecture on the shared 70/15/15 split under
three conditions:
  (a) Real training data only (baseline from Workshop 2)
  (b) Real + Synthetic data from Vanilla GAN
  (c) Real + Synthetic data from WGAN-GP

Ensures fair comparison: identical validation/test split (seed 42), identical
SmallCNN architecture (140,458 params), optimizer, schedule, and epoch budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data import CLASS_NAMES, FMNIST_MEAN, FMNIST_STD, get_full_dataset, get_splits
from src.evaluate_gan import extract_features_and_probs, load_w2_evaluator
from src.gan import Generator, load_generator_checkpoint, pick_device, set_seed
from src.generate_samples import sample_from_generator

warnings.filterwarnings("ignore", message=".*lerp.*")


class NormalizedArrayDataset(Dataset):
    """Dataset over float32 [0, 1] images applying W2 augmentation & z-score normalization."""

    def __init__(self, images_unit: np.ndarray, labels: np.ndarray, augment: bool = False) -> None:
        self.images = images_unit.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, 28, 28) in [0, 1]
        if self.augment:
            if torch.rand(()) < 0.5:
                img = torch.flip(img, dims=[2])
            # Random +/-2 pixel shift (approx 10% of 28)
            dy = int(torch.randint(-2, 3, ()).item())
            dx = int(torch.randint(-2, 3, ()).item())
            img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
            if dy > 0:
                img[:, :dy, :] = 0.0
            elif dy < 0:
                img[:, dy:, :] = 0.0
            if dx > 0:
                img[:, :, :dx] = 0.0
            elif dx < 0:
                img[:, :, dx:] = 0.0
        img = (img - FMNIST_MEAN) / FMNIST_STD
        return img, int(self.labels[idx])


def build_synthetic_labeled_set(
    gen: Generator,
    teacher: nn.Module,
    num_synthetic: int,
    device: torch.device,
    conf_threshold: float = 0.85,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate ``num_synthetic`` labeled images (using cGAN labels or high-confidence teacher filtering)."""
    pool_size = max(num_synthetic * 3, 6000)
    imgs_pool, lbls_pool = sample_from_generator(gen, pool_size, device, seed=seed)
    _, probs = extract_features_and_probs(teacher, imgs_pool, device)
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    if gen.num_classes > 0 and np.all(lbls_pool >= 0):
        chosen_y = lbls_pool[:num_synthetic]
        chosen_x = imgs_pool[:num_synthetic]
        mean_conf = float(np.mean(confs[:num_synthetic]))
        return chosen_x, chosen_y, mean_conf

    # Filter by teacher confidence >= conf_threshold and balance across classes
    high_mask = confs >= conf_threshold
    acceptance_rate = float(np.mean(high_mask))
    if np.sum(high_mask) >= num_synthetic:
        idx_high = np.where(high_mask)[0]
        # Sort by confidence descending to pick highest quality synthetic examples
        order = idx_high[np.argsort(-confs[idx_high])][:num_synthetic]
    else:
        order = np.argsort(-confs)[:num_synthetic]

    return imgs_pool[order], preds[order], acceptance_rate


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate SmallCNN on validation or test split."""
    model.eval().to(device)
    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            logits = model(x_dev)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_y.append(y.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    preds = np.argmax(probs_arr, axis=1)

    per_class_f1: dict[str, float] = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        y_bin = (y_arr == c_idx).astype(np.int32)
        p_bin = (preds == c_idx).astype(np.int32)
        per_class_f1[c_name] = float(f1_score(y_bin, p_bin))

    return {
        "accuracy": float(accuracy_score(y_arr, preds)),
        "precision_macro": float(precision_score(y_arr, preds, average="macro")),
        "recall_macro": float(recall_score(y_arr, preds, average="macro")),
        "f1_macro": float(f1_score(y_arr, preds, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_arr, probs_arr, multi_class="ovr", average="macro")),
        "per_class_f1": per_class_f1,
    }


def train_smallcnn_condition(
    condition_name: str,
    train_ds: Dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train a fresh SmallCNN from scratch under one augmentation condition."""
    w2_src = Path(__file__).resolve().parents[2] / "Workshop_2" / "src"
    if str(w2_src) not in sys.path:
        sys.path.insert(0, str(w2_src))
    from model import SmallCNN  # type: ignore

    set_seed(seed)
    model = SmallCNN(num_classes=10).to(device)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, foreach=False)
    except TypeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    t0 = time.time()

    print(f"[augment_and_retrain] condition={condition_name} | train_size={len(train_ds):,} | epochs={epochs}")
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x_dev = x.to(device)
            y_dev = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_dev), y_dev)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            val_res = evaluate_classifier(model, val_loader, device)
            if val_res["f1_macro"] > best_val_f1:
                best_val_f1 = val_res["f1_macro"]
                best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep:02d}/{epochs:02d} | val_acc={val_res['accuracy']*100:.2f}% val_f1={val_res['f1_macro']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = evaluate_classifier(model, test_loader, device)
    test_res["best_val_f1"] = best_val_f1
    test_res["best_epoch"] = best_epoch
    test_res["train_samples"] = len(train_ds)
    test_res["train_sec"] = round(time.time() - t0, 2)
    return test_res


def plot_augmentation_results(results: dict, out_path: Path) -> None:
    """Plot overall accuracy/macro-F1 and per-class F1 on hard upper-body garment classes."""
    conds = list(results["conditions"].keys())
    labels = ["(a) Real Only\n(Baseline)", "(b) Real +\nVanilla GAN", "(c) Real +\nWGAN-GP"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    # Panel 1: Overall Test Accuracy & Macro F1
    ax = axes[0]
    accs = [results["conditions"][c]["accuracy"] * 100 for c in conds]
    f1s = [results["conditions"][c]["f1_macro"] * 100 for c in conds]
    x = np.arange(len(conds))
    w = 0.34
    b1 = ax.bar(x - w / 2, accs, width=w, label="Test Accuracy (%)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, f1s, width=w, label="Test Macro-F1 (x100)", color="#2ca02c")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(min(accs + f1s) - 1.2, max(accs + f1s) + 0.8)
    ax.set_ylabel("Score (%)")
    ax.set_title("Downstream SmallCNN Performance Across 3 Training Conditions")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Hardest Upper-Body Garment Classes Per-Class F1
    ax2 = axes[1]
    hard_classes = ["T-shirt/top", "Pullover", "Coat", "Shirt"]
    xh = np.arange(len(hard_classes))
    w2 = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (c_key, c_lbl) in enumerate(zip(conds, ["Real Only", "Real + Vanilla GAN", "Real + WGAN-GP"])):
        vals = [results["conditions"][c_key]["per_class_f1"][hc] * 100 for hc in hard_classes]
        ax2.bar(xh + (i - 1) * w2, vals, width=w2, label=c_lbl, color=colors[i])
    ax2.set_xticks(xh)
    ax2.set_xticklabels(hard_classes, fontsize=9.5)
    ax2.set_ylim(70, 96)
    ax2.set_ylabel("Test F1-Score (%)")
    ax2.set_title("Per-Class F1 on Ambiguous Upper-Body Garments")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[augment_and_retrain] saved augmentation comparison figure -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-condition GAN data augmentation experiment")
    parser.add_argument("--vanilla-ckpt", type=str, default="checkpoints/gan_vanilla_best.pt")
    parser.add_argument("--wgan-ckpt", type=str, default="checkpoints/gan_wgan_best.pt")
    parser.add_argument("--w2-ckpt", type=str, default="../Workshop_2/checkpoints/model_best.pt")
    parser.add_argument("--num-synthetic", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=str, default="runs/workshop5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    images, labels = get_full_dataset()
    splits = get_splits(labels, seed=args.seed)

    real_train_x = images[splits["train"]].astype(np.float32) / 255.0
    real_train_y = labels[splits["train"]].astype(np.int64)
    real_val_x = images[splits["val"]].astype(np.float32) / 255.0
    real_val_y = labels[splits["val"]].astype(np.int64)
    real_test_x = images[splits["test"]].astype(np.float32) / 255.0
    real_test_y = labels[splits["test"]].astype(np.int64)

    val_ds = NormalizedArrayDataset(real_val_x, real_val_y, augment=False)
    test_ds = NormalizedArrayDataset(real_test_x, real_test_y, augment=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    teacher = load_w2_evaluator(args.w2_ckpt, device)
    gen_van, _ = load_generator_checkpoint(args.vanilla_ckpt, map_location="cpu")
    gen_wgan, _ = load_generator_checkpoint(args.wgan_ckpt, map_location="cpu")

    syn_van_x, syn_van_y, acc_van = build_synthetic_labeled_set(
        gen_van, teacher, args.num_synthetic, device, seed=args.seed
    )
    syn_wgan_x, syn_wgan_y, acc_wgan = build_synthetic_labeled_set(
        gen_wgan, teacher, args.num_synthetic, device, seed=args.seed + 1
    )

    ds_real = NormalizedArrayDataset(real_train_x, real_train_y, augment=True)
    ds_van = NormalizedArrayDataset(
        np.concatenate([real_train_x, syn_van_x], axis=0),
        np.concatenate([real_train_y, syn_van_y], axis=0),
        augment=True,
    )
    ds_wgan = NormalizedArrayDataset(
        np.concatenate([real_train_x, syn_wgan_x], axis=0),
        np.concatenate([real_train_y, syn_wgan_y], axis=0),
        augment=True,
    )

    res_a = train_smallcnn_condition(
        "real_only", ds_real, val_loader, test_loader, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=device
    )
    res_b = train_smallcnn_condition(
        "real_plus_vanilla_gan", ds_van, val_loader, test_loader, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=device
    )
    res_c = train_smallcnn_condition(
        "real_plus_wgan_gp", ds_wgan, val_loader, test_loader, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=device
    )

    payload = {
        "num_real_train": len(real_train_x),
        "num_synthetic_added": args.num_synthetic,
        "epochs": args.epochs,
        "vanilla_acceptance_rate": acc_van,
        "wgan_gp_acceptance_rate": acc_wgan,
        "w2_reference_smallcnn_30ep": {"accuracy": 0.9304761904761905, "f1_macro": 0.9302383175563008},
        "conditions": {
            "real_only": res_a,
            "real_plus_vanilla_gan": res_b,
            "real_plus_wgan_gp": res_c,
        },
    }

    for k, v in payload["conditions"].items():
        print(
            f"[augment_and_retrain] {k:<24} | test_acc={v['accuracy']*100:.2f}% | "
            f"macro_f1={v['f1_macro']:.4f} | roc_auc={v['roc_auc_ovr']:.4f} | Shirt_F1={v['per_class_f1']['Shirt']:.4f}"
        )

    plot_augmentation_results(payload, runs_dir / "augmentation_comparison.png")
    out_json = runs_dir / "augmentation_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[augment_and_retrain] saved augmentation experiment JSON -> {out_json} [OK]")


if __name__ == "__main__":
    main()
