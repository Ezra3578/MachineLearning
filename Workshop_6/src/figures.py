"""Figure generation for Workshop 6.

Rebuilds the three figures that the report uses but that no other script
produced:

- dataset_overview.png: normal / pneumonia / CLAHE samples at 28x28
- training_curves.png: validation AUC per epoch for every ablation run
- confusion_matrices.png: selected model at a permissive threshold, the
  default, and the Youden point

Everything is read from artifacts already on disk (data, train logs,
checkpoints), so the figures can be regenerated after a rerun:

    python -m src.figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import medical_preprocessing as mp
from src.device import pick_device
from src.medical_metrics import confusion_counts, youden_threshold
from src.plotting import save_figure
from src.train_medical import build_arrays, predict_proba
from src.transfer import SmallCNN

RUNS = ("scratch", "frozen", "finetune", "finetune_clahe", "finetune_weights", "full_recipe")


def dataset_overview(data_root: str, out_path: Path, n: int = 6, seed: int = 42) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images, labels = mp.load_split_arrays("train", data_root)
    rng = np.random.default_rng(seed)
    normal = rng.choice(np.where(labels == 0)[0], n, replace=False)
    pneumonia = rng.choice(np.where(labels == 1)[0], n, replace=False)

    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.2), squeeze=False)
    rows = (
        ("normal", normal, images[normal]),
        ("pneumonia", pneumonia, images[pneumonia]),
        ("normal + CLAHE", normal, mp.apply_clahe(images[normal])),
    )
    for r, (label, _, batch) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(batch[c], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(label, fontsize=10)
    fig.suptitle("PneumoniaMNIST training samples (28x28)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def training_curves(runs_dir: Path, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in RUNS:
        log = runs_dir / name / "train_log.csv"
        if not log.exists():
            continue
        data = np.genfromtxt(log, delimiter=",", names=True)
        ax.plot(data["epoch"], data["val_auc"], marker="o", ms=3, lw=1.4, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC-ROC")
    ax.set_title("Validation AUC per epoch (all ablation runs)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def confusion_matrices(
    checkpoint: str | Path, data_root: str, out_path: Path, device: str = "auto"
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = pick_device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SmallCNN(num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    arrays, labels, _ = build_arrays(cfg["preprocess"], data_root)
    proba = predict_proba(model, arrays["test"], dev)
    y = labels["test"]
    youden = round(youden_threshold(y, proba), 4)

    points = (0.2, 0.5, youden)
    titles = ("threshold 0.2", "threshold 0.5", f"Youden {youden:.4f}")
    fig, axes = plt.subplots(1, len(points), figsize=(4.2 * len(points), 4.2))
    for ax, thr, title in zip(axes, points, titles):
        tp, fp, tn, fn = confusion_counts(y, proba, thr)
        grid = np.array([[tn, fp], [fn, tp]])
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max())
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > grid.max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["pred normal", "pred pneumonia"])
        ax.set_yticks([0, 1], ["true normal", "true pneumonia"])
        ax.set_title(title)
    fig.suptitle(f"Confusion matrices of the selected model (test, n={len(y)})", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate the Workshop 6 figures")
    p.add_argument("--checkpoint", default="checkpoints/finetune_weights_best.pt")
    p.add_argument("--runs-dir", default="runs/workshop6")
    p.add_argument("--out-dir", default="runs/workshop6")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    runs_dir, out_dir = Path(args.runs_dir), Path(args.out_dir)
    written = [
        dataset_overview(args.data_root, out_dir / "dataset_overview.png"),
        training_curves(runs_dir, out_dir / "training_curves.png"),
        confusion_matrices(args.checkpoint, args.data_root,
                           out_dir / "confusion_matrices.png", args.device),
    ]
    for path in written:
        print(f"[figures] -> {path}")


if __name__ == "__main__":
    main()
