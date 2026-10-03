"""Test-set evaluation for Workshop 2.

Loads a checkpoint (``checkpoints/model_best.pt`` by default), evaluates it on the
held-out split with the same metrics used in Workshop 1 (accuracy, macro precision /
recall / F1 + per-class report and confusion matrix), and writes the results to
``runs/test_metrics.json``.

Usage:
    python -m src.evaluate --checkpoint checkpoints/model_best.pt
    python -m src.evaluate --checkpoint checkpoints/model_best.pt --split val --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import CLASS_NAMES, build_dataloaders
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Workshop 2 checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--plot", action="store_true", help="save a confusion-matrix PNG")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "directml":
        import torch_directml

        return torch_directml.device()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml
    except ImportError:
        return torch.device("cpu")
    return torch_directml.device() if torch_directml.is_available() else torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    classes = checkpoint.get("classes", CLASS_NAMES)
    print(f"[evaluate] checkpoint={args.checkpoint} | model={model_name} | trained epoch={checkpoint.get('epoch')}")

    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    _, val_loader, test_loader = build_dataloaders(
        model_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    loader = val_loader if args.split == "val" else test_loader

    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds_all.append(logits.argmax(dim=1).cpu())
            labels_all.append(labels)
            probs_all.append(torch.softmax(logits, dim=1).cpu())

    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    probs = torch.cat(probs_all).numpy()

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": _safe_roc_auc(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "per_class": _per_class_report(labels, preds),
    }

    print(f"\n[evaluate] split={args.split} n={len(labels)}")
    print(f"[evaluate] accuracy={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f} | "
          f"precision_macro={metrics['precision_macro']:.4f} | recall_macro={metrics['recall_macro']:.4f}")
    if metrics["roc_auc_ovr_macro"] is not None:
        print(f"[evaluate] roc_auc_ovr_macro={metrics['roc_auc_ovr_macro']:.4f}")
    print("\n" + classification_report(labels, preds, target_names=classes, zero_division=0))

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": int(len(labels)),
        "model": model_name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
    }
    out_path = runs_dir / f"{args.split}_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[evaluate] metrics written to {out_path}")

    if args.plot:
        _save_confusion_png(np.array(metrics["confusion_matrix"]), classes, runs_dir / f"{args.split}_confusion_matrix.png")


def _per_class_report(labels: np.ndarray, preds: np.ndarray) -> dict:
    report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return {name: {k: float(v) for k, v in stats.items()} for name, stats in report.items() if isinstance(stats, dict) and name in CLASS_NAMES}


def _safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _save_confusion_png(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"[evaluate] confusion matrix figure saved to {path}")
    except Exception as exc:  # pragma: no cover
        print(f"[evaluate] could not save plot: {exc}")


if __name__ == "__main__":
    main()
