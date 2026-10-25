"""Comprehensive medical evaluation for Workshop 6.

Evaluates one checkpoint (or every checkpoint in checkpoints/) on the test
split and produces:
- test_metrics.json: AUC-ROC, threshold table (sensitivity/specificity/PPV/NPV),
  Youden operating point, confusion counts
- roc_curves.png: ROC curves for all evaluated runs
- ablation_results.json: summary table across runs (for the report)

Usage:
    python -m src.evaluate_medical                       # all *_best.pt checkpoints
    python -m src.evaluate_medical --checkpoint checkpoints/full_recipe_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import medical_preprocessing as mp
from src.medical_metrics import full_evaluation
from src.train_medical import build_arrays, predict_proba
from src.transfer import SmallCNN


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str = "data",
    device: str = "cpu",
) -> dict:
    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SmallCNN(num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    arrays, labels, _ = build_arrays(cfg["preprocess"], data_root)
    proba = predict_proba(model, arrays["test"], dev)
    y = labels["test"]

    ev = full_evaluation(y, proba)
    ev.update(
        {
            "name": cfg["name"],
            "checkpoint": str(checkpoint),
            "config": {k: v for k, v in cfg.items() if k != "name"},
            "val_auc_at_selection": ckpt.get("val_auc"),
        }
    )
    return ev


def plot_roc_curves(results: list[dict], out_path: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for r in results:
        # recompute probas is expensive; reuse curve points stored during eval
        if "fpr" not in r:
            continue
        ax.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC {r['auc_roc']:.3f})")
    if not any("fpr" in r for r in results):
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC on PneumoniaMNIST test")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Medical evaluation (Workshop 6)")
    p.add_argument("--checkpoint", default=None, help="single checkpoint; default: all *_best.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="runs/workshop6")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from sklearn.metrics import roc_curve

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        ckpts = [Path(args.checkpoint)]
    else:
        ckpts = sorted(Path("checkpoints").glob("*_best.pt"))
    if not ckpts:
        raise SystemExit("no checkpoints found; train a run first")

    results = []
    for ck in ckpts:
        r = evaluate_checkpoint(ck, args.data_root, args.device)
        # attach ROC curve points for the figure
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        arrays, labels, _ = build_arrays(cfg["preprocess"], args.data_root)
        model = SmallCNN(num_classes=1)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        proba = predict_proba(model, arrays["test"], torch.device(args.device))
        fpr, tpr, _ = roc_curve(labels["test"], proba)
        r["fpr"] = fpr.tolist()
        r["tpr"] = tpr.tolist()
        results.append(r)
        youden = r["youden"]
        print(f"[eval] {r['name']:<20} AUC {r['auc_roc']:.4f} | @0.5: sens "
              f"{r['thresholds'][3]['sensitivity']:.3f} spec {r['thresholds'][3]['specificity']:.3f} | "
              f"youden thr {youden['threshold']:.2f} sens {youden['sensitivity']:.3f} "
              f"spec {youden['specificity']:.3f}")

    # per-checkpoint JSON + combined ablation table
    for r in results:
        slim = {k: v for k, v in r.items() if k not in ("fpr", "tpr")}
        (out_dir / f"test_metrics_{r['name']}.json").write_text(json.dumps(slim, indent=2))

    ablation = []
    for r in results:
        t5 = next(t for t in r["thresholds"] if abs(t["threshold"] - 0.5) < 1e-9)
        ablation.append(
            {
                "name": r["name"],
                "mode": r["config"].get("mode"),
                "preprocess": r["config"].get("preprocess"),
                "class_weights": r["config"].get("class_weights"),
                "augment": r["config"].get("augment"),
                "auc_roc": r["auc_roc"],
                "val_auc": r.get("val_auc_at_selection"),
                "sens@0.5": t5["sensitivity"],
                "spec@0.5": t5["specificity"],
                "ppv@0.5": t5["ppv"],
                "npv@0.5": t5["npv"],
            }
        )
    (out_dir / "ablation_results.json").write_text(json.dumps(ablation, indent=2))
    plot_roc_curves(results, out_dir / "roc_curves.png")
    print(f"[eval] ablation table -> {out_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
