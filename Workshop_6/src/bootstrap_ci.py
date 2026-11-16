"""Bootstrap confidence intervals for the Workshop 6 test AUCs.

The test split has 624 images, so differences of a few thousandths of an AUC
point are not resolvable from a single evaluation. This script reports a
percentile bootstrap interval for every run and for the paired contrasts that
the ablation discusses, using the same resampled indices for both members of a
pair (a paired bootstrap on the identical test images):

    python -m src.bootstrap_ci

Writes runs/workshop6/bootstrap_ci.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.device import pick_device
from src.train_medical import build_arrays, predict_proba
from src.transfer import SmallCNN

CONTRASTS = (
    ("finetune", "scratch"),            # warm start vs from-scratch control
    ("frozen", "scratch"),              # frozen features vs control
    ("finetune_weights", "finetune"),   # class weighting
    ("finetune_clahe", "finetune"),     # CLAHE
    ("full_recipe", "finetune_weights"),  # augmentation on top of CLAHE + weights
)


def load_scores(checkpoints: list[Path], data_root: str, device: str):
    dev = pick_device(device)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    scores, labels = {}, None
    for ckpt in checkpoints:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if cfg["preprocess"] not in cache:
            arrays, labels_split, _ = build_arrays(cfg["preprocess"], data_root)
            cache[cfg["preprocess"]] = (arrays["test"], labels_split["test"])
        x, y = cache[cfg["preprocess"]]
        model = SmallCNN(num_classes=1)
        model.load_state_dict(ck["state_dict"])
        model.to(dev).eval()
        scores[cfg["name"]] = predict_proba(model, x, dev)
        labels = y
    return scores, labels


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap AUC intervals (Workshop 6)")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="runs/workshop6/bootstrap_ci.json")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps|privateuseone")
    args = p.parse_args()

    checkpoints = sorted(Path(args.checkpoints).glob("*_best.pt"))
    if not checkpoints:
        raise SystemExit("no checkpoints found")
    scores, y = load_scores(checkpoints, args.data_root, args.device)
    n = len(y)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    valid = np.array([len(np.unique(y[i])) > 1 for i in idx])

    auc = {name: float(roc_auc_score(y, s)) for name, s in scores.items()}
    boot = {name: np.array([roc_auc_score(y[i], s[i]) for i in idx[valid]])
            for name, s in scores.items()}
    summary = {
        name: {
            "auc": round(auc[name], 4),
            "ci95": [round(float(np.percentile(boot[name], 2.5)), 4),
                     round(float(np.percentile(boot[name], 97.5)), 4)],
        }
        for name in scores
    }

    contrasts = {}
    for a, b in CONTRASTS:
        if a not in scores or b not in scores:
            continue
        diffs = np.array([roc_auc_score(y[i], scores[a][i]) - roc_auc_score(y[i], scores[b][i])
                          for i in idx[valid]])
        contrasts[f"{a}_minus_{b}"] = {
            "mean": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "p_gt_zero": round(float((diffs > 0).mean()), 3),
        }

    payload = {
        "n_test": int(n),
        "n_boot": int(valid.sum()),
        "seed": args.seed,
        "auc": summary,
        "paired_contrasts": contrasts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for name, s in summary.items():
        print(f"[bootstrap] {name:<18} AUC {s['auc']:.4f}  CI {s['ci95']}")
    for name, c in contrasts.items():
        print(f"[bootstrap] {name:<40} {c['mean']:+.4f}  CI {c['ci95']}")
    print(f"[bootstrap] -> {out}")


if __name__ == "__main__":
    main()
