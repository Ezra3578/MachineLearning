"""Clinical evaluation metrics for Workshop 6.

Standard ML metrics (accuracy, AUC-ROC) are not enough for clinical decisions.
This module reports the quantities clinicians actually use:

- sensitivity (recall): of the pneumonic patients, how many did we flag?
- specificity: of the healthy patients, how many did we clear?
- PPV (precision): of the flagged patients, how many truly have pneumonia?
- NPV: of the cleared patients, how many are truly healthy?

All as functions of the decision threshold, because the operating point is a
clinical choice (triage wants high sensitivity; autonomous clearance wants high
specificity), not a statistical default of 0.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def confusion_counts(y_true: np.ndarray, proba: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) with class 1 (pneumonia) as positive."""
    pred = (proba >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def clinical_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict:
    """Sensitivity/specificity/PPV/NPV (+accuracy, balanced accuracy) at `thr`."""
    tp, fp, tn, fn = confusion_counts(y_true, proba, thr)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "threshold": round(float(thr), 3),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round((sens + spec) / 2, 4) if sens == sens and spec == spec else None,
    }


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds=DEFAULT_THRESHOLDS,
) -> list[dict]:
    return [clinical_metrics(y_true, proba, t) for t in thresholds]


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity (Youden's J)."""
    thrs = np.unique(np.round(proba, 4))
    best_thr, best_j = 0.5, -np.inf
    for t in thrs:
        tp, fp, tn, fn = confusion_counts(y_true, proba, float(t))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec
        if j > best_j:
            best_j, best_thr = j, float(t)
    return float(best_thr)


def full_evaluation(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """AUC + threshold table + Youden operating point."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    t = youden_threshold(y_true, proba)
    return {
        "n": int(len(y_true)),
        "prevalence": round(float(y_true.mean()), 4),
        "auc_roc": round(float(roc_auc_score(y_true, proba)), 4),
        "thresholds": threshold_sweep(y_true, proba),
        "youden": {"threshold": round(t, 3), **clinical_metrics(y_true, proba, t)},
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.2 * y + rng.uniform(0, 0.7, 500), 0, 1)
    import json

    print(json.dumps(full_evaluation(y, p), indent=2))
