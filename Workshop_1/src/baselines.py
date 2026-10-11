"""
Trains and evaluates three baseline classifiers (Logistic Regression, Random Forest,
HistGradientBoosting). Results go to runs/results.json and models to checkpoints/.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow both `python src/baselines.py` and `python -m src.baselines`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .data_loader import CLASS_NAMES, get_or_create_splits, load_full_dataset
    from .features import FeaturePipeline
except ImportError:
    from src.data_loader import CLASS_NAMES, get_or_create_splits, load_full_dataset
    from src.features import FeaturePipeline


def evaluate_classifier(model, X, y, class_names: list[str]) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    metrics = {"accuracy": float(accuracy_score(y, y_pred))}
    for average in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average=average, zero_division=0)
        metrics[f"precision_{average}"] = float(p)
        metrics[f"recall_{average}"] = float(r)
        metrics[f"f1_{average}"] = float(f1)

    metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y, y_prob, multi_class="ovr", average="macro"))
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()

    p, r, f1, support = precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
    metrics["per_class"] = {
        name: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return metrics


def run_experiments(
    save_models: bool = True,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    seed: int = 42,
) -> dict:
    base_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = base_dir / "checkpoints"
    runs_dir = base_dir / "runs"

    X_all, y_all = load_full_dataset(base_dir / "data" / "raw")
    splits = get_or_create_splits(y_all, splits_file=base_dir / "data" / "splits.json", seed=seed)
    train_idx, val_idx, test_idx = splits["train_indices"], splits["val_indices"], splits["test_indices"]

    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    # Scaler and PCA are fitted on the training split only
    pipeline = FeaturePipeline(use_pca=use_pca, pca_variance=pca_variance, seed=seed)
    X_train = pipeline.fit_transform(X_all[train_idx])
    X_val = pipeline.transform(X_all[val_idx])
    X_test = pipeline.transform(X_all[test_idx])

    models = {
        # PCA components have very different scales, so rescale them for lbfgs to converge
        "Logistic_Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed),
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=seed),
    }

    results = {
        "config": {
            "use_pca": use_pca,
            "pca_components": pipeline.n_components_,
            "seed": seed,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
        },
        "models": {},
    }
    summary = []

    if save_models:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        val = evaluate_classifier(clf, X_val, y_val, CLASS_NAMES)
        test = evaluate_classifier(clf, X_test, y_test, CLASS_NAMES)
        print(f"{name}: {train_time:.1f}s | val acc {val['accuracy']:.4f} | test acc {test['accuracy']:.4f}")

        results["models"][name] = {"train_time_sec": train_time, "validation": val, "test": test}

        if save_models:
            joblib.dump(clf, checkpoints_dir / f"{name}.joblib")

        summary.append({
            "Model": name,
            "Time (s)": round(train_time, 2),
            "Val Acc": round(val["accuracy"], 4),
            "Val F1-Macro": round(val["f1_macro"], 4),
            "Test Acc": round(test["accuracy"], 4),
            "Test F1-Macro": round(test["f1_macro"], 4),
            "Test AUC-ROC": round(test["roc_auc_ovr_macro"], 4),
        })

    print(pd.DataFrame(summary).to_string(index=False))

    runs_dir.mkdir(parents=True, exist_ok=True)
    results_path = runs_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiments()