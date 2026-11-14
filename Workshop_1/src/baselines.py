"""
Trains and evaluates three baseline classifiers (Logistic Regression, Random Forest,
HistGradientBoosting) under four settings: with/without data augmentation and
with/without PCA. Results go to runs/results.json and models to
checkpoints/<feature_variant>/.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
    from .data_loader import (
        CLASS_NAMES,
        generate_augmented_training_batch,
        get_or_create_splits,
        load_full_dataset,
    )
    from .features import FeaturePipeline
except ImportError:
    from src.data_loader import (
        CLASS_NAMES,
        generate_augmented_training_batch,
        get_or_create_splits,
        load_full_dataset,
    )
    from src.features import FeaturePipeline


SplitData = dict[str, np.ndarray]
Dataset = dict[str, SplitData]
ExperimentData = dict[str, Dataset]

# Variant name -> (augment the training split, apply PCA)
VARIANT_SPECS: dict[str, tuple[bool, bool]] = {
    "no_aug_no_pca": (False, False),
    "no_aug_pca": (False, True),
    "aug_no_pca": (True, False),
    "aug_pca": (True, True),
}


def _variant_flags(variant: str) -> tuple[bool, bool]:
    """
    Return the (augment, use_pca) flags for a feature variant name.

    Known variants are resolved through VARIANT_SPECS; any other name falls back
    to a small heuristic so custom variant dictionaries still work.

    Args:
        variant (str): Feature variant name.

    Returns:
        tuple[bool, bool]: Whether training data is augmented and whether PCA is applied.
    """
    if variant in VARIANT_SPECS:
        return VARIANT_SPECS[variant]

    name = variant.lower()
    augment = not ("no_aug" in name or "without_aug" in name)
    use_pca = "pca" in name and not ("no_pca" in name or "without_pca" in name)
    return augment, use_pca


def build_default_models(seed: int = 42) -> dict[str, Any]:
    """
    Build the three baseline classifiers used in the workshop.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        dict[str, Any]: Model name to unfitted scikit-learn estimator mapping.
    """
    return {
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


def evaluate_classifier(model, X, y, class_names: list[str]) -> dict:
    """
    Compute overall and per-class classification metrics.

    Args:
        model (object): Fitted classifier exposing predict and predict_proba.
        X (np.ndarray): Feature matrix to evaluate.
        y (np.ndarray): True labels.
        class_names (list[str]): Human-readable class names ordered by label.

    Returns:
        dict: Accuracy, precision, recall, F1, ROC-AUC, per-class scores, and confusion matrix.
    """
    y_pred = model.predict(X)

    metrics = {"accuracy": float(accuracy_score(y, y_pred))}
    for average in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average=average, zero_division=0)
        metrics[f"precision_{average}"] = float(p)
        metrics[f"recall_{average}"] = float(r)
        metrics[f"f1_{average}"] = float(f1)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
        if len(np.unique(y)) == 2:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y, y_prob[:, 1]))
        else:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
            )
    else:
        metrics["roc_auc_ovr_macro"] = None

    metrics["confusion_matrix"] = confusion_matrix(
        y, y_pred, labels=np.arange(len(class_names))
    ).tolist()

    p, r, f1, support = precision_recall_fscore_support(
        y,
        y_pred,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
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


def _load_dataset_and_splits(
    *,
    seed: int = 42,
    data_dir: Path | None = None,
    splits_file: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load all images and labels plus the reproducible train/val/test split.

    Args:
        seed (int): Random seed used if the split file must be created.
        data_dir (Path | None): Directory containing the raw .gz files; if None, the default is used.
        splits_file (Path | None): Path to the split indices JSON file; if None, the default is used.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: Image array X, label array y, and split indices.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = data_dir or base_dir / "data" / "raw"
    splits_file = splits_file or base_dir / "data" / "splits.json"

    X_all, y_all = load_full_dataset(data_dir)
    splits = get_or_create_splits(y_all, splits_file=splits_file, seed=seed)
    return X_all, y_all, splits


def _build_split_arrays(
    X_all: np.ndarray,
    y_all: np.ndarray,
    splits: dict,
    *,
    use_pca: bool,
    pca_variance: float,
    seed: int,
    augment: bool = False,
    augmentation_ratio: float = 0.20,
) -> Dataset:
    """
    Scale and optionally reduce the data, returning train/val/test arrays.

    Augmentation (when enabled) is applied only to the training split, before
    the scaler and PCA are fitted, so validation and test stay untouched.

    Args:
        X_all (np.ndarray): All images with shape (N, 28, 28).
        y_all (np.ndarray): All labels.
        splits (dict): Split indices under train_indices, val_indices, and test_indices.
        use_pca (bool): Whether to apply PCA after standard scaling.
        pca_variance (float): Variance ratio retained by PCA.
        seed (int): Random seed for reproducibility.
        augment (bool): Whether to append augmented training images.
        augmentation_ratio (float): Fraction of training images to augment.

    Returns:
        dict[str, dict[str, np.ndarray]]: Nested train/val/test dictionaries with X and y arrays.
    """
    train_idx = np.asarray(splits["train_indices"], dtype=np.int64)
    val_idx = np.asarray(splits["val_indices"], dtype=np.int64)
    test_idx = np.asarray(splits["test_indices"], dtype=np.int64)

    X_train_images = X_all[train_idx]
    y_train = y_all[train_idx]
    if augment and augmentation_ratio > 0:
        X_train_images, y_train = generate_augmented_training_batch(
            X_train_images,
            y_train,
            augmentation_ratio=augmentation_ratio,
            seed=seed,
        )

    # The preprocessing pipeline is fitted on the training split only
    pipeline = FeaturePipeline(use_pca=use_pca, pca_variance=pca_variance, seed=seed)
    X_train = pipeline.fit_transform(X_train_images)
    X_val = pipeline.transform(X_all[val_idx])
    X_test = pipeline.transform(X_all[test_idx])

    return {
        "train": {"X": X_train, "y": y_train},
        "val": {"X": X_val, "y": y_all[val_idx]},
        "test": {"X": X_test, "y": y_all[test_idx]},
    }


def prepare_data(
    *,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    seed: int = 42,
    augment: bool = False,
    augmentation_ratio: float = 0.20,
    data_dir: Path | None = None,
    splits_file: Path | None = None,
) -> Dataset:
    """
    Prepare train/val/test feature matrices for one augmentation/PCA setting.

    Args:
        use_pca (bool): Whether to apply PCA after standard scaling.
        pca_variance (float): Variance ratio retained by PCA.
        seed (int): Random seed for reproducibility.
        augment (bool): Whether to append augmented training images.
        augmentation_ratio (float): Fraction of training images to augment.
        data_dir (Path | None): Directory containing the raw .gz files; if None, the default is used.
        splits_file (Path | None): Path to the split indices JSON file; if None, the default is used.

    Returns:
        dict[str, dict[str, np.ndarray]]: Nested train/val/test dictionaries with X and y arrays.
    """
    X_all, y_all, splits = _load_dataset_and_splits(
        seed=seed, data_dir=data_dir, splits_file=splits_file
    )
    return _build_split_arrays(
        X_all,
        y_all,
        splits,
        use_pca=use_pca,
        pca_variance=pca_variance,
        seed=seed,
        augment=augment,
        augmentation_ratio=augmentation_ratio,
    )


def build_experiment_data(
    *,
    pca_variance: float = 0.95,
    augmentation_ratio: float = 0.20,
    seed: int = 42,
    data_dir: Path | None = None,
    splits_file: Path | None = None,
) -> ExperimentData:
    """
    Prepare the data for all four comparison settings: augmentation x PCA.

    Args:
        pca_variance (float): Variance ratio retained by PCA.
        augmentation_ratio (float): Fraction of training images to augment.
        seed (int): Random seed for reproducibility.
        data_dir (Path | None): Directory containing the raw .gz files; if None, the default is used.
        splits_file (Path | None): Path to the split indices JSON file; if None, the default is used.

    Returns:
        dict[str, dict[str, dict[str, np.ndarray]]]: Data keyed by no_aug_no_pca,
        no_aug_pca, aug_no_pca, and aug_pca.
    """
    X_all, y_all, splits = _load_dataset_and_splits(
        seed=seed, data_dir=data_dir, splits_file=splits_file
    )
    return {
        variant: _build_split_arrays(
            X_all,
            y_all,
            splits,
            use_pca=use_pca,
            pca_variance=pca_variance,
            seed=seed,
            augment=augment,
            augmentation_ratio=augmentation_ratio,
        )
        for variant, (augment, use_pca) in VARIANT_SPECS.items()
    }


def train_models(
    models: Mapping[str, Any],
    data: Dataset,
    *,
    class_names: list[str] = CLASS_NAMES,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """
    Fit every model on the training split and evaluate it on validation and test.

    Args:
        models (Mapping[str, Any]): Model name to unfitted scikit-learn estimator mapping.
        data (dict[str, dict[str, np.ndarray]]): Dataset with train/val/test X and y arrays.
        class_names (list[str]): Human-readable class names ordered by label.

    Returns:
        tuple[dict[str, dict], dict[str, Any]]: Per-model metrics and fitted estimators.
    """
    results: dict[str, dict] = {}
    fitted_models: dict[str, Any] = {}

    for name, estimator in models.items():
        # Clone so the same estimator definition can be reused across feature variants
        model = clone(estimator)

        start = time.time()
        model.fit(data["train"]["X"], data["train"]["y"])
        train_time = time.time() - start

        val = evaluate_classifier(model, data["val"]["X"], data["val"]["y"], class_names)
        test = evaluate_classifier(model, data["test"]["X"], data["test"]["y"], class_names)
        print(f"{name}: {train_time:.1f}s | val acc {val['accuracy']:.4f} | test acc {test['accuracy']:.4f}")

        results[name] = {"train_time_sec": train_time, "validation": val, "test": test}
        fitted_models[name] = model

    return results, fitted_models


def build_comparison_table(experiments: dict, model_names: list[str]) -> pd.DataFrame:
    """
    Build a wide table comparing every model across augmentation and PCA settings.

    Args:
        experiments (dict): Per-variant experiment results keyed by variant name.
        model_names (list[str]): Model names to include in the table.

    Returns:
        pd.DataFrame: Validation/test scores for each variant plus augmentation and PCA deltas.
    """
    variants = list(experiments)
    rows = []
    for name in model_names:
        present: dict[str, dict] = {}
        row: dict[str, Any] = {"Model": name}

        for variant in variants:
            result = experiments.get(variant, {}).get("models", {}).get(name)
            if not result:
                continue
            present[variant] = result
            row[f"Val Acc ({variant})"] = round(result["validation"]["accuracy"], 4)
            row[f"Test Acc ({variant})"] = round(result["test"]["accuracy"], 4)
            row[f"Test F1 ({variant})"] = round(result["test"]["f1_macro"], 4)
            row[f"Train s ({variant})"] = round(result["train_time_sec"], 2)

        # Positive values mean the treatment improved test accuracy.
        comparisons = (
            ("no_aug_no_pca", "aug_no_pca", "Δ Test Acc (aug, no PCA)"),
            ("no_aug_pca", "aug_pca", "Δ Test Acc (aug, PCA)"),
            ("no_aug_no_pca", "no_aug_pca", "Δ Test Acc (PCA, no aug)"),
            ("aug_no_pca", "aug_pca", "Δ Test Acc (PCA, aug)"),
        )
        for baseline, treatment, column in comparisons:
            if baseline in present and treatment in present:
                row[column] = round(
                    present[treatment]["test"]["accuracy"]
                    - present[baseline]["test"]["accuracy"],
                    4,
                )

        rows.append(row)
    return pd.DataFrame(rows)


def run_experiments(
    models: Mapping[str, Any] | None = None,
    data: ExperimentData | None = None,
    *,
    save_models: bool = True,
    pca_variance: float = 0.95,
    augmentation_ratio: float = 0.20,
    seed: int = 42,
    runs_dir: Path | None = None,
    checkpoints_dir: Path | None = None,
) -> dict:
    """
    Train and evaluate the given models across augmentation and PCA settings.

    Args:
        models (Mapping[str, Any] | None): Model name to estimator mapping; defaults to the workshop baselines.
        data (dict[str, dict[str, dict[str, np.ndarray]]] | None): Prebuilt data keyed by feature variant;
            if None, it is built from the raw dataset.
        save_models (bool): Whether to save fitted models under checkpoints/<feature_variant>/.
        pca_variance (float): Variance ratio retained by PCA when data is built internally.
        augmentation_ratio (float): Fraction of training images augmented when data is built internally.
        seed (int): Random seed for reproducibility.
        runs_dir (Path | None): Directory where results.json is saved.
        checkpoints_dir (Path | None): Directory where fitted models are saved.

    Returns:
        dict: Configuration, per-variant metrics, and the model comparison table.
    """
    base_dir = Path(__file__).resolve().parent.parent
    runs_dir = runs_dir or base_dir / "runs"
    checkpoints_dir = checkpoints_dir or base_dir / "checkpoints"

    if models is None:
        models = build_default_models(seed=seed)
    if data is None:
        data = build_experiment_data(
            pca_variance=pca_variance,
            augmentation_ratio=augmentation_ratio,
            seed=seed,
        )
    if not data:
        raise ValueError("data must contain at least one feature variant.")

    results: dict[str, Any] = {
        "config": {
            "seed": seed,
            "pca_variance": pca_variance,
            "augmentation_ratio": augmentation_ratio,
            "feature_variants": list(data),
        },
        "experiments": {},
    }

    if save_models:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for variant, variant_data in data.items():
        augment, use_pca = _variant_flags(variant)
        n_features = int(variant_data["train"]["X"].shape[1])
        n_train = int(len(variant_data["train"]["y"]))
        print(f"\n=== {variant.replace('_', ' ').title()} ({n_features} features, {n_train:,} train samples) ===")

        model_results, fitted_models = train_models(models, variant_data)
        results["experiments"][variant] = {
            "config": {
                "use_pca": use_pca,
                "augmented": augment,
                "augmentation_ratio": augmentation_ratio if augment else 0.0,
                "n_features": n_features,
                "sample_counts": {
                    split: int(len(variant_data[split]["y"]))
                    for split in ("train", "val", "test")
                },
            },
            "models": model_results,
        }

        if save_models:
            variant_dir = checkpoints_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            for name, model in fitted_models.items():
                joblib.dump(model, variant_dir / f"{name}.joblib")

    comparison = build_comparison_table(results["experiments"], list(models))
    results["comparison"] = comparison.to_dict(orient="records")

    print("\n=== AUGMENTATION x PCA COMPARISON ===")
    print(comparison.to_string(index=False))

    runs_dir.mkdir(parents=True, exist_ok=True)
    results_path = runs_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main() -> dict:
    """
    Build the default models and data, then run the augmentation x PCA comparison.

    Args:
        None

    Returns:
        dict: Configuration, per-variant metrics, and the model comparison table.
    """
    seed = 42
    pca_variance = 0.95
    augmentation_ratio = 0.20
    models = build_default_models(seed=seed)
    data = build_experiment_data(
        pca_variance=pca_variance,
        augmentation_ratio=augmentation_ratio,
        seed=seed,
    )
    return run_experiments(
        models=models,
        data=data,
        pca_variance=pca_variance,
        augmentation_ratio=augmentation_ratio,
        seed=seed,
    )


if __name__ == "__main__":
    main()
