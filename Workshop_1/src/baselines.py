"""
src/baselines.py
----------------
Entrenamiento, evaluación y comparación de los tres modelos base de aprendizaje
supervisado: Regresión Logística, Random Forest e HistGradientBoosting.
Calcula métricas multiclase (Accuracy, Precision, Recall, F1, ROC-AUC) y exporta
resultados reproducibles a runs/results.json y modelos a checkpoints/.
"""

import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

import sys

# Permitir ejecución tanto directa (python src/baselines.py) como empaquetada (python -m src.baselines)
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from .data_loader import CLASS_NAMES, get_or_create_splits, load_full_dataset
    from .features import FeaturePipeline
except ImportError:
    from src.data_loader import CLASS_NAMES, get_or_create_splits, load_full_dataset
    from src.features import FeaturePipeline


def evaluate_classifier(model, X, y, class_names: list[str]) -> dict:
    """
    Calcula métricas completas de desempeño para un modelo clasificador sobre un conjunto dado.
    """
    y_pred = model.predict(X)

    # Probabilidades para ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        # Softmax simple para transformar distancias a probabilidades
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_prob = exp_df / np.sum(exp_df, axis=1, keepdims=True)
    else:
        y_prob = None

    acc = float(accuracy_score(y, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)

    # AUC-ROC One-vs-Rest
    if y_prob is not None:
        y_bin = label_binarize(y, classes=list(range(len(class_names))))
        auc_roc = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    else:
        auc_roc = None

    cm = confusion_matrix(y, y_pred).tolist()

    # Reporte por clase
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1_score": float(f1_cls[i]),
            "support": int(supp_cls[i]),
        }

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weight),
        "recall_weighted": float(r_weight),
        "f1_weighted": float(f1_weight),
        "roc_auc_ovr_macro": auc_roc,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def run_experiments(
    save_models: bool = True,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Ejecuta el pipeline completo: carga de datos, particionamiento 70/15/15,
    escalado y PCA en entrenamiento, ajuste de los 3 modelos base y evaluación.
    """
    base_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = base_dir / "checkpoints"
    runs_dir = base_dir / "runs"
    data_dir = base_dir / "data" / "raw"
    splits_file = base_dir / "data" / "splits.json"

    print("Cargando conjunto de datos unificado...")
    X_all, y_all = load_full_dataset(data_dir)

    print("Obteniendo particiones reproducibles (70% train, 15% val, 15% test)...")
    splits = get_or_create_splits(y_all, splits_file=splits_file, seed=seed)

    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]
    test_idx = splits["test_indices"]

    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
    X_val_raw, y_val = X_all[val_idx], y_all[val_idx]
    X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

    print(f"Dimensiones crudas: Train={X_train_raw.shape}, Val={X_val_raw.shape}, Test={X_test_raw.shape}")

    print("Ajustando FeaturePipeline (StandardScaler + PCA 95% varianza) sobre entrenamiento...")
    pipeline = FeaturePipeline(use_pca=use_pca, pca_variance=pca_variance, seed=seed)
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    print(f"Dimensiones transformadas (PCA={pipeline.n_components_} componentes):")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=seed,
            n_jobs=-1,
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=seed,
        ),
    }

    all_results = {
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

    summary_rows = []

    for name, clf in models.items():
        print("\n" + "=" * 50)
        print(f"Entrenando {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"Tiempo de entrenamiento: {train_time:.2f} s")

        val_metrics = evaluate_classifier(clf, X_val, y_val, CLASS_NAMES)
        test_metrics = evaluate_classifier(clf, X_test, y_test, CLASS_NAMES)

        print(f"  Validacion - Accuracy: {val_metrics['accuracy']:.4f} | F1 macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Prueba     - Accuracy: {test_metrics['accuracy']:.4f} | F1 macro: {test_metrics['f1_macro']:.4f} | AUC: {test_metrics['roc_auc_ovr_macro']:.4f}")

        all_results["models"][name] = {
            "train_time_sec": train_time,
            "validation": val_metrics,
            "test": test_metrics,
        }

        if save_models:
            model_path = checkpoints_dir / f"{name}.joblib"
            joblib.dump(clf, model_path)
            print(f"  Modelo serializado en: {model_path}")

        summary_rows.append({
            "Modelo": name,
            "Tiempo (s)": round(train_time, 2),
            "Val Acc": round(val_metrics["accuracy"], 4),
            "Val F1-Macro": round(val_metrics["f1_macro"], 4),
            "Test Acc": round(test_metrics["accuracy"], 4),
            "Test F1-Macro": round(test_metrics["f1_macro"], 4),
            "Test AUC-ROC": round(test_metrics["roc_auc_ovr_macro"], 4) if test_metrics["roc_auc_ovr_macro"] else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO DE MODELOS BASE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    runs_dir.mkdir(parents=True, exist_ok=True)
    results_path = runs_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados guardados exitosamente en: {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiments()
