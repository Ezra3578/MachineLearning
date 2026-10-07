"""Placeholders for the Workshop 1 / Workshop 2 comparison (TODO).

W1/W2 are not fully finished, so `src/evaluate.py` imports these constants to
build the comparison table. When W1/W2 close, update ONLY this file:
replace the values below with the final test metrics and delete the TODOs.

Current values come from Workshop_2/README.md (test split, same 70/15/15
protocol, seed 42) and the W1 description (Fashion-MNIST, F-score focus).
"""

from __future__ import annotations

# TODO(W1): reemplazar con las metricas finales del notebook
# Workshop_1/code/SupervisedLearning.ipynb (accuracy, macro-F1 por modelo).
W1_RESULTS: dict[str, dict[str, float | None]] = {
    # Mejor baseline tradicional conocido hasta ahora (HistGradientBoosting).
    "HistGradientBoosting (W1 best)": {"accuracy": 0.8748, "f1_macro": 0.8740, "roc_auc_ovr": None},
    # TODO(W1): agregar filas finales para RandomForest, XGBoost y K-Means.
    "RandomForest (W1)": {"accuracy": None, "f1_macro": None, "roc_auc_ovr": None},  # TODO(W1)
    "XGBoost (W1)": {"accuracy": None, "f1_macro": None, "roc_auc_ovr": None},  # TODO(W1)
}

# TODO(W2): confirmar con runs/test_metrics.json y runs/resnet18/test_metrics.json.
W2_RESULTS: dict[str, dict[str, float | None]] = {
    "SmallCNN (W2)": {"accuracy": 0.9305, "f1_macro": 0.9302, "roc_auc_ovr": 0.9961},
    "ResNet-18 (W2)": {"accuracy": 0.9494, "f1_macro": 0.9494, "roc_auc_ovr": 0.9979},
}

# RL no tiene accuracy supervisada directa: se deriva del retorno con reward ±1.
# accuracy = (mean_return + max_steps) / (2 * max_steps). Ver src/evaluate.py.
RL_METRIC_HELP = "RL accuracy derives from mean episode return; not directly comparable to supervised epochs — see sample-efficiency plot."

TODO_W1_W2 = [
    "TODO(W1): fill final accuracy/macro-F1 for RandomForest, XGBoost, K-Means from Workshop_1.",
    "TODO(W2): re-validate SmallCNN/ResNet-18 numbers against runs/test_metrics.json.",
    "TODO(comparison): align total sample complexity (RL env-steps vs supervised labeled examples).",
]


def comparison_table(rl_metrics: dict) -> str:
    """Render a markdown comparison table: RL agent vs baselines vs W1/W2 stubs."""
    rows = ["| Model | Test accuracy | Macro F1 | Notes |", "|---|---|---|---|"]
    for name, m in {**W1_RESULTS, **W2_RESULTS}.items():
        acc = f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "TODO"
        f1 = f"{m['f1_macro']:.4f}" if m["f1_macro"] is not None else "TODO"
        rows.append(f"| {name} | {acc} | {f1} | supervised baseline |")
    rl_acc = rl_metrics.get("accuracy")
    rl_f1 = rl_metrics.get("f1_macro")
    acc_str = f"{rl_acc:.4f}" if isinstance(rl_acc, float) else "TODO"
    f1_str = f"{rl_f1:.4f}" if isinstance(rl_f1, float) else "TODO"
    rows.append(f"| DQN (W3, this work) | {acc_str} | {f1_str} | RL agent (return-derived) |")
    rows.append("")
    rows.append(RL_METRIC_HELP)
    for todo in TODO_W1_W2:
        rows.append(f"- {todo}")
    return "\n".join(rows)
