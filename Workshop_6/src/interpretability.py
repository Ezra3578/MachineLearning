"""Interpretability analysis for Workshop 6: Grad-CAM on chest X-rays.

For every selected test case the figure shows: input X-ray, model output
(pneumonia probability) and the gradient-weighted class activation map
(Selvaraju et al., 2017) overlaid on the image. Cases are grouped into
clinically meaningful categories: correct pneumonia, correct normal, false
negatives (missed pneumonia, the dangerous error) and false positives.

The qualitative validation question (guide: "do saliency maps highlight
clinically-relevant anatomy?") is answered in the report by inspecting whether
attention concentrates over the lung fields rather than the image borders,
diaphragm or medially projected hardware.

Usage:
    python -m src.interpretability --checkpoint checkpoints/full_recipe_best.pt \
        --out runs/workshop6/gradcam_analysis.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import medical_preprocessing as mp
from src.medical_metrics import confusion_counts
from src.train_medical import build_arrays, predict_proba
from src.transfer import SmallCNN


class GradCAM:
    """Gradient-weighted Class Activation Mapping via forward/backward hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._on_forward)
        self._bwd = target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module, inputs, output) -> None:  # noqa: ARG002
        self._activation = output

    def _on_backward(self, module, grad_input, grad_output) -> None:  # noqa: ARG002
        self._gradient = grad_output[0]

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """CAM for the pneumonia logit of a single (C,H,W) tensor, normalized 0-1."""
        self.model.zero_grad(set_to_none=True)
        logit = self.model(image.unsqueeze(0)).squeeze()
        logit.backward()  # positive class = pneumonia: map shows evidence FOR pneumonia
        weights = self._gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activation).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().numpy()

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def select_cases(y_true, proba, thr: float = 0.5, n_per_group: int = 4) -> dict[str, list[int]]:
    """Indices for: correct pneumonia, correct normal, false negatives, false positives."""
    pred = (proba >= thr).astype(int)
    correct_p = [i for i in range(len(y_true)) if y_true[i] == 1 and pred[i] == 1]
    correct_n = [i for i in range(len(y_true)) if y_true[i] == 0 and pred[i] == 0]
    fn = [i for i in range(len(y_true)) if y_true[i] == 1 and pred[i] == 0]
    fp = [i for i in range(len(y_true)) if y_true[i] == 0 and pred[i] == 1]
    # most confident first for correct groups; most confident errors first too
    correct_p.sort(key=lambda i: -proba[i])
    correct_n.sort(key=lambda i: proba[i])
    fn.sort(key=lambda i: proba[i])       # lowest proba = most severe miss
    fp.sort(key=lambda i: -proba[i])      # highest proba = most confident false alarm
    return {
        "correct_pneumonia": correct_p[:n_per_group],
        "correct_normal": correct_n[:n_per_group],
        "false_negative": fn[:n_per_group],
        "false_positive": fp[:n_per_group],
    }


def make_figure(
    images: np.ndarray,  # uint8 (N,28,28)
    cases: dict[str, list[int]],
    y_true, proba, thr,
    cam_fn,
    out_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(cases.keys())
    n_rows = len(groups)
    n_cols = max(max(len(v) for v in cases.values()), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.6 * n_rows), squeeze=False)
    for r, group in enumerate(groups):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(cases[group]):
                ax.axis("off")
                continue
            i = cases[group][c]
            img = images[i].astype(float) / 255.0
            cam = cam_fn(i)
            cam_up = F.interpolate(
                torch.from_numpy(cam)[None, None], size=(28, 28), mode="bilinear", align_corners=False
            )[0, 0].numpy()
            ax.imshow(img, cmap="gray")
            ax.imshow(cam_up, cmap="jet", alpha=0.45)
            pred_label = "pneumonia" if proba[i] >= thr else "normal"
            true_label = "pneumonia" if y_true[i] == 1 else "normal"
            ok = "OK" if pred_label == true_label else "ERR"
            ax.set_title(f"{ok} true={true_label}\npred={pred_label} p={proba[i]:.2f}", fontsize=8)
        axes[r][0].set_ylabel(group.replace("_", " "), fontsize=9)
    fig.suptitle("Grad-CAM: pneumonia evidence (red = high attribution)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_analysis(
    checkpoint: str | Path,
    out_dir: str | Path = "runs/workshop6",
    n_per_group: int = 4,
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
    x_test, y_test = arrays["test"], labels["test"]
    proba = predict_proba(model, x_test, dev)
    thr = 0.5
    cases = select_cases(y_test, proba, thr, n_per_group)

    # raw uint8 images for display
    raw, _ = mp.load_split_arrays("test", data_root)
    if cfg["preprocess"] == "clahe":
        raw = mp.apply_clahe(raw)

    cam_tool = GradCAM(model, model.features[6])  # last conv block: 128 x 7 x 7
    mean, std = ckpt["normalization"]["mean"], ckpt["normalization"]["std"]

    def cam_fn(i: int) -> np.ndarray:
        t = torch.from_numpy(x_test[i]).to(dev)  # (C,H,W)
        return cam_tool(t)

    out_dir = Path(out_dir)
    fig_path = make_figure(raw, cases, y_test, proba, thr, cam_fn,
                           out_dir / "gradcam_analysis.png")
    cam_tool.close()

    tp, fp_, tn, fn_ = confusion_counts(y_test, proba, thr)
    report = {
        "checkpoint": str(checkpoint),
        "threshold": thr,
        "confusion_at_threshold": {"TP": tp, "FP": fp_, "TN": tn, "FN": fn_},
        "cases": {g: [int(i) for i in idxs] for g, idxs in cases.items()},
        "case_details": {
            g: [{"index": int(i), "true": ("pneumonia" if y_test[i] == 1 else "normal"),
                 "p_pneumonia": round(float(proba[i]), 4)} for i in idxs]
            for g, idxs in cases.items()
        },
        "figure": str(fig_path),
    }
    (out_dir / "interpretability_report.json").write_text(json.dumps(report, indent=2))
    print(f"[interpretability] figure -> {fig_path}")
    for g, idxs in cases.items():
        for i in idxs:
            print(f"  {g:<18} idx={i:<4} true={'P' if y_test[i] else 'N'} p={proba[i]:.3f}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grad-CAM analysis for Workshop 6")
    ap.add_argument("--checkpoint", default="checkpoints/full_recipe_best.pt")
    ap.add_argument("--out-dir", default="runs/workshop6")
    ap.add_argument("--n-per-group", type=int, default=4)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cpu")
    run_analysis(**vars(ap.parse_args()))
