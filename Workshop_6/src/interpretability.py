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
from src.device import pick_device
from src.medical_metrics import confusion_counts
from src.plotting import save_figure
from src.train_medical import build_arrays, predict_proba
from src.transfer import SmallCNN


def region_masks(size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Lung-field and image-border masks on the 28x28 grid.

    The lung box covers both hemithoraces from the apices to the costophrenic
    angles (rows 6-21, columns 3-24 here), leaving the mediastinum and the
    corners out. The border ring is the outer three pixels, where a radiograph
    has no lung tissue.
    """
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    left = (rows >= 6) & (rows <= 21) & (cols >= 3) & (cols <= 10)
    right = (rows >= 6) & (rows <= 21) & (cols >= 18) & (cols <= 24)
    border = (rows < 3) | (rows > size - 4) | (cols < 3) | (cols > size - 4)
    return left | right, border


def attribution_by_group(cams: np.ndarray, y_true, proba, thr: float = 0.5) -> dict:
    """Share of attribution energy on the lung fields and on the border.

    Energy fractions are computed per image over the upsampled CAM, then
    averaged within each outcome group. `peak_on_border` counts how often the
    maximum of the map falls on the border ring.
    """
    lung, border = region_masks(cams.shape[-1])
    pred = (proba >= thr).astype(int)
    groups = {
        "correct_pneumonia": (y_true == 1) & (pred == 1),
        "correct_normal": (y_true == 0) & (pred == 0),
        "false_negative": (y_true == 1) & (pred == 0),
        "false_positive": (y_true == 0) & (pred == 1),
    }
    out = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            continue
        c = cams[mask]
        total = c.sum(axis=(1, 2)) + 1e-9
        peaks = np.array([np.unravel_index(np.argmax(c[i]), c.shape[-2:]) for i in range(n)])
        out[name] = {
            "n": n,
            "lung_energy_share": round(float((c[:, lung].sum(axis=1) / total).mean()), 4),
            "border_energy_share": round(float((c[:, border].sum(axis=1) / total).mean()), 4),
            "peak_on_border_share": round(float(border[peaks[:, 0], peaks[:, 1]].mean()), 4),
        }
    return out


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
        return cam.detach().cpu().numpy()

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
    save_figure(fig, out_path, dpi=200)
    plt.close(fig)
    return out_path


def failure_examples(model, raw, y, scores, cams, stats, device) -> dict:
    """Return representative test cases for the deployment discussion."""
    # A true positive at 0.5 that would be missed at a stricter threshold.
    stricter = np.flatnonzero((y == 1) & (scores >= 0.5) & (scores < 0.8))
    # A true negative whose positive-class explanation peaks at the edge.
    _, border = region_masks(cams.shape[-1])
    peak = cams.reshape(len(cams), -1).argmax(axis=1)
    has_attribution = cams.max(axis=(1, 2)) > 0
    corner = np.flatnonzero((y == 0) & (scores < 0.5) & has_attribution & border.ravel()[peak])
    examples = {}
    if stricter.size:
        i = int(stricter[0])
        examples["threshold_sensitive_positive"] = {"index": i, "score": float(scores[i])}
    if corner.size:
        i = int(corner[np.argmin(scores[corner])])
        examples["normal_with_border_attribution"] = {"index": i, "score": float(scores[i])}
    if stats["preprocess"] == "none":
        # Simulate a pipeline mismatch without changing the checkpoint or its normalization.
        altered = mp.make_tensors(mp.apply_clahe(raw), stats["mean"], stats["std"])
        shifted = predict_proba(model, altered, device)
        correct = (scores >= 0.5) == y
        flips = np.flatnonzero(correct & ((shifted >= 0.5) != y))
        if flips.size:
            i = int(flips[np.argmax(np.abs(shifted[flips] - scores[flips]))])
            examples["unexpected_clahe_error"] = {
                "index": i, "true": mp.CLASS_NAMES[int(y[i])],
                "score": float(scores[i]), "altered_score": float(shifted[i]),
                "new_errors": int(flips.size),
            }
    return examples


def run_analysis(
    checkpoint: str | Path,
    out_dir: str | Path = "runs/workshop6",
    n_per_group: int = 4,
    data_root: str = "data",
    device: str = "auto",
) -> dict:
    dev = pick_device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SmallCNN(num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    arrays, labels, stats = build_arrays(cfg["preprocess"], data_root)
    x_test, y_test = arrays["test"], labels["test"]
    proba = predict_proba(model, x_test, dev)
    thr = 0.5
    cases = select_cases(y_test, proba, thr, n_per_group)

    # raw uint8 images for display
    raw, _ = mp.load_split_arrays("test", data_root)
    if cfg["preprocess"] == "clahe":
        raw = mp.apply_clahe(raw)

    cam_tool = GradCAM(model, model.features[6])  # last conv block: 128 x 7 x 7

    def cam_fn(i: int) -> np.ndarray:
        t = torch.from_numpy(x_test[i]).to(dev)  # (C,H,W)
        return cam_tool(t)

    # attribution over the whole test split, for the regional summary
    cams = np.zeros((len(y_test), 28, 28), dtype=np.float32)
    for i in range(len(y_test)):
        cam = cam_fn(i)
        cam = torch.from_numpy(cam)[None, None]
        cam = torch.nn.functional.interpolate(cam, size=(28, 28), mode="bilinear",
                                              align_corners=False)
        cams[i] = cam[0, 0].numpy()
    regions = attribution_by_group(cams, y_test, proba, thr)

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
        "attribution_by_group": regions,
        "failure_examples": failure_examples(model, raw, y_test, proba, cams, stats, dev),
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
    print("[interpretability] attribution by group (energy share)")
    for g, stats in regions.items():
        print(f"  {g:<18} n={stats['n']:<4} lung={stats['lung_energy_share']:.3f} "
              f"border={stats['border_energy_share']:.3f} "
              f"peak_on_border={stats['peak_on_border_share']:.3f}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grad-CAM analysis for Workshop 6")
    ap.add_argument("--checkpoint", default="checkpoints/full_recipe_best.pt")
    ap.add_argument("--out-dir", default="runs/workshop6")
    ap.add_argument("--n-per-group", type=int, default=4)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps|privateuseone")
    run_analysis(**vars(ap.parse_args()))
