"""Grad-CAM analysis for Workshop 2 (Selvaraju et al., 2016 — arXiv:1610.02391).

Generates a figure with correctly classified and misclassified test examples,
overlaying the gradient-weighted class activation map that drives each prediction.

Usage:
    python -m src.gradcam --checkpoint checkpoints/model_best.pt
    python -m src.gradcam --n-correct 4 --n-missed 4 --out runs/gradcam_examples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import CLASS_NAMES, FMNIST_MEAN, FMNIST_STD, build_dataloaders
from src.model import build_model


# --------------------------------------------------------------------------- #
# Grad-CAM core
# --------------------------------------------------------------------------- #
class GradCAM:
    """Gradient-weighted Class Activation Mapping via forward/backward hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._handle_fwd = target_layer.register_forward_hook(self._on_forward)
        self._handle_bwd = target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module, inputs, output) -> None:  # noqa: ARG002
        self._activation = output

    def _on_backward(self, module, grad_input, grad_output) -> None:  # noqa: ARG002
        self._gradient = grad_output[0]

    def __call__(self, image: torch.Tensor, class_idx: int | None = None):
        """Return (cam, class_idx, probabilities) for a single image tensor (C,H,W)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image.unsqueeze(0))
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        logits[0, class_idx].backward()

        # α_k = global-average-pooled gradients; CAM = ReLU(Σ α_k · A^k)
        weights = self._gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activation).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        probs = torch.softmax(logits.detach(), dim=1)[0]
        return cam.detach().cpu().numpy(), class_idx, probs.cpu()

    def close(self) -> None:
        self._handle_fwd.remove()
        self._handle_bwd.remove()


def target_layer_for(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    """Last convolutional stage that still carries spatial information."""
    if model_name == "smallcnn":
        return model.features[6]   # last conv block (7×7 feature maps)
    if model_name == "resnet18":
        return model.layer4[-1]    # last residual block (7×7 at 224 input)
    raise ValueError(f"no Grad-CAM target defined for {model_name!r}")


# --------------------------------------------------------------------------- #
# Example selection
# --------------------------------------------------------------------------- #
def _denormalize(image: torch.Tensor, model_name: str) -> np.ndarray:
    """Convert a normalized (C,H,W) tensor back to a displayable grayscale array."""
    if model_name == "resnet18":
        gray = image.mean(dim=0).numpy()
        mean, std = 0.449, 0.226  # ImageNet grayscale approx, display only
    else:
        gray = image[0].numpy()
        mean, std = FMNIST_MEAN, FMNIST_STD
    return np.clip(gray * std + mean, 0.0, 1.0)


def select_examples(model, loader, device, n_correct: int = 3, n_missed: int = 3,
                    miss_classes: tuple[str, ...] = ("Shirt", "T-shirt/top")):
    """Pick diverse correct examples + the most confident confusions of `miss_classes`."""
    miss_ids = {CLASS_NAMES.index(c) for c in miss_classes}
    correct_pool: list[tuple] = []
    missed_pool: list[tuple] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            for i in range(images.size(0)):
                true, pred = int(labels[i]), int(preds[i])
                entry = (images[i].clone(), true, pred, float(probs[i, pred]))
                if true == pred:
                    correct_pool.append(entry)
                elif true in miss_ids:
                    missed_pool.append(entry)
            if len(correct_pool) > 400 and len(missed_pool) > 80:
                break

    picked_correct: list[tuple] = []
    for class_name in ("Shirt", "T-shirt/top", "Pullover", "Sneaker", "Bag", "Ankle boot"):
        class_id = CLASS_NAMES.index(class_name)
        for entry in correct_pool:
            if entry[1] == class_id:
                picked_correct.append(entry)
                break
        if len(picked_correct) >= n_correct:
            break

    missed_pool.sort(key=lambda e: -e[3])  # most confident errors first
    return picked_correct[:n_correct], missed_pool[:n_missed]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(examples: list[tuple], cams: list[np.ndarray], titles: list[str],
                model_name: str, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(examples)
    fig, axes = plt.subplots(2, n * 2, figsize=(2.1 * n * 2, 4.6))
    for k, (entry, cam, title) in enumerate(zip(examples, cams, titles)):
        image, true, pred, prob = entry
        gray = _denormalize(image, model_name)
        h, w = gray.shape
        cam_up = F.interpolate(
            torch.from_numpy(cam)[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0].numpy()

        axes[0, 2 * k].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[1, 2 * k].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[0, 2 * k + 1].imshow(cam_up, cmap="jet")
        axes[1, 2 * k + 1].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[1, 2 * k + 1].imshow(cam_up, cmap="jet", alpha=0.45)

        for row in (0, 1):
            axes[row, 2 * k].set_xticks([])
            axes[row, 2 * k].set_yticks([])
            axes[row, 2 * k + 1].set_xticks([])
            axes[row, 2 * k + 1].set_yticks([])
        axes[0, 2 * k].set_title(f"{CLASS_NAMES[true]} → {CLASS_NAMES[pred]} ({prob:.2f})", fontsize=8)
        axes[0, 2 * k + 1].set_title("Grad-CAM", fontsize=8)

    axes[0, 0].set_ylabel("correct", fontsize=9)
    axes[1, 0].set_ylabel("misclassified", fontsize=9)
    fig.suptitle(f"Grad-CAM — {model_name} (Fashion-MNIST test)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_analysis(
    checkpoint: str | Path = "checkpoints/model_best.pt",
    out_path: str | Path = "runs/gradcam_examples.png",
    device: str = "cpu",
    n_correct: int = 3,
    n_missed: int = 3,
    data_root: str | Path = "data",
) -> Path:
    """Full Grad-CAM analysis: load checkpoint, pick examples, save figure."""
    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint_data["model_name"]

    model = build_model(model_name, num_classes=len(checkpoint_data.get("classes", CLASS_NAMES)), pretrained=False)
    model.load_state_dict(checkpoint_data["state_dict"])
    model.to(torch_device).eval()

    _, _, test_loader = build_dataloaders(model_name, data_root=data_root, batch_size=256, num_workers=0, augment=False)
    correct, missed = select_examples(model, test_loader, torch_device, n_correct, n_missed)
    examples = correct + missed

    cam_tool = GradCAM(model, target_layer_for(model_name, model))
    cams: list[np.ndarray] = []
    titles: list[str] = []
    print(f"[gradcam] selected examples ({model_name}):")
    for image, true, pred, prob in examples:
        cam, _, _ = cam_tool(image.to(torch_device))
        cams.append(cam)
        titles.append(f"{CLASS_NAMES[true]} → {CLASS_NAMES[pred]} ({prob:.2f})")
        marker = "ok" if true == pred else "XX"
        print(f"  {marker} true={CLASS_NAMES[true]:<12} pred={CLASS_NAMES[pred]:<12} p={prob:.3f}")
    cam_tool.close()

    figure_path = make_figure(examples, cams, titles, model_name, Path(out_path))
    print(f"[gradcam] figure saved to {figure_path}")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM examples for a Workshop 2 checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--out", type=str, default="runs/gradcam_examples.png")
    parser.add_argument("--n-correct", type=int, default=3)
    parser.add_argument("--n-missed", type=int, default=3)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu", help="CPU recommended (small analysis; avoids MPS hook quirks)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        checkpoint=args.checkpoint, out_path=args.out, device=args.device,
        n_correct=args.n_correct, n_missed=args.n_missed, data_root=args.data_root,
    )
