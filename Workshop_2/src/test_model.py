"""Checkpoint verification script for Workshop 2 (professor requirement).

Loads ``checkpoints/model_best.pt``, rebuilds the architecture, and runs inference
on a small batch of test images to confirm the saved weights produce valid
predictions with the expected output shape.

Usage:
    python -m src.test_model --checkpoint checkpoints/model_best.pt --samples 8
"""

from __future__ import annotations

import argparse

import torch

from src.data import build_dataloaders
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a Workshop 2 checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "directml":
        import torch_directml

        return torch_directml.device()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml
    except ImportError:
        return torch.device("cpu")
    return torch_directml.device() if torch_directml.is_available() else torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert "state_dict" in checkpoint, "checkpoint missing 'state_dict'"
    model_name = checkpoint["model_name"]
    classes = checkpoint.get("classes", [str(i) for i in range(10)])

    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not missing and not unexpected, f"state_dict mismatch: missing={missing} unexpected={unexpected}"
    model.to(device).eval()
    print(f"[test_model] loaded {args.checkpoint} (model={model_name}, epoch={checkpoint.get('epoch')})")
    if "val_metrics" in checkpoint:
        vm = checkpoint["val_metrics"]
        print(f"[test_model] recorded val metrics: acc={vm['accuracy']:.4f} f1_macro={vm['f1_macro']:.4f}")

    _, _, test_loader = build_dataloaders(model_name, data_root=args.data_root, batch_size=max(args.samples, 1), num_workers=0, augment=False)
    images, labels = next(iter(test_loader))
    images = images[: args.samples]

    with torch.no_grad():
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu()

    assert logits.shape == (images.size(0), len(classes)), f"unexpected output shape {tuple(logits.shape)}"
    preds = probs.argmax(dim=1)

    print(f"[test_model] inference on {images.size(0)} samples - output shape {tuple(logits.shape)} [OK]")
    print(f"[test_model] {'#':>2}  {'true':<12} {'pred':<12} {'p(pred)':>8}  ok")
    for i in range(images.size(0)):
        true_name = classes[int(labels[i])]
        pred_name = classes[int(preds[i])]
        mark = "ok" if int(preds[i]) == int(labels[i]) else "XX"
        print(f"[test_model] {i:>2}  {true_name:<12} {pred_name:<12} {probs[i, preds[i]]:>8.4f}  {mark}")

    print("[test_model] checkpoint verification passed [OK]")


if __name__ == "__main__":
    main()
