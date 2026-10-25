"""Domain transfer implementation for Workshop 6.

Transfers the Workshop 2 SmallCNN (trained on Fashion-MNIST, 10 classes,
1x28x28 grayscale) to PneumoniaMNIST chest X-ray classification (binary).

Transfer modes:
- "scratch":  same architecture, random initialization (no transfer; ablation
              baseline).
- "frozen":   feature reuse. Convolutional features loaded from the W2
              checkpoint and frozen; only a new binary head is trained.
- "finetune": hybrid. Convolutional features loaded from W2, everything
              trainable, new binary head.

The W2 checkpoint (../Workshop_2/checkpoints/model_best.pt) stores the 10-class
classifier head, which we drop: the head is task-specific, the conv backbone is
the transferable component.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

W2_CHECKPOINT = Path(__file__).resolve().parents[2] / "Workshop_2" / "checkpoints" / "model_best.pt"


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Same architecture as Workshop 2 src/model.py (state_dict compatible).

    Keys must match exactly for checkpoint loading; copied verbatim from W2.
    """

    def __init__(self, num_classes: int = 10, head_dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 32),
            nn.MaxPool2d(2),          # 28 -> 14
            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),          # 14 -> 7
            _conv_block(64, 128),
            nn.MaxPool2d(2),          # 7 -> 3
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_transferred_model(
    mode: str = "finetune",
    w2_checkpoint: str | Path = W2_CHECKPOINT,
    head_dropout: float = 0.5,
    seed: int = 42,
) -> tuple[SmallCNN, dict]:
    """Return (model, transfer_report) for the requested mode.

    Binary task: the head outputs one logit (pneumonia score, used with
    BCEWithLogitsLoss).
    """
    if mode not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"mode must be scratch|frozen|finetune, got {mode!r}")

    torch.manual_seed(seed)
    model = SmallCNN(num_classes=1, head_dropout=head_dropout)
    report = {"mode": mode, "w2_checkpoint": str(w2_checkpoint), "loaded": False}

    if mode == "scratch":
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report.update({"trainable_params": trainable, "frozen_params": 0})
        return model, report

    ckpt = torch.load(w2_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    # Drop the task-specific head; keep the conv backbone.
    backbone = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    report.update(
        {
            "loaded": True,
            "source_model": ckpt.get("model_name", "smallcnn"),
            "source_classes": ckpt.get("classes"),
            "transferred_layers": len(backbone),
            "reinitialized": [m for m in missing],
        }
    )

    if mode == "frozen":
        for p in model.features.parameters():
            p.requires_grad = False
    # BatchNorm running stats: in frozen mode keep W2 statistics; in finetune
    # mode they update during training.
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report.update({"trainable_params": trainable, "frozen_params": frozen})
    return model, report


if __name__ == "__main__":
    for mode in ("scratch", "frozen", "finetune"):
        model, report = build_transferred_model(mode)
        x = torch.zeros(2, 1, 28, 28)
        assert model(x).shape == (2, 1)
        print(mode, {k: v for k, v in report.items() if k != "w2_checkpoint"})
