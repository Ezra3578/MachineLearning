"""Model definitions for Workshop 2.

Two tracks:
- ``smallcnn``  — compact VGG-style CNN trained from scratch (≈ 140k parameters).
- ``resnet18``  — ImageNet-pretrained ResNet-18 for two-stage fine-tuning.

Run ``python -m src.model`` to print parameter counts and verify a forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# From-scratch CNN
# --------------------------------------------------------------------------- #
def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Compact CNN for 28×28 grayscale images.

    Three convolutional blocks (32 → 64 → 128 channels, two conv layers in the
    first two blocks), each followed by 2×2 max-pooling, then global average
    pooling, dropout and a linear classifier.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, head_dropout: float = 0.5) -> None:
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
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Transfer learning
# --------------------------------------------------------------------------- #
def build_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a fresh classifier head.

    Input adaptation (resize to 224×224, RGB) is handled by ``src.data``.
    """
    from torchvision import models
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        set_backbone_trainable(model, False)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the final classifier (two-stage fine-tuning)."""
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith("fc.")


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model: {name!r} (expected 'smallcnn' or 'resnet18')")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    for name in ("smallcnn", "resnet18"):
        model = build_model(name, pretrained=False)
        trainable, total = count_parameters(model)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(4, 1 if name == "smallcnn" else 3, 28 if name == "smallcnn" else 224, 28 if name == "smallcnn" else 224))
        print(f"{name:>8}: trainable={trainable:,} total={total:,} | forward output {tuple(out.shape)}")
        assert out.shape == (4, NUM_CLASSES)
    print("forward-pass check passed ✓")
