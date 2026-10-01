"""Fashion-MNIST data pipeline for Workshop 2.

Loads the original IDX files (preferring the Workshop 1 copy at
``../Workshop_1/code/data``), falls back to the torchvision download, and builds a
reproducible stratified 70/15/15 split (seed 42) over the 70,000 images — the same
protocol as the Workshop 1 team implementation, so metrics stay comparable.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Global Fashion-MNIST statistics (Workshop 1 EDA: 72.94/255 and 90.02/255).
FMNIST_MEAN = 0.2860402
FMNIST_STD = 0.3530242

# ImageNet statistics for the pretrained ResNet-18 variant.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

_IDX_IMAGE_MAGIC = 2051
_IDX_LABEL_MAGIC = 2049


# --------------------------------------------------------------------------- #
# Raw IDX parsing (same approach as Workshop 1)
# --------------------------------------------------------------------------- #
def load_idx_images(path: str | Path) -> np.ndarray:
    """Read an IDX3 image file and return a ``(N, 28, 28)`` uint8 array."""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        magic, n, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != _IDX_IMAGE_MAGIC:
            raise ValueError(f"{path}: bad magic number {magic}")
        buffer = fh.read()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(n, rows, cols)


def load_idx_labels(path: str | Path) -> np.ndarray:
    """Read an IDX1 label file and return an ``(N,)`` uint8 array."""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        magic, n = struct.unpack(">II", fh.read(8))
        if magic != _IDX_LABEL_MAGIC:
            raise ValueError(f"{path}: bad magic number {magic}")
        buffer = fh.read()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(n)


def _load_local_raw(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four IDX files from ``raw_dir`` and concatenate the splits."""
    train_x = load_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_y = load_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_x = load_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_y = load_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")
    return train_x, train_y, test_x, test_y


def _load_torchvision(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: let torchvision download (and cache) the dataset."""
    from torchvision.datasets import FashionMNIST

    data_root.mkdir(parents=True, exist_ok=True)
    train = FashionMNIST(root=data_root, train=True, download=True)
    test = FashionMNIST(root=data_root, train=False, download=True)
    return (
        train.data.numpy().astype(np.uint8),
        train.targets.numpy().astype(np.uint8),
        test.data.numpy().astype(np.uint8),
        test.targets.numpy().astype(np.uint8),
    )


def load_fashion_mnist(
    data_root: str | Path = "data",
    local_raw_dir: str | Path | None = "../Workshop_1/code/data",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full 70,000-image dataset as ``(images, labels)``.

    Tries the local Workshop 1 raw files first, then falls back to a torchvision
    download under ``data_root/raw``.
    """
    data_root = Path(data_root)
    if local_raw_dir is not None:
        raw = Path(local_raw_dir)
        if all(
            (raw / name).exists()
            for name in (
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            )
        ):
            train_x, train_y, test_x, test_y = _load_local_raw(raw)
        else:
            print(f"[data] local raw files not found at {raw} — downloading via torchvision")
            train_x, train_y, test_x, test_y = _load_torchvision(data_root / "raw")
    else:
        train_x, train_y, test_x, test_y = _load_torchvision(data_root / "raw")

    images = np.concatenate([train_x, test_x], axis=0)
    labels = np.concatenate([train_y, test_y], axis=0)
    return images, labels


# --------------------------------------------------------------------------- #
# Reproducible stratification
# --------------------------------------------------------------------------- #
def make_or_load_splits(
    labels: np.ndarray,
    data_root: str | Path = "data",
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[int]]:
    """Create (or reload) the stratified 70/15/15 split index lists."""
    splits_path = Path(data_root) / "splits.json"
    if splits_path.exists():
        payload = json.loads(splits_path.read_text())
        splits = payload.get("indices", {})
        total = sum(len(v) for v in splits.values())
        if {"train", "val", "test"} <= set(splits) and total == len(labels):
            print(f"[data] reusing cached split from {splits_path}")
            return splits
        print(f"[data] cached split unusable ({total} != {len(labels)}) — rebuilding")

    indices = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        indices, test_size=val_fraction + test_fraction, stratify=labels, random_state=seed
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=relative_test, stratify=labels[rest_idx], random_state=seed
    )

    splits = {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(
            {
                "config": {
                    "seed": seed,
                    "val_fraction": val_fraction,
                    "test_fraction": test_fraction,
                    "total_samples": int(len(labels)),
                },
                "split_sizes": {k: len(v) for k, v in splits.items()},
                "indices": splits,
            },
            indent=2,
        )
    )
    print(
        f"[data] split created: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    return splits


# --------------------------------------------------------------------------- #
# Dataset and loaders
# --------------------------------------------------------------------------- #
class FashionMNISTDataset(Dataset):
    """Simple map-style dataset over uint8 image arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        idx = self.indices[item]
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform is not None:
            from PIL import Image

            sample = Image.fromarray(image, mode="L")
            image_tensor = self.transform(sample)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        return image_tensor, label


def build_transforms(model_name: str, train: bool) -> transforms.Compose:
    """Augmentation (train only) + normalization per model family."""
    base: list = []
    if model_name == "resnet18":
        base.append(transforms.Grayscale(num_output_channels=3))
        base.append(transforms.Resize(224))
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = (FMNIST_MEAN,), (FMNIST_STD,)

    if train:
        base.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )
    base.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(base)


def build_dataloaders(
    model_name: str,
    data_root: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders with the shared 70/15/15 split."""
    images, labels = load_fashion_mnist(data_root=data_root)
    splits = make_or_load_splits(labels, data_root=data_root, seed=seed)

    generator = torch.Generator().manual_seed(seed)
    train_ds = FashionMNISTDataset(
        images, labels, splits["train"], build_transforms(model_name, train=True) if augment else build_transforms(model_name, train=False)
    )
    val_ds = FashionMNISTDataset(images, labels, splits["val"], build_transforms(model_name, train=False))
    test_ds = FashionMNISTDataset(images, labels, splits["test"], build_transforms(model_name, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=generator, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader
