"""Data access for Workshop 5 (GAN training in [-1, 1] and W2 augmentation retraining).

Reuses the Workshop 2 stratified 70/15/15 split (seed 42).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
FMNIST_MEAN = 0.2860402
FMNIST_STD = 0.3530242
SPLIT_SEED = 42

_W2_DIR = Path(__file__).resolve().parents[2] / "Workshop_2"
_W2_SRC = _W2_DIR / "src"
if str(_W2_SRC) not in sys.path:
    sys.path.insert(0, str(_W2_SRC))


def get_full_dataset(data_root: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Load the 70,000 Fashion-MNIST images (uint8 N×28×28) and labels (uint8 N)."""
    w1_raw = Path(__file__).resolve().parents[2] / "Workshop_1" / "code" / "data"
    try:
        from data import load_fashion_mnist as w2_load  # type: ignore

        return w2_load(data_root=data_root, local_raw_dir=w1_raw)
    except Exception:
        from torchvision.datasets import FashionMNIST

        root = Path(data_root) / "raw"
        root.mkdir(parents=True, exist_ok=True)
        train = FashionMNIST(root=root, train=True, download=True)
        test = FashionMNIST(root=root, train=False, download=True)
        images = np.concatenate([train.data.numpy(), test.data.numpy()], axis=0).astype(np.uint8)
        labels = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0).astype(np.uint8)
        return images, labels


def get_splits(labels: np.ndarray, data_root: str | Path = "data", seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    """Load cached 70/15/15 splits from Workshop_2/data/splits.json or local data_root."""
    w2_splits = _W2_DIR / "data" / "splits.json"
    try:
        from data import make_or_load_splits as w2_make_splits  # type: ignore

        if w2_splits.exists():
            return w2_make_splits(labels, data_root=w2_splits.parent, seed=seed)
        return w2_make_splits(labels, data_root=data_root, seed=seed)
    except Exception:
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(labels))
        train_idx, rest = train_test_split(indices, test_size=0.30, stratify=labels, random_state=seed)
        val_idx, test_idx = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=seed)
        return {
            "train": sorted(int(i) for i in train_idx),
            "val": sorted(int(i) for i in val_idx),
            "test": sorted(int(i) for i in test_idx),
        }


def build_gan_tensor_dataset(images: np.ndarray, labels: np.ndarray, indices: Sequence[int]) -> TensorDataset:
    """Vectorized TensorDataset scaling Fashion-MNIST images to [-1, 1] for Tanh Generator."""
    idx = np.asarray(indices, dtype=np.int64)
    x = torch.from_numpy(images[idx]).unsqueeze(1).float().div_(127.5).sub_(1.0)
    y = torch.from_numpy(labels[idx]).long()
    return TensorDataset(x, y)


def build_gan_dataloaders(
    data_root: str | Path = "data",
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = SPLIT_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders normalized to [-1, 1]."""
    images, labels = get_full_dataset(data_root=data_root)
    splits = get_splits(labels, data_root=data_root, seed=seed)

    g = torch.Generator().manual_seed(seed)
    train_ds = build_gan_tensor_dataset(images, labels, splits["train"])
    val_ds = build_gan_tensor_dataset(images, labels, splits["val"])
    test_ds = build_gan_tensor_dataset(images, labels, splits["test"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
