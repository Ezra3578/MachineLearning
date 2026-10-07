"""Data access for Workshop 3.

Thin wrapper that reuses the Workshop 2 pipeline (same 70/15/15 stratified
split, seed 42, and global normalization stats) so RL metrics stay comparable
with the supervised baselines. Falls back to a torchvision download if the
Workshop 1/2 files are not present.

Usage:
    from src.data import get_split_arrays, CLASS_NAMES, FMNIST_MEAN, FMNIST_STD
    images, labels = get_split_arrays("train")
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reused constants (duplicated here so Workshop_3 stays runnable standalone).
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

_W2_SRC = Path(__file__).resolve().parents[2] / "Workshop_2" / "src"
if str(_W2_SRC) not in sys.path:
    sys.path.insert(0, str(_W2_SRC))


def _w2_loader():
    """Return Workshop 2 (load_fashion_mnist, make_or_load_splits) if importable."""
    try:
        import importlib

        data_mod = importlib.import_module("data")
        return data_mod.load_fashion_mnist, data_mod.make_or_load_splits
    except Exception:
        return None, None


def get_full_dataset(data_root: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Return the full 70k (images uint8 N×28×28, labels) using W2 loader or torchvision."""
    load_fn, _ = _w2_loader()
    if load_fn is not None:
        try:
            return load_fn(data_root=data_root)
        except Exception:
            pass
    # Standalone fallback: torchvision download.
    from torchvision.datasets import FashionMNIST

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train = FashionMNIST(root=root, train=True, download=True)
    test_ds = FashionMNIST(root=root, train=False, download=True)
    images = np.concatenate([train.data.numpy(), test_ds.data.numpy()], axis=0).astype(np.uint8)
    labels = np.concatenate([train.targets.numpy(), test_ds.targets.numpy()], axis=0).astype(np.uint8)
    return images, labels


def get_split_indices(
    labels: np.ndarray, data_root: str | Path = "data", seed: int = SPLIT_SEED
) -> dict[str, list[int]]:
    """Return {train, val, test} index lists, reusing the cached W2 split when possible."""
    _, split_fn = _w2_loader()
    if split_fn is not None:
        try:
            return split_fn(labels, data_root=data_root, seed=seed)
        except Exception:
            pass
    # Standalone fallback: same stratified 70/15/15 protocol.
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, rest = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest, test_size=0.5, stratify=labels[rest], random_state=seed
    )
    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def get_split_arrays(
    split: str = "train", data_root: str | Path = "data", seed: int = SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for one split."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    images, labels = get_full_dataset(data_root=data_root)
    indices = get_split_indices(labels, data_root=data_root, seed=seed)[split]
    idx = np.array(indices, dtype=np.int64)
    return images[idx], labels[idx]
