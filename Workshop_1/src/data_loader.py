"""
Reads the compressed Fashion-MNIST binary files, builds reproducible
train/val/test splits saved to splits.json, and provides data augmentation.
"""

import gzip
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

CLASS_NAMES = [
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


def _read_idx3_images(file_path: Path) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        _, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, n_rows, n_cols)


def _read_idx1_labels(file_path: Path) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        _, n_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(n_labels)


def load_raw_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four original gzip files. Returns (X_train, y_train, X_test, y_test)."""
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        primary = base_dir / "data" / "raw"
        fallback = base_dir / "code" / "data"
        if primary.exists() and any(primary.glob("*.gz")):
            data_dir = primary
        elif fallback.exists() and any(fallback.glob("*.gz")):
            data_dir = fallback
        else:
            data_dir = base_dir / "notebooks" / "data"

    paths = [
        data_dir / "train-images-idx3-ubyte.gz",
        data_dir / "train-labels-idx1-ubyte.gz",
        data_dir / "t10k-images-idx3-ubyte.gz",
        data_dir / "t10k-labels-idx1-ubyte.gz",
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    return (
        _read_idx3_images(paths[0]),
        _read_idx1_labels(paths[1]),
        _read_idx3_images(paths[2]),
        _read_idx1_labels(paths[3]),
    )


def load_full_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Combine the original 60,000 train and 10,000 test samples into one array."""
    X_tr, y_tr, X_te, y_te = load_raw_dataset(data_dir)
    return np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te])


def get_or_create_splits(
    y: np.ndarray,
    splits_file: Path | None = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """Load the stratified train/val/test split from splits.json, or create and save it."""
    if splits_file is None:
        splits_file = Path(__file__).resolve().parent.parent / "data" / "splits.json"

    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            return json.load(f)

    indices = np.arange(len(y))

    train_idx, temp_idx = next(
        StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed).split(indices, y)
    )

    # Split the remaining 30% evenly into validation and test
    val_sub, test_sub = next(
        StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed).split(temp_idx, y[temp_idx])
    )
    val_idx, test_idx = temp_idx[val_sub], temp_idx[test_sub]

    splits = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "metadata": {
            "total_samples": len(y),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "proportions": [train_size, val_size, test_size],
            "seed": seed,
        },
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


def shift_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Shift a 2D image by (shift_x, shift_y) pixels, filling the borders with 0."""
    shifted = np.zeros_like(img)
    rows, cols = img.shape

    src_r_start, src_r_end = max(0, -shift_y), min(rows, rows - shift_y)
    src_c_start, src_c_end = max(0, -shift_x), min(cols, cols - shift_x)
    dst_r_start, dst_r_end = max(0, shift_y), min(rows, rows + shift_y)
    dst_c_start, dst_c_end = max(0, shift_x), min(cols, cols + shift_x)

    shifted[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = img[src_r_start:src_r_end, src_c_start:src_c_end]
    return shifted


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random horizontal flip plus a shift of up to 2 pixels in each direction."""
    out = img.copy()
    if rng.random() > 0.5:
        out = np.fliplr(out)

    shift_x = int(rng.choice([-2, -1, 0, 1, 2]))
    shift_y = int(rng.choice([-2, -1, 0, 1, 2]))
    if shift_x != 0 or shift_y != 0:
        out = shift_image(out, shift_x, shift_y)

    return out


def generate_augmented_training_batch(
    X_train_3d: np.ndarray,
    y_train: np.ndarray,
    augmentation_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Append augmented copies of a random fraction of the training images.
    Only apply this to the training subset to avoid data leakage.
    """
    rng = np.random.default_rng(seed)
    n_augment = int(len(X_train_3d) * augmentation_ratio)

    sample_indices = rng.choice(len(X_train_3d), size=n_augment, replace=False)
    aug_images = np.zeros((n_augment, 28, 28), dtype=np.uint8)
    for i, idx in enumerate(sample_indices):
        aug_images[i] = augment_image(X_train_3d[idx], rng)

    X_train_aug = np.concatenate([X_train_3d, aug_images], axis=0)
    y_train_aug = np.concatenate([y_train, y_train[sample_indices]], axis=0)
    return X_train_aug, y_train_aug