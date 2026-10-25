"""Medical data handling and preprocessing for Workshop 6 (PneumoniaMNIST).

Guide requirements covered here:
- modality-specific preprocessing: CLAHE contrast enhancement for chest X-rays
- normalization with dataset statistics (computed once, cached in data/stats.json)
- medical-appropriate augmentation: small rotations/translations and brightness
  jitter (implemented in src/train_medical.py as tensor ops). Horizontal flips
  are deliberately excluded: chest anatomy is lateralized (cardiac silhouette,
  aortic arch), so a flipped X-ray is not a valid X-ray (guide: "not generic
  augmentations that break medical meaningfulness").
- class imbalance reporting (class weighting itself lives in train_medical.py)

Data: PneumoniaMNIST (Yang et al., 2021), pediatric chest radiographs from
Guangzhou Women and Children's Medical Center, 28x28 grayscale, binary labels
(0 = normal, 1 = pneumonia).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from medmnist import PneumoniaMNIST

CLASS_NAMES = ["normal", "pneumonia"]
POS_CLASS = 1  # pneumonia is the clinically positive class


def load_split_arrays(split: str, root: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 (N,28,28), labels (N,)) for one PneumoniaMNIST split."""
    root = Path(root).resolve()
    ds = PneumoniaMNIST(split=split, download=True, root=str(root))
    return ds.imgs, ds.labels.squeeze().astype(np.int64)


def apply_clahe(images: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE contrast enhancement on uint8 grayscale X-rays (all splits).

    Chest radiographs span a wide intensity range with diagnostically relevant
    lung texture in low-contrast regions; CLAHE is the standard contrast
    stretching used before CNN training on X-ray data.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return np.stack([clahe.apply(img) for img in images])


def dataset_stats(images: np.ndarray) -> tuple[float, float]:
    """Mean/std of pixel values in [0,1] over the given images."""
    x = images.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())


def get_stats(root: str | Path = "data", use_clahe: bool = False) -> tuple[float, float]:
    """Normalization stats from the training split, cached in data/stats.json."""
    root = Path(root)
    cache = root / "stats.json"
    payload = {"raw": None, "clahe": None}
    if cache.exists():
        payload = json.loads(cache.read_text())
    key = "clahe" if use_clahe else "raw"
    if payload.get(key) is None:
        imgs, _ = load_split_arrays("train", root)
        if use_clahe:
            imgs = apply_clahe(imgs)
        payload[key] = dataset_stats(imgs)
        cache.write_text(json.dumps(payload, indent=2))
    return tuple(payload[key])


def make_tensors(
    images: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """uint8 (N,28,28) -> normalized float32 (N,1,28,28)."""
    x = images.astype(np.float32) / 255.0
    x = (x - mean) / std
    return x[:, None, :, :]


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[c]: int((labels == c).sum()) for c in (0, 1)}


def imbalance_report(root: str | Path = "data") -> dict:
    """Class balance per split, for the data-handling section of the report."""
    out = {}
    for split in ("train", "val", "test"):
        _, y = load_split_arrays(split, root)
        counts = class_counts(y)
        out[split] = {"n": len(y), **counts,
                      "pct_pneumonia": round(counts["pneumonia"] / len(y) * 100, 2)}
    return out


if __name__ == "__main__":
    print(json.dumps(imbalance_report(), indent=2))
    for use_clahe in (False, True):
        m, s = get_stats(use_clahe=use_clahe)
        print(f"stats (clahe={use_clahe}): mean={m:.4f} std={s:.4f}")
