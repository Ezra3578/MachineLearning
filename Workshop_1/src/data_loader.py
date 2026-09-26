"""
src/data_loader.py
------------------
Módulo encargado de la lectura de archivos binarios comprimidos de Fashion-MNIST,
la generación de particiones reproducibles (Train/Val/Test) con serialización
en splits.json, y el aumento de datos (Data Augmentation) para entrenamiento.
"""

import gzip
import json
from pathlib import Path
import struct
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
        _ = struct.unpack(">I", f.read(4))[0]  # magic number
        n_images = struct.unpack(">I", f.read(4))[0]
        n_rows = struct.unpack(">I", f.read(4))[0]
        n_cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, n_rows, n_cols)
    return data


def _read_idx1_labels(file_path: Path) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        _ = struct.unpack(">I", f.read(4))[0]  # magic number
        n_labels = struct.unpack(">I", f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(n_labels)


def load_raw_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga los 4 archivos gzip binarios originales de Fashion-MNIST.
    Retorna (X_train_raw, y_train_raw, X_test_raw, y_test_raw).
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        primary = base_dir / "data" / "raw"
        fallback = base_dir / "code" / "data"
        fallback2 = base_dir / "notebooks" / "data"
        if primary.exists() and any(primary.glob("*.gz")):
            data_dir = primary
        elif fallback.exists() and any(fallback.glob("*.gz")):
            data_dir = fallback
        else:
            data_dir = fallback2

    train_img_p = data_dir / "train-images-idx3-ubyte.gz"
    train_lbl_p = data_dir / "train-labels-idx1-ubyte.gz"
    test_img_p = data_dir / "t10k-images-idx3-ubyte.gz"
    test_lbl_p = data_dir / "t10k-labels-idx1-ubyte.gz"

    for p in [train_img_p, train_lbl_p, test_img_p, test_lbl_p]:
        if not p.exists():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {p}")

    X_train_raw = _read_idx3_images(train_img_p)
    y_train_raw = _read_idx1_labels(train_lbl_p)
    X_test_raw = _read_idx3_images(test_img_p)
    y_test_raw = _read_idx1_labels(test_lbl_p)

    return X_train_raw, y_train_raw, X_test_raw, y_test_raw


def load_full_dataset(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Combina los 60,000 registros de entrenamiento y 10,000 de prueba originales
    en un arreglo unificado de 70,000 muestras para particionado controlado.
    """
    X_tr, y_tr, X_te, y_te = load_raw_dataset(data_dir)
    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    return X_all, y_all


def get_or_create_splits(
    y: np.ndarray,
    splits_file: Path | None = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """
    Genera o recupera particiones estratificadas train (70%), val (15%), test (15%).
    Garantiza reproducibilidad absoluta guardando o cargando desde splits.json.
    """
    if splits_file is None:
        base_dir = Path(__file__).resolve().parent.parent
        splits_file = base_dir / "data" / "splits.json"

    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        return splits

    total_samples = len(y)
    indices = np.arange(total_samples)

    # Primera particion: separar Train (70%) del resto (Val + Test = 30%)
    split_1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(split_1.split(indices, y))

    # Segunda particion: dividir el 30% restante equitativamente en Val (15%) y Test (15%)
    y_temp = y[temp_idx]
    split_2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_sub_idx, test_sub_idx = next(split_2.split(temp_idx, y_temp))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    splits = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "metadata": {
            "total_samples": total_samples,
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
    """
    Desplaza una imagen 2D (28x28) por shift_x y shift_y rellenando los bordes con 0.
    """
    shifted = np.zeros_like(img)
    rows, cols = img.shape

    src_r_start = max(0, -shift_y)
    src_r_end = min(rows, rows - shift_y)
    src_c_start = max(0, -shift_x)
    src_c_end = min(cols, cols - shift_x)

    dst_r_start = max(0, shift_y)
    dst_r_end = min(rows, rows + shift_y)
    dst_c_start = max(0, shift_x)
    dst_c_end = min(cols, cols + shift_x)

    shifted[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = img[src_r_start:src_r_end, src_c_start:src_c_end]
    return shifted


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Aplica volteo horizontal aleatorio y ligero desplazamiento espacial (1-2 px).
    Válido para prendas de vestir con simetría bilateral e invariancia de encuadre.
    """
    out = img.copy()
    # Volteo horizontal con probabilidad 0.5
    if rng.random() > 0.5:
        out = np.fliplr(out)

    # Desplazamiento de +/- 1 o 2 pixeles
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
    Genera un conjunto aumentado de entrenamiento muestreando una proporción de imágenes.
    Se aplica EXCLUSIVAMENTE al subconjunto de entrenamiento para evitar fuga de información.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(X_train_3d)
    n_augment = int(n_samples * augmentation_ratio)

    sample_indices = rng.choice(n_samples, size=n_augment, replace=False)
    aug_images = np.zeros((n_augment, 28, 28), dtype=np.uint8)

    for i, idx in enumerate(sample_indices):
        aug_images[i] = augment_image(X_train_3d[idx], rng)

    aug_labels = y_train[sample_indices]

    X_train_aug = np.concatenate([X_train_3d, aug_images], axis=0)
    y_train_aug = np.concatenate([y_train, aug_labels], axis=0)

    return X_train_aug, y_train_aug
