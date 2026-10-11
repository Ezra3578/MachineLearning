import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from .data_loader import CLASS_NAMES, load_raw_dataset, shift_image
    from .features import FeaturePipeline
except ImportError:
    from data_loader import CLASS_NAMES, load_raw_dataset, shift_image
    from features import FeaturePipeline


class FashionMNISTDataset:
    """Builds and saves a reproducible Fashion-MNIST preprocessing pipeline."""

    def __init__(
        self,
        *,
        seed: int = 42,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        augmentation_ratio: float = 0.20,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        scaling: str = "standard",
        reuse_splits: bool = True,
        output_dir: str | Path | None = None,
        splits_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.output_dir = Path(output_dir) if output_dir else root / "data" / "preprocessed"
        self.splits_path = Path(splits_path) if splits_path else root / "data" / "splits.json"

        self.seed = seed
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.augmentation_ratio = augmentation_ratio
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.scaling = scaling.lower().strip()
        self.reuse_splits = reuse_splits

        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("train_size + val_size + test_size must equal 1.0.")
        if min(train_size, val_size, test_size) <= 0:
            raise ValueError("All split sizes must be greater than zero.")
        if not 0 <= augmentation_ratio <= 1:
            raise ValueError("augmentation_ratio must be between 0 and 1.")
        if not 0 < pca_variance <= 1:
            raise ValueError("pca_variance must be in the interval (0, 1].")
        if self.scaling not in {"standard", "minmax", "none"}:
            raise ValueError("scaling must be 'standard', 'minmax', or 'none'.")
        if use_pca and self.scaling != "standard":
            raise ValueError("PCA requires scaling='standard'.")

        self.imputer = None
        self.scaler = None
        self.feature_pipeline = None
        self.arrays: dict[str, np.ndarray] = {}

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Combine the original train and test sets into 70,000 samples."""
        X_train, y_train, X_test, y_test = load_raw_dataset()
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test)).astype(np.int64)

        if X.shape[1:] != (28, 28):
            raise ValueError(f"Expected images of shape (N, 28, 28); got {X.shape}.")
        if not np.isin(y, np.arange(len(CLASS_NAMES))).all():
            raise ValueError("Labels must be integers from 0 to 9.")
        return X, y

    def split_data(self, y: np.ndarray) -> list[np.ndarray]:
        """Load saved split indices or create a stratified train/val/test split."""
        keys = ("train_indices", "val_indices", "test_indices")
        n = len(y)
        n_train = round(n * self.train_size)
        n_val = round(n * self.val_size)

        if self.reuse_splits and self.splits_path.exists():
            try:
                saved = json.loads(self.splits_path.read_text(encoding="utf-8"))
                splits = [np.asarray(saved[k], dtype=np.int64) for k in keys]
                all_idx = np.concatenate(splits)
                sizes = (n_train, n_val, n - n_train - n_val)
                if (
                    tuple(len(s) for s in splits) == sizes
                    and len(np.unique(all_idx)) == n
                    and all_idx.min() >= 0
                    and all_idx.max() < n
                ):
                    return splits
            except (OSError, KeyError, ValueError, TypeError):
                pass

        train_idx, temp_idx = train_test_split(
            np.arange(n), train_size=self.train_size, random_state=self.seed, stratify=y
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=self.val_size / (self.val_size + self.test_size),
            random_state=self.seed,
            stratify=y[temp_idx],
        )
        splits = [train_idx, val_idx, test_idx]

        self.splits_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: s.tolist() for k, s in zip(keys, splits)}
        payload["metadata"] = {
            "total": n,
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "seed": self.seed,
            "stratified": True,
        }
        self.splits_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return splits

    def augment(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append augmented copies of training images, balanced per class."""
        if self.augmentation_ratio == 0:
            return X, y

        rng = np.random.default_rng(self.seed)
        new_images, new_labels = [], []

        for label in np.unique(y):
            class_idx = np.flatnonzero(y == label)
            count = round(len(class_idx) * self.augmentation_ratio)
            for i in rng.choice(class_idx, size=count, replace=False):
                img = X[i]
                if rng.random() < 0.5:
                    img = np.fliplr(img)
                img = shift_image(img, int(rng.choice([-2, -1, 0, 1, 2])), int(rng.choice([-2, -1, 0, 1, 2])))
                new_images.append(img)
                new_labels.append(label)

        if not new_images:
            return X, y
        return (
            np.concatenate((X, np.stack(new_images).astype(X.dtype))),
            np.concatenate((y, np.asarray(new_labels, dtype=y.dtype))),
        )

    def run(self) -> dict[str, np.ndarray]:
        X, y = self.load_data()
        train_idx, val_idx, test_idx = self.split_data(y)

        X_train, y_train = self.augment(X[train_idx], y[train_idx])
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        images = {"train": X_train, "val": X_val, "test": X_test}
        flat = {k: v.reshape(len(v), -1).astype(np.float32) for k, v in images.items()}

        # Every estimator is fitted on the training split only
        self.imputer = SimpleImputer(strategy="median").fit(flat["train"])
        normalized = {k: self.imputer.transform(v).astype(np.float32) / 255.0 for k, v in flat.items()}

        if self.scaling == "standard":
            self.feature_pipeline = FeaturePipeline(self.use_pca, self.pca_variance, self.seed)
            self.feature_pipeline.fit(normalized["train"])
            scaler = self.feature_pipeline.scaler
        elif self.scaling == "minmax":
            self.scaler = scaler = MinMaxScaler().fit(normalized["train"])
        else:
            scaler = None

        scaled = {
            k: (scaler.transform(v) if scaler else v).astype(np.float32)
            for k, v in normalized.items()
        }

        self.arrays = {"y_train": y_train, "y_val": y_val, "y_test": y_test}
        for k in images:
            self.arrays[f"X_{k}_images"] = images[k]
            self.arrays[f"X_{k}_normalized"] = normalized[k]
            self.arrays[f"X_{k}_scaled"] = scaled[k]
            if self.use_pca:
                pca_features = self.feature_pipeline.pca.transform(scaled[k])
                self.arrays[f"X_{k}_pca"] = pca_features.astype(np.float32)

        self.save(y)
        print(f"Preprocessing done: train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
        return self.arrays

    def save(self, y_all: np.ndarray) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, array in self.arrays.items():
            np.save(self.output_dir / f"{name}.npy", array)

        pipeline = self.feature_pipeline
        metadata = {
            "dataset": "Fashion-MNIST",
            "seed": self.seed,
            "split_ratios": [self.train_size, self.val_size, self.test_size],
            "augmentation_ratio": self.augmentation_ratio,
            "scaling": self.scaling,
            "pca_variance_threshold": self.pca_variance if self.use_pca else None,
            "pca_components": int(pipeline.n_components_) if self.use_pca else None,
            "pca_explained_variance": (
                float(np.sum(pipeline.explained_variance_ratio_)) if self.use_pca else None
            ),
            "sample_counts": {
                "total_before_augmentation": len(y_all),
                "train_after_augmentation": len(self.arrays["y_train"]),
                "val": len(self.arrays["y_val"]),
                "test": len(self.arrays["y_test"]),
            },
            "class_names": CLASS_NAMES,
            "class_counts": {
                k: np.bincount(self.arrays[f"y_{k}"], minlength=10).tolist()
                for k in ("train", "val", "test")
            },
            "saved_arrays": sorted(self.arrays),
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        joblib.dump(
            {
                "imputer": self.imputer,
                "feature_pipeline": pipeline,
                "scaler": self.scaler,
                "config": {
                    "seed": self.seed,
                    "augmentation_ratio": self.augmentation_ratio,
                    "scaling": self.scaling,
                    "use_pca": self.use_pca,
                    "pca_variance": self.pca_variance,
                },
            },
            self.output_dir / "preprocessing_objects.joblib",
        )


if __name__ == "__main__":
    FashionMNISTDataset().run()