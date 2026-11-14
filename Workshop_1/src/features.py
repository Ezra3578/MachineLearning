"""
Feature scaling and PCA dimensionality reduction. All estimators are fitted
on the training subset only, to avoid data leakage.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeaturePipeline:
    """Standard scaling (z-score) followed by optional PCA."""

    def __init__(self, use_pca: bool = True, pca_variance: float = 0.95, seed: int = 42):
        """
        Initialize the scaler and the optional PCA transformer.

        Args:
            use_pca (bool): Whether to apply PCA after standard scaling.
            pca_variance (float): Variance ratio retained by PCA.
            seed (int): Random seed for reproducibility.

        Returns:
            None: This method initializes the instance in place.
        """
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance, random_state=seed) if use_pca else None
        self.is_fitted = False
        self.n_components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X_train: np.ndarray) -> "FeaturePipeline":
        """
        Fit the scaler and PCA on the training data, shape (N, 784) or (N, 28, 28).

        Args:
            X_train (np.ndarray): The features of training.

        Returns:
            FeaturePipeline: Instance of the trained object.

        """
        X_scaled = self.scaler.fit_transform(X_train.reshape(len(X_train), -1))

        if self.pca is not None:
            self.pca.fit(X_scaled)
            self.n_components_ = int(self.pca.n_components_)
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted transformations to new samples (val / test).

        Args:
            X (np.ndarray): The features that will be transformed (scaled).

        Returns:
            np.ndarray: The transformed features

        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        X_scaled = self.scaler.transform(X.reshape(len(X), -1))
        return self.pca.transform(X_scaled) if self.pca is not None else X_scaled

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Transforms the training features.

        Args:
            X_train (np.ndarray): The training features to transform.

        Returns:
            np.ndarray: The transformed features.
        """
        return self.fit(X_train).transform(X_train)

    def save(self, file_path: Path) -> None:
        """
        Saves the features on a folder.

        Args:
            file_path (Path): The path where the files will be saved.

        Returns:
            None
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "FeaturePipeline":

        """
        Loads the features from a folder.

        Args:
            file_path (Path): The path where the files will be loaded from.

        Returns:
            FeaturePipeline: Instance of the loaded features
        """
        return joblib.load(file_path)
