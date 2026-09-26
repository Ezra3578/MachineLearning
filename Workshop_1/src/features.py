"""
src/features.py
---------------
Módulo para el escalado de características y reducción de dimensionalidad con PCA.
Garantiza un pipeline riguroso libre de fuga de información (data leakage),
ajustando todos los estimadores únicamente sobre el subconjunto de entrenamiento.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeaturePipeline:
    """
    Pipeline de ingeniería de características que combina escalado estándar
    (z-score) y reducción de dimensionalidad mediante Análisis de Componentes
    Principales (PCA).
    """

    def __init__(self, use_pca: bool = True, pca_variance: float = 0.95, seed: int = 42):
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
        Ajusta el escalador y PCA exclusivamente sobre la matriz de entrenamiento.
        X_train puede tener forma (N, 784) o (N, 28, 28).
        """
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        # Ajuste de escalador
        X_scaled = self.scaler.fit_transform(X_train)

        # Ajuste de PCA si está habilitado
        if self.use_pca and self.pca is not None:
            self.pca.fit(X_scaled)
            self.n_components_ = self.pca.n_components_
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica las transformaciones previamente calculadas a nuevas muestras (Val / Test).
        """
        if not self.is_fitted:
            raise RuntimeError("El pipeline debe ser ajustado mediante fit() antes de invocar transform().")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X)

        if self.use_pca and self.pca is not None:
            return self.pca.transform(X_scaled)

        return X_scaled

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Ajusta y transforma de forma atómica sobre el conjunto de entrenamiento.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, file_path: Path):
        """Serializa el pipeline ajustado a disco."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "FeaturePipeline":
        """Carga un pipeline ajustado desde disco."""
        return joblib.load(file_path)
