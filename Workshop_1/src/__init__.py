"""
src/__init__.py
--------------
Módulos de preprocesamiento, extracción de características y modelos base
para el Taller 1 de Aprendizaje Automático (Fashion-MNIST).
"""

from .data_loader import CLASS_NAMES, get_or_create_splits, load_full_dataset, load_raw_dataset
from .features import FeaturePipeline

__all__ = [
    "CLASS_NAMES",
    "load_raw_dataset",
    "load_full_dataset",
    "get_or_create_splits",
    "FeaturePipeline",
]
