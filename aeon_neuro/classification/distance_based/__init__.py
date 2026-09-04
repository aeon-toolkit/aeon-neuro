"""Distance based EEG classifiers."""

__all__ = [
    "RiemannianKNNClassifier",
    "RiemannianMDMClassifier",
    "RiemannInterval"
]


from aeon_neuro.classification.distance_based._riemannian_knn import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)
from aeon_neuro.classification.distance_based.riemann_interval import RiemannInterval