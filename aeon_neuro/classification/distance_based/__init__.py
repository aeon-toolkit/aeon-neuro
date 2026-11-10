"""Distance based EEG classifiers."""

__all__ = [
    "RiemannianKNNClassifier",
    "RiemannianMDMClassifier",
    "TemplateCCAClassifier",
    "TRCAClassifier",
]


from aeon_neuro.classification.distance_based._riemannian_knn import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)
from aeon_neuro.classification.distance_based._template import (
    TemplateCCAClassifier,
    TRCAClassifier,
)
