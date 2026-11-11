"""Distance based EEG classifiers."""

__all__ = [
    "RiemannianKNNClassifier",
    "RiemannianMDMClassifier",
    "TemplateCCAClassifier",
    "TRACE",
    "TRCAClassifier",
]


# TRACE (PR #296, copied unchanged) imports pyriemann.geometry, which is new in
# pyriemann 0.12; the shim maps it to pyriemann.utils on older versions.
from aeon_neuro.classification.distance_based import (  # noqa: F401, I001
    _pyriemann_geometry_shim,
)
from aeon_neuro.classification.distance_based._riemannian_knn import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)
from aeon_neuro.classification.distance_based._template import (
    TemplateCCAClassifier,
    TRCAClassifier,
)
from aeon_neuro.classification.distance_based._trace import TRACE
