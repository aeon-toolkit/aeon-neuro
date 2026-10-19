"""Regression tests for estimator test parameters."""

import pytest

from aeon_neuro.classification.deep_learning import (
    DeepConvNetClassifier,
    EEGNetClassifier,
)
from aeon_neuro.classification.distance_based._riemannian_knn import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
    _BaseRiemannianCovarianceClassifier,
)
from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_selection._riemannian import (
    Riemannian,
)
from aeon_neuro.transformations.collection.embedding._umap import UMAP


@pytest.mark.parametrize(
    "estimator_cls, expected_key",
    [
        (EEGNetClassifier, "n_epochs"),
        (DeepConvNetClassifier, "n_epochs"),
        (_BaseRiemannianCovarianceClassifier, "covariance_estimator"),
        (RiemannianMDMClassifier, "metric"),
        (RiemannianKNNClassifier, "n_neighbors"),
        (UMAP, "n_neighbors"),
        (Riemannian, "proportion"),
        (CommonSpatialPatterns, "n_components"),
    ],
)
def test_estimator_test_parameters(estimator_cls, expected_key):
    """Check that estimators expose their intended test settings."""
    assert "_get_test_params" in estimator_cls.__dict__

    params = estimator_cls._get_test_params()
    variants = params if isinstance(params, list) else [params]

    assert variants
    assert all(isinstance(p, dict) and expected_key in p for p in variants)
