"""Test riemannian distance functions."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.distance_based import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)


def test_riemannian_mdm_classifier():
    """Basic test for RiemannianMDMClassifier."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=40,
        random_state=0,
    )

    clf = RiemannianMDMClassifier(metric="riemann")
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_riemannian_knn_classifier():
    """Basic test for RiemannianKNNClassifier."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=40,
        random_state=0,
    )

    clf = RiemannianKNNClassifier(n_neighbors=3, metric="riemann")
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
