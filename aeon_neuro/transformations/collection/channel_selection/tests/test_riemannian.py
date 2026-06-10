"""Tests for the Riemannian channel selector."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection.channel_selection import Riemannian


def test_riemannian_regularizes_singular_covariances():
    """Diagonal loading should make singular covariances positive definite."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 2, 50))
    X = np.concatenate((X, X), axis=1)
    y = np.array([0, 1] * 10)

    selector = Riemannian(proportion=0.5, regularization=1e-6)
    Xt = selector.fit_transform(X, y)

    assert Xt.shape == (20, 2, 50)
    assert np.all(np.linalg.eigvalsh(selector.covariances_) > 0)


def test_riemannian_rejects_negative_regularization():
    """Regularization cannot be negative."""
    X = np.ones((4, 2, 10))
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="regularization"):
        Riemannian(regularization=-1).fit(X, y)
