"""Tests for the Detach-ROCKET channel selector."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import RidgeClassifier

from aeon_neuro.transformations.collection.channel_selection import (
    DetachRocketChannelSelector,
)
from aeon_neuro.transformations.collection.channel_selection._detach_rocket import (
    _feature_detachment,
    _select_detach_model,
)


def test_detach_rocket_selects_informative_channel():
    """Test that the informative channel is retained."""
    rng = np.random.RandomState(42)
    y = np.array([0, 1] * 20)
    X = rng.normal(size=(40, 3, 20))
    X[y == 1, 0] += 3

    selector = DetachRocketChannelSelector(
        proportion=1 / 3,
        n_kernels=84,
        max_detach_steps=4,
        alphas=[0.1, 1.0],
        random_state=0,
    )
    Xt = selector.fit_transform(X, y)

    assert Xt.shape == (40, 1, 20)
    assert_array_equal(selector.channels_selected_, [0])
    assert selector.channel_scores_.shape == (3,)


def test_detach_rocket_is_reproducible():
    """Test deterministic selection for a fixed random state."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(30, 4, 20))
    y = np.array([0, 1] * 15)
    params = {
        "proportion": 0.5,
        "n_kernels": 84,
        "max_detach_steps": 3,
        "alphas": [1.0],
        "random_state": 7,
    }

    first = DetachRocketChannelSelector(**params).fit(X, y)
    second = DetachRocketChannelSelector(**params).fit(X, y)

    assert_array_equal(first.channels_selected_, second.channels_selected_)
    assert_array_equal(first.channel_scores_, second.channel_scores_)


def test_sfd_matches_upstream_reference():
    """Test SFD output captured from upstream commit aa046a3."""
    rng = np.random.RandomState(123)
    X_train = rng.normal(size=(24, 12))
    X_validation = rng.normal(size=(12, 12))
    y_train = np.array([0, 1, 2] * 8)
    y_validation = np.array([0, 1, 2] * 4)
    classifier = RidgeClassifier(alpha=1.0).fit(X_train, y_train)

    percentages, validation_scores = _feature_detachment(
        classifier,
        X_train,
        X_validation,
        y_train,
        y_validation,
        drop_percentage=0.2,
        max_steps=8,
        multilabel_type="max",
    )

    assert_allclose(
        percentages,
        [1, 0.75, 7 / 12, 0.5, 1 / 3, 0.25, 1 / 6],
    )
    assert_allclose(
        validation_scores,
        [0.5, 7 / 12, 0.5, 0.5, 1 / 3, 1 / 3, 5 / 12],
    )
    assert _select_detach_model(percentages, validation_scores, 0.1) == 2
