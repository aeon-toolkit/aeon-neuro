"""Tests for the TimeDelayEmbedding transformer."""

import numpy as np
import pytest
from aeon.pipeline import make_pipeline

from aeon_neuro.classification.distance_based import RiemannianMDMClassifier
from aeon_neuro.transformations.collection.channel_creation import (
    TimeDelayEmbedding,
)


def _make_data(n_cases=10, n_channels=3, n_timepoints=50):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_cases, n_channels, n_timepoints))


@pytest.mark.parametrize("order, lag", [(1, 1), (2, 1), (3, 2), (4, 5)])
def test_output_matches_delayed_copies(order, lag):
    """Block p of the output is the input delayed by p * lag samples."""
    X = _make_data()
    Xt = TimeDelayEmbedding(order=order, lag=lag).fit_transform(X)

    n_out = 50 - (order - 1) * lag
    assert Xt.shape == (10, 3 * order, n_out)
    for p in range(order):
        np.testing.assert_array_equal(
            Xt[:, 3 * p : 3 * (p + 1)], X[:, :, p * lag : p * lag + n_out]
        )


def test_order_one_is_identity():
    """With order=1 the data is unchanged."""
    X = _make_data()
    np.testing.assert_array_equal(TimeDelayEmbedding(order=1).fit_transform(X), X)


def test_matches_moabb_up_to_the_last_lag():
    """MOABB's AugmentedDataset gives the same output, minus the last lag."""
    X = _make_data()
    order, lag = 3, 2
    moabb = np.concatenate(
        [X[:, :, p * lag : -(order - p) * lag] for p in range(order)], axis=1
    )
    Xt = TimeDelayEmbedding(order=order, lag=lag).fit_transform(X)
    np.testing.assert_array_equal(Xt[:, :, :-lag], moabb)


@pytest.mark.parametrize(
    "params",
    [
        {"order": 0},
        {"order": 1.5},
        {"order": True},
        {"lag": 0},
        {"lag": "1"},
    ],
)
def test_invalid_params(params):
    """Order and lag must be positive integers."""
    with pytest.raises(ValueError, match="must be an integer of at least 1"):
        TimeDelayEmbedding(**params).fit_transform(_make_data())


def test_numpy_integer_params():
    """Order and lag accept numpy integers, e.g. from a parameter grid."""
    X = _make_data()
    Xt = TimeDelayEmbedding(order=np.int64(3), lag=np.int32(2)).fit_transform(X)
    np.testing.assert_array_equal(
        Xt, TimeDelayEmbedding(order=3, lag=2).fit_transform(X)
    )


def test_series_too_short_raises():
    """The embedding must leave at least two timepoints."""
    with pytest.raises(ValueError, match="need series longer than"):
        TimeDelayEmbedding(order=6, lag=10).fit_transform(_make_data())


def _make_dynamics_data(n_cases, random_state=0):
    """Two classes with equal channel covariances but different dynamics.

    Class 0 is white noise. Class 1 is moving-average noise with the same unit
    variance, so only the correlation of each channel with its own past differs.
    """
    rng = np.random.default_rng(random_state)
    y = np.arange(n_cases) % 2
    noise = rng.standard_normal((n_cases, 4, 201))
    smooth = (noise[:, :, 1:] + noise[:, :, :-1]) / np.sqrt(2)
    X = np.where(y[:, None, None] == 1, smooth, noise[:, :, 1:])
    return X, y


def test_acm_captures_dynamics_that_covariance_misses():
    """ACM+MDM separates classes that differ only in their dynamics."""
    X, y = _make_dynamics_data(n_cases=200)
    X_train, y_train, X_test, y_test = X[:140], y[:140], X[140:], y[140:]

    def accuracy(model):
        return np.mean(model.fit(X_train, y_train).predict(X_test) == y_test)

    acm = make_pipeline(TimeDelayEmbedding(order=2), RiemannianMDMClassifier())
    assert accuracy(RiemannianMDMClassifier()) < 0.75
    assert accuracy(acm) > 0.9
