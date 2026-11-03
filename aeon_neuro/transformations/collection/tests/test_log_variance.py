"""Tests for the LogVariance transformer."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.transformations.collection import LogVariance


def test_log_variance_shape_and_values():
    """Test output is (n_cases, n_channels) log-variance over time."""
    X, _ = make_example_3d_numpy(
        n_cases=6, n_channels=3, n_timepoints=50, random_state=0
    )
    Xt = LogVariance().fit_transform(X)

    assert Xt.shape == (6, 3)
    np.testing.assert_allclose(Xt, np.log(np.var(X, axis=2)))


def test_log_variance_raw_variance():
    """Test log=False returns raw variance."""
    X, _ = make_example_3d_numpy(
        n_cases=6, n_channels=3, n_timepoints=50, random_state=0
    )
    Xt = LogVariance(log=False).fit_transform(X)

    assert Xt.shape == (6, 3)
    np.testing.assert_allclose(Xt, np.var(X, axis=2))


@pytest.mark.parametrize("log", [True, False])
def test_log_variance_zero_variance_is_finite(log):
    """Test constant channels give finite features."""
    X = np.zeros((4, 2, 30))
    X[:, 1] = np.random.default_rng(0).normal(size=(4, 30))

    Xt = LogVariance(log=log).fit_transform(X)

    assert np.all(np.isfinite(Xt))
    if log:
        assert np.all(Xt[:, 0] < Xt[:, 1])
    else:
        np.testing.assert_array_equal(Xt[:, 0], 0.0)


def test_log_variance_univariate():
    """Test a single channel gives a single feature."""
    X = np.random.default_rng(0).normal(size=(5, 1, 40))
    assert LogVariance().fit_transform(X).shape == (5, 1)
