"""Tests for the TimeCrop transformer."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection import TimeCrop


def test_time_crop_numpy_collection():
    """Crop an array using times relative to an event."""
    X = np.arange(2 * 3 * 2048).reshape(2, 3, 2048)

    Xt = TimeCrop(sfreq=512, tmin=0.5, tmax=2.5, start_time=-1.5).fit_transform(X)

    np.testing.assert_array_equal(Xt, X[:, :, 1024:2048])
    assert Xt.shape == (2, 3, 1024)
    assert not np.shares_memory(X, Xt)


def test_time_crop_unequal_length_collection():
    """Apply the same time window to every unequal-length case."""
    X = [
        np.arange(16).reshape(2, 8),
        np.arange(20).reshape(2, 10),
    ]

    Xt = TimeCrop(sfreq=4, tmin=0.5, tmax=1.5).fit_transform(X)

    assert isinstance(Xt, list)
    for actual, original in zip(Xt, X):
        np.testing.assert_array_equal(actual, original[:, 2:6])
        assert not np.shares_memory(actual, original)


@pytest.mark.parametrize(
    "tmin, tmax, expected",
    [
        (None, 1.0, slice(0, 4)),
        (0.5, None, slice(2, None)),
        (None, None, slice(None)),
    ],
)
def test_time_crop_open_window_bounds(tmin, tmax, expected):
    """Use the series bounds when either crop time is None."""
    X = np.arange(16).reshape(1, 2, 8)

    Xt = TimeCrop(sfreq=4, tmin=tmin, tmax=tmax).fit_transform(X)

    np.testing.assert_array_equal(Xt, X[:, :, expected])
    assert not np.shares_memory(X, Xt)


@pytest.mark.parametrize("sfreq", [0, -1, np.inf, np.nan])
def test_time_crop_invalid_sfreq(sfreq):
    """Reject sampling frequencies that are not finite and positive."""
    X = np.zeros((2, 1, 10))

    with pytest.raises(ValueError, match="sfreq must be finite and positive"):
        TimeCrop(sfreq=sfreq).fit_transform(X)


@pytest.mark.parametrize("tmin, tmax", [(1.0, 1.0), (2.0, 1.0)])
def test_time_crop_invalid_window_order(tmin, tmax):
    """Reject empty or reversed explicitly bounded windows."""
    X = np.zeros((2, 1, 10))

    with pytest.raises(ValueError, match="tmin must be less than tmax"):
        TimeCrop(sfreq=2, tmin=tmin, tmax=tmax).fit_transform(X)


@pytest.mark.parametrize("kwargs", [{"tmin": 0.11}, {"tmax": 0.61}, {"tmin": np.nan}])
def test_time_crop_rejects_invalid_sample_times(kwargs):
    """Reject non-finite times and times that are not on the sample grid."""
    X = np.zeros((2, 1, 10))

    with pytest.raises(ValueError, match="sample grid|finite"):
        TimeCrop(sfreq=10, **kwargs).fit_transform(X)


@pytest.mark.parametrize(
    "kwargs", [{"tmin": -1.25, "tmax": 0}, {"tmin": 0, "tmax": 1.75}]
)
def test_time_crop_rejects_out_of_bounds_window(kwargs):
    """Reject a requested window outside any input series."""
    X = np.zeros((2, 1, 10))

    with pytest.raises(ValueError, match="outside the series"):
        TimeCrop(sfreq=4, start_time=-1, **kwargs).fit_transform(X)


def test_time_crop_accepts_roundoff_on_sample_grid():
    """Tolerate floating-point roundoff for a valid sample time."""
    X = np.arange(10).reshape(1, 1, 10)

    Xt = TimeCrop(sfreq=10, tmin=0.3, tmax=0.6, start_time=0.1).fit_transform(X)

    np.testing.assert_array_equal(Xt, X[:, :, 2:5])
