"""Tests for epoching."""

import numpy as np

from aeon_neuro.transformations import epoching

# set parameters
n_cases, n_channels, n_timepoints, sfreq = 3, 2, 12, 1
epoch_size, percent = 3000, 25
n_timepoints_per_epoch = int((epoch_size / 1000) * sfreq)
n_epochs = int(n_timepoints / n_timepoints_per_epoch)

# test series
series = np.arange(n_channels * n_timepoints).reshape(n_channels, -1)
series_expected = np.array(
    [
        [[0, 1, 2], [12, 13, 14]],
        [[3, 4, 5], [15, 16, 17]],
        [[6, 7, 8], [18, 19, 20]],
        [[9, 10, 11], [21, 22, 23]],
    ]
)

# test dataset
X = np.array([series, series + 1, series + 2])
X_expected = np.vstack([series_expected, series_expected + 1, series_expected + 2])
y = np.arange(n_cases)
y_expected = np.repeat(y, n_epochs)


def test_epoch_series_by_percentage():
    """Test series-to-collection transformation."""
    series_transformed = epoching.epoch_series_by_percentage(series, percent)
    np.testing.assert_array_equal(series_expected, series_transformed)


def test_epoch_series_by_time():
    """Test series-to-collection transformation."""
    series_transformed = epoching.epoch_series_by_time(series, sfreq, epoch_size)
    np.testing.assert_array_equal(series_expected, series_transformed)


def test_epoch_dataset():
    """Test collection-to-collection transformation."""
    X_transformed, y_transformed = epoching.epoch_dataset(X, y, sfreq, epoch_size)
    np.testing.assert_array_equal(X_transformed, X_expected)
    np.testing.assert_array_equal(y_transformed, y_expected)
