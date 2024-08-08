"""Tests for epoching."""

import numpy as np

from aeon_neuro.transformations import EpochSeriesTransformer
from aeon_neuro.transformations._epoching import epoch_dataset

# set parameters
n_cases, n_channels, n_timepoints, sfreq = 3, 2, 12, 1
epoch_size, percent = 3000, 25
n_timepoints_per_epoch = int((epoch_size / 1000) * sfreq)
n_epochs = int(n_timepoints / n_timepoints_per_epoch)

# test series
X_series = np.arange(n_channels * n_timepoints).reshape(n_channels, -1)
X_series_expected = np.array(
    [
        [[0, 1, 2], [12, 13, 14]],
        [[3, 4, 5], [15, 16, 17]],
        [[6, 7, 8], [18, 19, 20]],
        [[9, 10, 11], [21, 22, 23]],
    ]
)

# test dataset
X_collection = np.array([X_series, X_series + 1, X_series + 2])
X_collection_expected = np.vstack(
    [X_series_expected, X_series_expected + 1, X_series_expected + 2]
)
y = np.arange(n_cases)
y_expected = np.repeat(y, n_epochs)


def test_epoch_series_by_percentage():
    """Test series-to-collection transformation."""
    transformer = EpochSeriesTransformer(percent=percent)
    X_series_transformed = transformer.fit_transform(X_series)
    assert isinstance(X_series_transformed, np.ndarray)
    np.testing.assert_array_equal(X_series_expected, X_series_transformed)


def test_epoch_series_by_time():
    """Test series-to-collection transformation."""
    transformer = EpochSeriesTransformer(sfreq=sfreq, epoch_size=epoch_size)
    X_series_transformed = transformer.fit_transform(X_series)
    assert isinstance(X_series_transformed, np.ndarray)
    np.testing.assert_array_equal(X_series_expected, X_series_transformed)


def test_epoch_dataset():
    """Test collection-to-collection transformation."""
    X_transformed, y_transformed = epoch_dataset(X_collection, y, sfreq, epoch_size)
    np.testing.assert_array_equal(X_transformed, X_collection_expected)
    np.testing.assert_array_equal(y_transformed, y_expected)
