"""Tests for downsample_series."""

import numpy as np
import pytest

from aeon_neuro.transformations.downsample import downsample_series

X = np.array(
    [
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        [
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        ],
    ]
)

testdata = [
    (
        X,
        10,
        5,
        np.array(
            [
                [[1, 3, 5, 7, 9], [10, 8, 6, 4, 2]],
                [[10, 30, 50, 70, 90], [100, 80, 60, 40, 20]],
            ]
        ),
    ),
    (
        X,
        10,
        3,
        np.array(
            [[[1, 4, 7, 10], [10, 7, 4, 1]], [[10, 40, 70, 100], [100, 70, 40, 10]]]
        ),
    ),
    (X, 10, 2, np.array([[[1, 6], [10, 5]], [[10, 60], [100, 50]]])),
]


@pytest.mark.parametrize("X, sfreq, target_sample_rate, expected", testdata)
def test_downsample_series(X, sfreq, target_sample_rate, expected):
    """Test the downsampling of a time series."""
    result = downsample_series(X, sfreq, target_sample_rate)
    np.testing.assert_array_equal(result, expected)
