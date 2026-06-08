"""Tests for downsample_series."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection import DownsampleCollectionTransformer

# list of 2D numpy arrays, unequal lengths
X = [
    np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ]
    ),
    np.array(
        [
            [10, 20, 30, 40, 50, 60, 70, 80],
            [100, 90, 80, 70, 60, 50, 40, 30],
        ]
    ),
]

testdata = [
    (
        X,
        10,
        5,
        [
            np.array([[1, 3, 5, 7, 9], [10, 8, 6, 4, 2]]),
            np.array([[10, 30, 50, 70], [100, 80, 60, 40]]),
        ],
    ),
    (
        X,
        10,
        3,
        [
            np.array([[1, 4, 7, 10], [10, 7, 4, 1]]),
            np.array([[10, 40, 70], [100, 70, 40]]),
        ],
    ),
    (
        X,
        10,
        2,
        [
            np.array([[1, 6], [10, 5]]),
            np.array([[10, 60], [100, 50]]),
        ],
    ),
]


@pytest.mark.parametrize("X, source_sfreq, target_sfreq, X_expected", testdata)
def test_downsample_series(X, source_sfreq, target_sfreq, X_expected):
    """Test the downsampling of a time series."""
    proportion = 1 - (target_sfreq / source_sfreq)
    freq_downsample = DownsampleCollectionTransformer(
        downsample_by="frequency", source_sfreq=source_sfreq, target_sfreq=target_sfreq
    )
    prop_downsample = DownsampleCollectionTransformer(
        downsample_by="proportion", proportion=proportion
    )
    X_freq_downsample = freq_downsample.fit_transform(X)
    X_prop_downsample = prop_downsample.fit_transform(X)
    assert isinstance(X_freq_downsample, list)
    assert isinstance(X_prop_downsample, list)
    for idx in range(2):
        np.testing.assert_array_equal(X_freq_downsample[idx], X_expected[idx])
        np.testing.assert_array_equal(X_prop_downsample[idx], X_expected[idx])


def test_value_errors():
    """Test DowmsampleCollectionTransformer ValueErrors."""
    # default initialization
    rng = np.random.default_rng(seed=0)
    X = rng.random((32, 1000))
    transformer = DownsampleCollectionTransformer()
    transformer.fit_transform(X)

    # invalid initializations
    with pytest.raises(
        ValueError, match='downsample_by must be either "frequency" or "proportion"'
    ):
        DownsampleCollectionTransformer(downsample_by="invalid")

    with pytest.raises(ValueError, match="source_sfreq must be > target_sfreq"):
        DownsampleCollectionTransformer(
            downsample_by="frequency", source_sfreq=1.0, target_sfreq=2.0
        )

    with pytest.raises(
        ValueError, match="proportion must be provided and between 0-1."
    ):
        DownsampleCollectionTransformer(downsample_by="proportion", proportion=1)

    with pytest.raises(
        ValueError, match="proportion must be provided and between 0-1."
    ):
        DownsampleCollectionTransformer(downsample_by="proportion", proportion=0)
