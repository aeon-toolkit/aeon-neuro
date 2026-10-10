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


@pytest.mark.parametrize(
    "kwargs, n_timepoints_expected",
    [
        ({"downsample_by": "proportion", "proportion": 0.1}, 900),
        ({"downsample_by": "proportion", "proportion": 0.6}, 400),
        ({"downsample_by": "proportion", "proportion": 0.5}, 500),
        ({"downsample_by": "frequency", "source_sfreq": 250, "target_sfreq": 100}, 400),
        ({"downsample_by": "frequency", "source_sfreq": 250, "target_sfreq": 125}, 500),
        ({"downsample_by": "frequency", "source_sfreq": 10, "target_sfreq": 3}, 300),
    ],
)
def test_downsample_non_integer_ratio_shape(kwargs, n_timepoints_expected):
    """Test the output length matches the requested ratio (#171)."""
    X = np.zeros((2, 3, 1000))
    X_downsampled = DownsampleCollectionTransformer(**kwargs).fit_transform(X)
    assert X_downsampled.shape == (2, 3, n_timepoints_expected)


def test_downsample_non_integer_ratio_values():
    """Test a non-integer ratio resamples a slow sine onto the new time grid."""
    source_sfreq, target_sfreq, n_timepoints = 250, 100, 2500
    signal_freq = 2.0  # Hz, far below the new Nyquist frequency of 50 Hz
    t_source = np.arange(n_timepoints) / source_sfreq
    X = np.sin(2 * np.pi * signal_freq * t_source).reshape(1, 1, -1)

    X_downsampled = DownsampleCollectionTransformer(
        downsample_by="frequency", source_sfreq=source_sfreq, target_sfreq=target_sfreq
    ).fit_transform(X)

    t_target = np.arange(X_downsampled.shape[-1]) / target_sfreq
    expected = np.sin(2 * np.pi * signal_freq * t_target)
    # ignore filter edge effects at both ends
    inner = slice(50, -50)
    np.testing.assert_allclose(X_downsampled[0, 0, inner], expected[inner], atol=1e-2)


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
