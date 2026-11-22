"""Tests for the BoVWTransformer.

Comparison with Wang et al. (2013)
----------------------------------
The comparison below was run by Claude (Anthropic) as an AI coding assistant.
It is not part of the automated test suite, because it downloads data and
takes about 20 minutes.

Data: Bonn epilepsy EEG dataset (Andrzejak et al., 2001), sets A-E, 5 classes,
500 single-channel series truncated to 4096 samples.

Setup: BoVWTransformer(n_words=1000, window_size=128, stride=8,
descriptor="dwt", wavelet="db3"), then 1-NN with Chi-squared distance,
stratified 10-fold cross-validation.

Fold accuracies: 76.0, 94.0, 94.0, 92.0, 92.0, 94.0, 84.0, 90.0, 88.0, 84.0
Mean: 88.8% +/- 5.6%. Paper (Table II, 5-class EEG): 93.8%.

The paper reports its best result over a parameter sweep (codebook size up to
3500), while this run used a single untuned setting with the coarsest step the
paper mentions (8). The folds also differ, since the paper's partition is not
published.
"""

import sys

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.estimator_checking import check_estimator

from aeon_neuro.transformations.collection import BoVWTransformer


def _data(n_cases=6, n_channels=1, n_timepoints=40):
    X, _ = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=0,
    )
    return X


def test_bovw_estimator_checks():
    """Test BoVWTransformer passes the aeon estimator checks."""
    check_estimator(BoVWTransformer, raise_exceptions=True)


@pytest.mark.parametrize("n_channels", [1, 3])
def test_bovw_output_shape(n_channels):
    """Test output is (n_cases, n_words) for uni- and multivariate input."""
    X = _data(n_channels=n_channels)
    Xt = BoVWTransformer(
        n_words=5, window_size=8, descriptor="raw", random_state=0
    ).fit_transform(X)

    assert Xt.shape == (6, 5)


def test_bovw_normalised_histograms_sum_to_one():
    """Test normalised histograms sum to one per case."""
    X = _data()
    Xt = BoVWTransformer(
        n_words=5, window_size=8, descriptor="raw", random_state=0
    ).fit_transform(X)

    np.testing.assert_allclose(Xt.sum(axis=1), 1.0)
    assert np.all(Xt >= 0)


@pytest.mark.parametrize("stride", [1, 3])
def test_bovw_counts_sum_to_n_windows(stride):
    """Test unnormalised histograms count every window once."""
    X = _data(n_timepoints=40)
    Xt = BoVWTransformer(
        n_words=5,
        window_size=8,
        stride=stride,
        descriptor="raw",
        normalise=False,
        random_state=0,
    ).fit_transform(X)

    n_windows = len(range(0, 40 - 8 + 1, stride))
    np.testing.assert_array_equal(Xt.sum(axis=1), n_windows)


def test_bovw_invariant_to_offset_and_scale():
    """Test z-normalised windows ignore per-case offset and scale."""
    X = _data()
    bovw = BoVWTransformer(n_words=5, window_size=8, descriptor="raw", random_state=0)
    bovw.fit(X)

    np.testing.assert_allclose(bovw.transform(X), bovw.transform(2 * X + 9))


def test_bovw_flat_series_is_finite():
    """Test constant windows do not produce NaNs."""
    X = _data()
    X[0] = 5.0
    Xt = BoVWTransformer(
        n_words=3, window_size=8, descriptor="raw", random_state=0
    ).fit_transform(X)

    assert np.all(np.isfinite(Xt))


def test_bovw_random_state_is_reproducible():
    """Test the same random_state gives the same output."""
    X = _data()
    kwargs = dict(n_words=5, window_size=8, descriptor="raw", random_state=1)

    np.testing.assert_array_equal(
        BoVWTransformer(**kwargs).fit_transform(X),
        BoVWTransformer(**kwargs).fit_transform(X),
    )


def test_bovw_dwt_descriptor():
    """Test the DWT descriptor halves the codebook dimension."""
    pytest.importorskip("pywt")
    X = _data(n_channels=2, n_timepoints=64)
    bovw = BoVWTransformer(
        n_words=4, window_size=16, descriptor="dwt", wavelet="db3", random_state=0
    )
    Xt = bovw.fit_transform(X)

    assert Xt.shape == (6, 4)
    np.testing.assert_allclose(Xt.sum(axis=1), 1.0)
    # db3 single level: (16 + 5) // 2 = 10 coefficients per channel
    assert bovw.kmeans_.cluster_centers_.shape[1] == 2 * 10


def test_bovw_dwt_without_pywavelets_raises(monkeypatch):
    """Test a clear ImportError when PyWavelets is missing."""
    monkeypatch.setitem(sys.modules, "pywt", None)
    with pytest.raises(ImportError, match="PyWavelets"):
        BoVWTransformer(n_words=3, window_size=8, descriptor="dwt").fit(_data())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_words": 0},
        {"window_size": -1},
        {"stride": 1.5},
        {"descriptor": "fft"},
    ],
)
def test_bovw_invalid_params_raise(kwargs):
    """Test invalid parameters raise a ValueError on fit."""
    params = dict(n_words=3, window_size=8, descriptor="raw") | kwargs
    with pytest.raises(ValueError):
        BoVWTransformer(**params).fit(_data())


def test_bovw_window_longer_than_series_raises():
    """Test window_size longer than the series raises a ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        BoVWTransformer(n_words=3, window_size=50, descriptor="raw").fit(
            _data(n_timepoints=40)
        )
