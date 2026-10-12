"""Tests for bandpower."""

import numpy as np
import pytest

from aeon_neuro.transformations.series import BandPowerSeriesTransformer

# set paramaters, assuming X ~ iid = flat PSD
n_channels, n_timepoints, window_size, stride = 3, 30000, 1024, 100
n_windows = (n_timepoints - window_size) // stride + 1


def test_transform():
    """Test BandPowerSeriesTransformer."""
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(n_channels, n_timepoints))  # axis=1
    transformer = BandPowerSeriesTransformer(
        sfreq=256, window_size=window_size, stride=stride, relative=True
    )

    # estimated power bands
    power_bands = transformer.fit_transform(X)
    # check array and shape
    assert isinstance(power_bands, np.ndarray)
    assert power_bands.shape == (5, n_windows)
    # check relative=True, so powers sum to 1
    np.testing.assert_allclose(power_bands.sum(axis=0), np.ones(shape=n_windows))

    # expected power bands
    power_bands_expected = np.array(
        [
            (max_freq - min_freq)
            for (min_freq, max_freq) in transformer.FREQ_BANDS.values()
        ],
        dtype=np.float64,
    )
    power_bands_expected /= power_bands_expected.sum()
    power_bands_expected = np.tile(power_bands_expected.reshape(-1, 1), (1, n_windows))
    # check expected power bands for iid = flat PSD
    np.testing.assert_allclose(power_bands, power_bands_expected, atol=0.08)


def test_transform_nyquist():
    """Test BandPowerSeriesTransformer above/below nyquist."""
    rng = np.random.default_rng(seed=0)
    X = rng.random((32, 1000))
    transformer = BandPowerSeriesTransformer()  # sfreq = 120, window_size = 256
    transformer.fit_transform(X)

    with pytest.raises(ValueError, match="sfreq must be at least .* Hz."):
        bp = BandPowerSeriesTransformer(sfreq=119)
        bp.fit_transform(X)

    with pytest.raises(
        ValueError,
        match="window_size must be at least .* for lowest freqs.",
    ):
        bp = BandPowerSeriesTransformer(sfreq=120, window_size=59)
        bp.fit_transform(X)
    with pytest.raises(ValueError, match="stride must be between 1 and .*"):
        bp = BandPowerSeriesTransformer(window_size=100, stride=101)
        bp.fit_transform(X)
    with pytest.raises(ValueError, match="stride must be between 1 and .*"):
        bp = BandPowerSeriesTransformer(window_size=100, stride=0)
        bp.fit_transform(X)


def test_bands_partition_the_spectrum():
    """Band edges must be contiguous (#173)."""
    bands = list(BandPowerSeriesTransformer.FREQ_BANDS.values())
    assert bands[0][0] == 0 and bands[-1][1] == 60
    for (_, hi), (lo, _) in zip(bands, bands[1:]):
        assert hi == lo, "bands must be contiguous"


def test_flat_spectrum_band_powers_equal_band_widths():
    """On a flat spectrum of height 1 the transformer must recover the exact
    band widths: simpson integrates between the first and last selected
    sample, so inclusive masks are required (#173)."""
    from unittest.mock import patch

    freqs = np.arange(0, 60.25, 0.25)
    powers = np.ones((2, len(freqs), 3))
    with patch(
        "aeon_neuro.transformations.series._bandpower.psd_array_welch",
        return_value=(powers, freqs),
    ):
        transformer = BandPowerSeriesTransformer(
            sfreq=256, window_size=1024, relative=False
        )
        out = transformer.fit_transform(np.zeros((2, 3000)))
    expected = np.tile(
        np.array([4, 4, 5, 17, 30], dtype=float).reshape(-1, 1), (1, 3)
    )
    np.testing.assert_allclose(out, expected, atol=1e-9)
