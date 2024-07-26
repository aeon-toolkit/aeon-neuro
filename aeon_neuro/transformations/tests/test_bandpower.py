"""Tests for bandpower."""

import numpy as np
import pytest

from aeon_neuro.transformations import bandpower

# set paramaters, assuming X ~ iid = flat PSD
n_channels, n_timepoints = 3, 10000


@pytest.fixture
def sim_X():
    """Simulate X ~ iid = flat PSD."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(size=(n_channels, n_timepoints))  # axis=1


def test_transform(sim_X):
    """Test BandPowerSeriesTransformer."""
    X = sim_X
    transformer = bandpower.BandPowerSeriesTransformer(
        sfreq=256, n_per_seg=1024, n_overlap=512, relative=True
    )

    # estimated power bands
    power_bands = transformer.fit_transform(X)
    # check array and shape
    assert isinstance(power_bands, np.ndarray)
    assert power_bands.shape == (n_channels, 5)
    # check relative=True, so powers sum to 1
    np.testing.assert_equal(power_bands.sum(axis=-1), np.ones(shape=n_channels))

    # expected power bands
    power_bands_expected = np.array(
        [
            (max_freq - min_freq)
            for (min_freq, max_freq) in transformer.freq_bands.values()
        ],
        dtype=np.float64,
    )
    power_bands_expected /= power_bands_expected.sum()
    power_bands_expected = np.tile(power_bands_expected, (n_channels, 1))
    # check expected power bands for iid = flat PSD
    np.testing.assert_allclose(power_bands, power_bands_expected, atol=0.03)
