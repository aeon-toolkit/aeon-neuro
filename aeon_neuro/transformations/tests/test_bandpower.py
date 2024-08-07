"""Tests for bandpower."""

import numpy as np
import pytest

from aeon_neuro.transformations import bandpower

# set paramaters, assuming X ~ iid = flat PSD
n_channels, n_timepoints, n_per_seg = 3, 30000, 1024
n_segs = int(n_timepoints / n_per_seg)


@pytest.fixture
def sim_X():
    """Simulate X ~ iid = flat PSD."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(size=(n_channels, n_timepoints))  # axis=1


def test_transform(sim_X):
    """Test BandPowerSeriesTransformer."""
    X = sim_X
    transformer = bandpower.BandPowerSeriesTransformer(
        sfreq=256, n_per_seg=n_per_seg, relative=True
    )

    # estimated power bands
    power_bands = transformer.fit_transform(X)
    # check array and shape
    assert isinstance(power_bands, np.ndarray)
    assert power_bands.shape == (5, n_segs)
    # check relative=True, so powers sum to 1
    np.testing.assert_allclose(power_bands.sum(axis=0), np.ones(shape=n_segs))

    # expected power bands
    power_bands_expected = np.array(
        [
            (max_freq - min_freq)
            for (min_freq, max_freq) in transformer.FREQ_BANDS.values()
        ],
        dtype=np.float64,
    )
    power_bands_expected /= power_bands_expected.sum()
    power_bands_expected = np.tile(
        power_bands_expected.reshape(-1, 1), (1, int(n_timepoints / n_per_seg))
    )
    # check expected power bands for iid = flat PSD
    np.testing.assert_allclose(power_bands, power_bands_expected, atol=0.05)
