"""Tests for bandpower."""

import numpy as np
import pytest

from aeon_neuro.transformations import bandpower

# set paramaters, assuming X ~ iid = flat PSD
n_cases, n_channels, n_timepoints = 2, 3, 10000
freq_bands = {
    "delta": (0, 4),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 60),
}
power_bands_expected = [
    (max_freq - min_freq) for (min_freq, max_freq) in freq_bands.values()
]


@pytest.fixture
def sim_X():
    """Simulate X ~ iid = flat PSD."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(size=(n_cases, n_channels, n_timepoints))


def test_transform(sim_X):
    """Test BandpowerExtraction.transform method."""
    X = sim_X
    bpe = bandpower.BandpowerExtraction(
        band="delta", fs=256, window_width=5120, window_space=10
    )
    power_bands = []
    for band in freq_bands.keys():
        bpe.band = band
        power_bands.append(bpe.transform(X).mean())
    ratios = np.array(power_bands) / np.array(power_bands_expected)
    np.testing.assert_allclose(ratios, ratios[0], rtol=0.01)
