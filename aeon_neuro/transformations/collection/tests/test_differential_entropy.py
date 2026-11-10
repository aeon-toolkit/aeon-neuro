"""Tests for the DifferentialEntropy transformer."""

import numpy as np
import pytest
from aeon.testing.estimator_checking import check_estimator

from aeon_neuro.transformations.collection import DifferentialEntropy
from aeon_neuro.transformations.collection._differential_entropy import DE_BANDS


def test_differential_entropy_estimator_checks():
    """Run aeon's estimator checks."""
    check_estimator(DifferentialEntropy, raise_exceptions=True)


def test_differential_entropy_matches_closed_form_for_a_sine():
    """A sine inside one band has DE 0.5 * log(2 * pi * e * A**2 / 2) there."""
    sfreq, amplitude = 200.0, 3.0
    t = np.arange(4000) / sfreq
    X = (amplitude * np.sin(2 * np.pi * 10.0 * t))[None, None, :]
    de = DifferentialEntropy(sfreq=sfreq).fit_transform(X)[0]
    expected = 0.5 * np.log(2 * np.pi * np.e * amplitude**2 / 2)
    alpha = DE_BANDS.index((8.0, 14.0))
    assert de[alpha] == pytest.approx(expected, abs=0.05)
    others = np.delete(de, alpha)
    assert np.all(others < expected - 2)


def test_differential_entropy_shape_and_feature_order():
    """Features are ordered band by band, n_channels per band."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 3, 500))
    X[:, 1] *= 10  # channel 1 has 100 times the power in every band
    de = DifferentialEntropy(sfreq=250.0).fit_transform(X)
    assert de.shape == (4, 3 * len(DE_BANDS))
    per_band = de.reshape(4, len(DE_BANDS), 3)
    assert np.all(per_band[:, :, 1] > per_band[:, :, 0])
    assert np.all(per_band[:, :, 1] > per_band[:, :, 2])


def test_differential_entropy_drops_and_caps_bands_above_nyquist():
    """At 64 Hz a high edge is capped and a band above Nyquist is dropped."""
    X = np.random.default_rng(0).standard_normal((2, 2, 256))
    trans = DifferentialEntropy(sfreq=64.0, bands=[(8, 14), (20, 50), (40, 60)])
    out = trans.fit_transform(X)
    assert trans.bands_ == [(8.0, 14.0), (20.0, 0.95 * 32.0)]
    assert out.shape == (2, 4)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sfreq": 0}, "sfreq"),
        ({"sfreq": 100.0, "bands": [(10, 5)]}, "Invalid band"),
        ({"sfreq": 10.0, "bands": [(8, 14)]}, "Nyquist"),
    ],
)
def test_differential_entropy_invalid_parameters(kwargs, match):
    """Bad sampling rates and bands raise a ValueError."""
    X = np.random.default_rng(0).standard_normal((2, 2, 100))
    with pytest.raises(ValueError, match=match):
        DifferentialEntropy(**kwargs).fit(X)
