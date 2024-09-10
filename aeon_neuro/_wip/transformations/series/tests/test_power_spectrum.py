"""Tests for cross spectral matrix transform."""

import numpy as np
import pytest

from aeon_neuro._wip.distances._riemannian_matrix import _is_hpd
from aeon_neuro._wip.transformations.series._power_spectrum import CrossSpectralMatrix

n_channels, n_timepoints = 7, 12000
rng = np.random.default_rng(seed=0)


def test_transform():
    """Test cross spectral matrix transformer."""
    transformer = CrossSpectralMatrix()
    X = rng.standard_normal(size=(n_channels, n_timepoints))
    X_transformed = transformer.fit_transform(X)

    assert X_transformed.shape == (n_channels, n_channels)  # expected shape
    assert _is_hpd(X_transformed)  # Hermitian Positive Definite


def test_value_errors():
    """Test cross spectral matrix errors."""
    transformer = CrossSpectralMatrix()
    X = rng.standard_normal(size=(n_channels, n_timepoints))
    transformer.fit_transform(X)

    with pytest.raises(ValueError, match="FFT cannot be computed for sfreq=.*"):
        transformer.set_params(fmin=100, fmax=110)
        transformer.fit_transform(X)
