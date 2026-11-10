"""Tests for the Nuttall-Strand transformer."""

import numpy as np
import pytest

from aeon_neuro.transformations.series import NuttallStrandSeriesTransformer


def _simulate_var(coefficient, noise_covariance, n_timepoints, seed=0):
    """Simulate a first-order vector autoregressive process."""
    rng = np.random.default_rng(seed)
    n_channels = coefficient.shape[0]
    burn_in = 500
    innovations = np.linalg.cholesky(noise_covariance) @ rng.standard_normal(
        (n_channels, n_timepoints + burn_in)
    )
    series = np.zeros_like(innovations)
    for index in range(1, series.shape[1]):
        series[:, index] = coefficient @ series[:, index - 1] + innovations[:, index]
    return series[:, burn_in:]


def _var_spectrum(coefficient, noise_covariance, frequencies, sfreq):
    """Return the theoretical two-sided spectrum of a VAR(1) process."""
    identity = np.eye(coefficient.shape[0])
    spectrum = []
    for frequency in frequencies:
        polynomial = identity - coefficient * np.exp(-2j * np.pi * frequency / sfreq)
        transfer = np.linalg.solve(polynomial, identity)
        spectrum.append(transfer @ noise_covariance @ transfer.conj().T / sfreq)
    return np.asarray(spectrum)


def test_known_var_spectrum():
    """Estimated matrices follow the theoretical spectrum of a known VAR process."""
    coefficient = np.array([[0.55, 0.15], [-0.1, 0.35]])
    noise_covariance = np.array([[1.0, 0.2], [0.2, 0.6]])
    sfreq = 100
    n_freqs = 6
    series = _simulate_var(coefficient, noise_covariance, 30000)

    result = NuttallStrandSeriesTransformer(
        model_order=1, n_freqs=n_freqs, sfreq=sfreq, fmax=40
    ).fit_transform(series)
    expected = _var_spectrum(
        coefficient, noise_covariance, np.linspace(0, 40, n_freqs), sfreq
    )

    np.testing.assert_allclose(result, expected, rtol=0.12, atol=2e-4)


def test_spectrum_is_hermitian_positive_definite():
    """Every returned spectral density matrix is Hermitian positive definite."""
    rng = np.random.default_rng(1)
    result = NuttallStrandSeriesTransformer(model_order=3, n_freqs=7).fit_transform(
        rng.standard_normal((4, 500))
    )

    assert result.shape == (7, 4, 4)
    np.testing.assert_allclose(result, result.conj().swapaxes(1, 2), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(result) > 0)


def test_univariate_ar_spectrum():
    """The multivariate recursion reduces to the scalar AR spectrum."""
    coefficient = np.array([[0.7]])
    noise_covariance = np.array([[0.5]])
    sfreq = 80
    series = _simulate_var(coefficient, noise_covariance, 20000, seed=2)

    result = NuttallStrandSeriesTransformer(
        model_order=1, n_freqs=5, sfreq=sfreq, fmax=30
    ).fit_transform(series)
    expected = _var_spectrum(
        coefficient, noise_covariance, np.linspace(0, 30, 5), sfreq
    )

    np.testing.assert_allclose(result, expected, rtol=0.1, atol=2e-4)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"model_order": 0}, "model_order must be a positive integer"),
        ({"model_order": 1.5}, "model_order must be a positive integer"),
        ({"n_freqs": 0}, "n_freqs must be a positive integer"),
        ({"sfreq": 0}, "sfreq must be positive"),
        ({"fmin": -1}, "fmin must be non-negative"),
        ({"fmin": 10, "fmax": 10}, "fmax must be greater than fmin"),
        ({"sfreq": 100, "fmax": 51}, "fmax must not exceed"),
    ],
)
def test_invalid_parameters(kwargs, message):
    """Invalid parameters raise informative errors at transform time."""
    with pytest.raises(ValueError, match=message):
        NuttallStrandSeriesTransformer(**kwargs).fit_transform(np.ones((2, 20)))


def test_model_order_must_fit_series():
    """The model order must be smaller than the series length."""
    with pytest.raises(ValueError, match="smaller than the series length"):
        NuttallStrandSeriesTransformer(model_order=20).fit_transform(np.ones((2, 20)))
