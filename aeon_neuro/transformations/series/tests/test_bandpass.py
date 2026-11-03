"""Tests for bandpass."""

import numpy as np
import pytest

from aeon_neuro.transformations.series import BandPassSeriesTransformer

sfreq, n_channels, duration = 250, 3, 10  # 10 s at 250 Hz
t = np.arange(sfreq * duration) / sfreq


def _sine(freq):
    return np.sin(2 * np.pi * freq * t)


def _amplitude(x, freq):
    """Amplitude of a single frequency component via the FFT."""
    spectrum = np.abs(np.fft.rfft(x)) * 2 / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / sfreq)
    return spectrum[np.argmin(np.abs(freqs - freq))]


@pytest.mark.parametrize("method", ["fir", "iir"])
def test_transform_shape(method):
    """Output keeps the input shape and is a float array."""
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(n_channels, len(t)))
    transformer = BandPassSeriesTransformer(
        sfreq=sfreq, l_freq=8, h_freq=30, method=method
    )
    Xt = transformer.fit_transform(X)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == X.shape
    assert Xt.dtype == np.float64


@pytest.mark.parametrize("method", ["fir", "iir"])
def test_transform_keeps_passband(method):
    """A 20 Hz tone survives an 8-30 Hz filter; 2 Hz and 60 Hz are removed."""
    x = _sine(2) + _sine(20) + _sine(60)
    X = np.tile(x, (n_channels, 1))
    transformer = BandPassSeriesTransformer(
        sfreq=sfreq, l_freq=8, h_freq=30, method=method
    )
    Xt = transformer.fit_transform(X)

    for channel in Xt:
        assert _amplitude(channel, 20) == pytest.approx(1, abs=0.1)
        assert _amplitude(channel, 2) < 0.05
        assert _amplitude(channel, 60) < 0.05


def test_transform_highpass_and_lowpass():
    """l_freq=None gives a lowpass, h_freq=None gives a highpass."""
    X = np.tile(_sine(2) + _sine(60), (n_channels, 1))

    lowpass = BandPassSeriesTransformer(sfreq=sfreq, l_freq=None, h_freq=30)
    Xt = lowpass.fit_transform(X)
    assert _amplitude(Xt[0], 2) == pytest.approx(1, abs=0.1)
    assert _amplitude(Xt[0], 60) < 0.05

    highpass = BandPassSeriesTransformer(sfreq=sfreq, l_freq=30, h_freq=None)
    Xt = highpass.fit_transform(X)
    assert _amplitude(Xt[0], 60) == pytest.approx(1, abs=0.1)
    assert _amplitude(Xt[0], 2) < 0.05


def test_transform_integer_input():
    """Integer input is cast to float64 rather than rejected by MNE."""
    rng = np.random.default_rng(seed=0)
    X = rng.integers(-100, 100, size=(n_channels, len(t)))
    Xt = BandPassSeriesTransformer(sfreq=sfreq).fit_transform(X)
    assert Xt.shape == X.shape
    assert Xt.dtype == np.float64


@pytest.mark.parametrize(
    "params, match",
    [
        ({"l_freq": None, "h_freq": None}, "At least one of l_freq or h_freq"),
        ({"l_freq": 0}, "l_freq must be greater than 0 Hz."),
        ({"l_freq": -1}, "l_freq must be greater than 0 Hz."),
        ({"h_freq": 125}, "h_freq must be less than 125.0 Hz."),
        ({"h_freq": 200}, "h_freq must be less than 125.0 Hz."),
        ({"l_freq": 30, "h_freq": 30}, "l_freq must be less than h_freq."),
        ({"l_freq": 40, "h_freq": 30}, "l_freq must be less than h_freq."),
        ({"method": "butter"}, "method must be 'fir' or 'iir'."),
        ({"l_freq": 125, "h_freq": None}, "l_freq must be less than 125.0 Hz."),
        ({"l_freq": 200, "h_freq": None}, "l_freq must be less than 125.0 Hz."),
        ({"sfreq": 0}, "sfreq must be a finite positive number."),
        ({"sfreq": -250}, "sfreq must be a finite positive number."),
        ({"sfreq": np.inf}, "sfreq must be a finite positive number."),
        ({"sfreq": np.nan}, "sfreq must be a finite positive number."),
        ({"sfreq": "250"}, "sfreq must be a finite positive number."),
        ({"sfreq": True}, "sfreq must be a finite positive number."),
    ],
)
def test_transform_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X = np.zeros((n_channels, len(t)))
    transformer = BandPassSeriesTransformer(**{"sfreq": sfreq, **params})
    with pytest.raises(ValueError, match=match):
        transformer.fit_transform(X)


def test_sfreq_is_required():
    """The sampling frequency has no default and must be passed."""
    with pytest.raises(TypeError):
        BandPassSeriesTransformer()
