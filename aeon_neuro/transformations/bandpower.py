"""Band power transformations."""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.integrate import simps


class BandpowerExtraction:
    """Band power transformer.

    EEG signals occupy the frequency range of 0 − 60Hz,
    which is roughly divided into five constituent physiological EEG sub bands:
    δ (0 − 4Hz), θ (4 − 7Hz), α (8 − 12Hz), β (13 − 30Hz), and γ (30 − 60Hz).

    Parameters
    ----------
    fs : int or float
        Sampling frequency in Hz (samples per second).
    band : str
        The frequency band in {"delta", "theta", "alpha", "beta", "gamma"}.
    window_width : int, optional
        The width of the sliding window in number of samples, by default 1000.
    window_space : int, optional
        The space between the start of each window in number of samples, by default 5.
    """

    def __init__(self, fs, band, window_width=1000, window_space=5):
        self.fs = fs
        self.band = band
        self.window_width = window_width
        self.window_space = window_space

    def transform(self, X, y=None):
        """Transform the input collection to extract band power features.

        Parameters
        ----------
        X : list or np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Input time series collection.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_cases, n_channels, n_windows)
            The band power features for a given frequency band.
        """
        max_freq, min_freq = selectBandFreqs(self.band)
        n_instances, n_channels, n_timepoints = np.shape(X)
        final_data = np.zeros(
            (
                n_instances,
                n_channels,
                int((n_timepoints - self.window_width) / self.window_space) + 1,
            )
        )
        for instance in range(n_instances):
            for channel in range(n_channels):
                for timepoint in range(
                    1,
                    n_timepoints - self.window_width,
                    self.window_space,
                ):
                    w = X[instance][channel][timepoint : timepoint + self.window_width]
                    coefficients = rfft(w)[1:]
                    freqs = rfftfreq(self.window_width, 1 / self.fs)[1:]
                    delta = np.logical_and(freqs >= min_freq, freqs <= max_freq)
                    powers = np.abs(coefficients[delta])
                    value = simps(powers, dx=freqs[1] - freqs[0])
                    final_data[instance][channel][
                        int(timepoint / self.window_space)
                    ] = value
        return final_data


def selectBandFreqs(band):
    """Return (max, min) frequency for given frequency band."""
    if band == "delta":
        return 4, 0
    elif band == "theta":
        return 7, 4
    elif band == "alpha":
        return 12, 8
    elif band == "beta":
        return 30, 13
    elif band == "gamma":
        return 60, 30
    else:
        return 60, 0
