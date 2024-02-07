import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.integrate import simps


class BandpowerExtraction:
    def __init__(self, fs, band, window_width=1000, window_space=5):
        self.fs = fs
        self.band = band
        self.window_width = window_width
        self.window_space = window_space

    def transform(self, X, y=None):
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
                    1, n_timepoints - self.window_width, self.window_space
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
    if band == "delta":
        return 4, 0
    elif band == "theta":
        return 8, 4
    elif band == "alpha":
        return 12, 8
    elif band == "beta":
        return 30, 12
    elif band == "gamma":
        return 100, 30
    else:
        return 100, 0
