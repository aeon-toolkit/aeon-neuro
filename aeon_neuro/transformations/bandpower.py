"""Band power transformations."""

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer
from scipy.integrate import simpson
from scipy.signal import welch


class BandPowerSeriesTransformer(BaseSeriesTransformer):
    """Band power transformer.

    EEG signals occupy the frequency range of 0 - 60Hz,
    which is roughly divided into five constituent physiological EEG sub bands:
    δ (0 - 4Hz), θ (4 - 7Hz), α (8 - 12Hz), β (13 - 30Hz), and γ (30 - 60Hz).
    Power within each frequency band is estimated using Welch's method,
    which averages over windowed FFTs.

    Parameters
    ----------
    sfreq : int or float
        Sampling frequency in Hz, by default 1.0.
    n_per_seg : int, optional
        Length of each Welch segment in number of timepoints, by default 256.
    n_overlap : int, optional
        Number of timepoints to overlap between segments, by default 0.
    window : str, optional
        Windowing function to use. See `scipy.signal.get_window()`
        for a list of available windows, by default "hamming".
    average : "mean" or "median", optional
        Method to use when averaging Welch segments, by default "mean".
    relative : bool, optional
        If True, return the relative power (divide by total power across freq bands).
        If False, return the absolute power in V^2/Hz, by default True.
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    freq_bands = {
        "delta": (0, 4),
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta": (13, 30),
        "gamma": (30, 60),
    }

    def __init__(
        self,
        sfreq=1.0,  # scipy default
        n_per_seg=256,  # mne/scipy default, for window=str
        n_overlap=0,  # mne default
        window="hamming",  # mne default
        average="mean",  # scipy/mne default
        relative=True,
    ):
        super().__init__(axis=1)  # (n_channels, n_timepoints)
        self.sfreq = sfreq
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.window = window
        self.average = average
        self.relative = relative

    def _transform(self, X, y=None):
        """Transform the input series to extract band power series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Input time series.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_channels, 5_bands)
            Power within δ, θ, α, β, and γ bands for each channel.
        """
        # next power of 2, for FFT efficiency
        n_fft = int(2 ** np.ceil(np.log2(self.n_per_seg)))
        freqs, powers = welch(
            X,
            fs=self.sfreq,
            window=self.window,
            nperseg=self.n_per_seg,
            noverlap=self.n_overlap,
            nfft=n_fft,
            scaling="density",  # V^2/Hz
            axis=-1,
            average=self.average,
        )

        freq_res = freqs[1] - freqs[0]

        band_powers = np.zeros(shape=(len(powers), len(self.freq_bands)))
        for band_idx, (min_freq, max_freq) in enumerate(self.freq_bands.values()):
            freq_mask = np.logical_and(freqs >= min_freq, freqs <= max_freq)
            band_powers[:, band_idx] = simpson(
                powers[:, freq_mask], dx=freq_res, axis=-1
            )

        if self.relative:
            total_powers = band_powers.sum(axis=-1).reshape(-1, 1)
            band_powers /= total_powers

        return band_powers
