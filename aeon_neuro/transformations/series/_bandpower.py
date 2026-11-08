"""Band power transformations."""

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation import check_n_jobs
from mne.time_frequency import psd_array_welch
from scipy.integrate import simpson

__all__ = ["BandPowerSeriesTransformer"]


class BandPowerSeriesTransformer(BaseSeriesTransformer):
    """Band power transformer.

    EEG signals occupy the frequency range of 0 - 60Hz,
    which is roughly divided into five constituent physiological EEG sub bands:
    delta (δ): 0 - 4Hz, theta (θ): 4 - 8Hz, alpha (α): 8 - 13Hz, beta (β): 13 - 30Hz
    and gamma (γ): 30 - 60Hz.
    Power within each frequency band is estimated over time using windowed FFTs

    The transformer uses psd_array_welch from MNE to calculate power spectral
    densities for each window for the given sampling frequency. Band powers are then
    found through summing over band widths and channels.

    Parameters
    ----------
    sfreq : int or float
        Sampling frequency in Hz, by default 120.
    window_size : int, optional
        Size of each window in number of timepoints, by default 256.
    window_function : str, optional
        Windowing function to use. See `scipy.signal.get_window()`
        for a list of available windows, by default "hamming".
    stride : int, optional
        Step size between successive windows in number of timepoints.
        If None, `stride = window_size`, by default None.
    relative : bool, optional
        If True, return the relative power (divide by total power across freq bands).
        If False, return the absolute power in V^2/Hz, by default True.
    n_jobs : int, optional
        Number of jobs to calculate power spectral densities, by default 1.

    Raises
    ------
    ValueError
        If `sfreq` is too low to capture the highest frequency band.
        If `window_size` is too small to capture the lowest frequency band.
        If `stride` is not between 1 and `window_size`.

    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    # Contiguous partition of [0, 60] Hz.
    FREQ_BANDS = {
        "delta": (0, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60),
    }

    def __init__(
        self,
        sfreq=120,  # 2x60Hz = 120Hz
        window_size=256,
        window_function="hamming",  # mne default
        stride=None,
        relative=True,
        n_jobs=1,
    ):
        self.sfreq = sfreq
        self.window_size = window_size
        self.window_function = window_function
        self.stride = stride
        self.relative = relative
        self.n_jobs = n_jobs
        super().__init__(axis=1)  # (n_channels, n_timepoints)

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
        np.ndarray of shape (5_bands, (n_timepoints - window_size) // stride + 1)
            Power within δ, θ, α, β, and γ bands over time.
        """
        # checks
        n_jobs = check_n_jobs(self.n_jobs)
        nyquist_freq = 2 * self.FREQ_BANDS["gamma"][1]
        if self.sfreq < nyquist_freq:
            raise ValueError(f"sfreq must be at least {nyquist_freq} Hz.")

        min_n = self.sfreq // 2
        if self.window_size < min_n:
            raise ValueError(f"window_size must be at least {min_n} for lowest freqs.")

        stride = self.stride
        if self.stride is None:
            stride = self.window_size
        elif not (1 <= stride <= self.window_size):
            raise ValueError(f"stride must be between 1 and {self.window_size}.")

        n_overlap = self.window_size - stride
        # next power of 2, for FFT efficiency
        n_fft = int(2 ** np.ceil(np.log2(self.window_size)))
        freq_res = self.sfreq / n_fft
        max_band_freq = self.FREQ_BANDS["gamma"][1]
        powers, freqs = psd_array_welch(
            X,
            sfreq=self.sfreq,
            fmin=self.FREQ_BANDS["delta"][0],
            # Include a bin above the final band edge when interpolation needs it.
            fmax=min(self.sfreq / 2, max_band_freq + freq_res),
            n_fft=n_fft,
            n_overlap=n_overlap,
            n_per_seg=self.window_size,
            n_jobs=n_jobs,
            average=None,
            window=self.window_function,
            verbose="error",
        )

        band_powers = np.zeros(shape=(len(self.FREQ_BANDS), powers.shape[-1]))
        for band_idx, (min_freq, max_freq) in enumerate(self.FREQ_BANDS.values()):
            interior_mask = np.logical_and(freqs > min_freq, freqs < max_freq)
            band_freqs = np.concatenate(([min_freq], freqs[interior_mask], [max_freq]))
            band_values = np.concatenate(
                (
                    self._power_at_frequency(powers, freqs, min_freq)[:, None, :],
                    powers[:, interior_mask, :],
                    self._power_at_frequency(powers, freqs, max_freq)[:, None, :],
                ),
                axis=1,
            )
            # integrate over frequencies, average over channels
            band_powers[band_idx, :] = simpson(band_values, x=band_freqs, axis=1).mean(
                axis=0
            )

        if self.relative:
            band_powers /= band_powers.sum(axis=0)

        return band_powers

    @staticmethod
    def _power_at_frequency(powers, freqs, frequency):
        """Return PSD values at a frequency, interpolating between FFT bins."""
        right_idx = np.searchsorted(freqs, frequency)
        if right_idx < len(freqs) and np.isclose(freqs[right_idx], frequency):
            return powers[:, right_idx, :]

        left_idx = right_idx - 1
        weight = (frequency - freqs[left_idx]) / (freqs[right_idx] - freqs[left_idx])
        return powers[:, left_idx, :] + weight * (
            powers[:, right_idx, :] - powers[:, left_idx, :]
        )
