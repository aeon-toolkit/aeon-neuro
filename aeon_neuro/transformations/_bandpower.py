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
    δ (0 - 4Hz), θ (4 - 7Hz), α (8 - 12Hz), β (13 - 30Hz), and γ (30 - 60Hz).
    Power within each frequency band is estimated over time using windowed FFTs,
    then averaged across channels.

    Parameters
    ----------
    sfreq : int or float
        Sampling frequency in Hz, by default 120.
    n_per_seg : int, optional
        Length of each segment/window in number of timepoints, by default 256.
    window : str, optional
        Windowing function to use. See `scipy.signal.get_window()`
        for a list of available windows, by default "hamming".
    relative : bool, optional
        If True, return the relative power (divide by total power across freq bands).
        If False, return the absolute power in V^2/Hz, by default True.
    n_jobs : int, optional
        Number of jobs to calculate power spectral densities, by default 1.

    Raises
    ------
    ValueError
        If sfreq is too low to capture power within each frequency band.
        If n_per_seg is less than half the sampling frequency.
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    FREQ_BANDS = {
        "delta": (0, 4),
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta": (13, 30),
        "gamma": (30, 60),
    }

    def __init__(
        self,
        sfreq=120,  # 2x60Hz = 120Hz
        n_per_seg=256,  # mne/scipy default, for window=str
        window="hamming",  # mne default
        relative=True,
        n_jobs=1,
    ):
        super().__init__(axis=1)  # (n_channels, n_timepoints)
        nyquist_freq = 2 * self.FREQ_BANDS["gamma"][1]
        if sfreq < nyquist_freq:
            raise ValueError(
                f"Sampling frequency (sfreq) must be at least {nyquist_freq} Hz."
            )
        min_n = sfreq // 2
        if n_per_seg < min_n:
            raise ValueError(f"n_per_seg must be at least {min_n} for lowest freqs.")
        self.sfreq = sfreq
        self.n_per_seg = n_per_seg
        self.window = window
        self.relative = relative
        self.n_jobs = check_n_jobs(n_jobs)

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
        np.ndarray of shape (5_bands, int(n_timepoints/n_per_seg))
            Power within δ, θ, α, β, and γ bands over time.
        """
        # next power of 2, for FFT efficiency
        n_fft = int(2 ** np.ceil(np.log2(self.n_per_seg)))
        powers, freqs = psd_array_welch(
            X,
            sfreq=self.sfreq,
            fmin=self.FREQ_BANDS["delta"][0],
            fmax=self.FREQ_BANDS["gamma"][1],
            n_fft=n_fft,
            n_overlap=0,
            n_per_seg=self.n_per_seg,
            n_jobs=self.n_jobs,
            average=None,
            window=self.window,
        )

        freq_res = freqs[1] - freqs[0]

        band_powers = np.zeros(shape=(len(self.FREQ_BANDS), powers.shape[-1]))
        for band_idx, (min_freq, max_freq) in enumerate(self.FREQ_BANDS.values()):
            freq_mask = np.logical_and(freqs >= min_freq, freqs <= max_freq)
            # integrate over frequencies, average over channels
            band_powers[band_idx, :] = simpson(
                powers[:, freq_mask, :], dx=freq_res, axis=1
            ).mean(axis=0)

        if self.relative:
            band_powers /= band_powers.sum(axis=0)

        return band_powers
