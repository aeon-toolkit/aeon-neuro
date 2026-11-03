"""Band pass transformations."""

from numbers import Real

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation import check_n_jobs
from mne.filter import filter_data

__all__ = ["BandPassSeriesTransformer"]


class BandPassSeriesTransformer(BaseSeriesTransformer):
    """Band pass filter transformer.

    Keeps frequencies between `l_freq` and `h_freq` and attenuates the rest,
    returning a series of the same shape. Setting `l_freq=None` gives a lowpass
    filter and `h_freq=None` gives a highpass filter. Filtering is done with
    `mne.filter.filter_data`.

    Parameters
    ----------
    sfreq : int or float
        Sampling frequency of the input in Hz. Must be finite and positive.
    l_freq : float or None, optional
        Lower cutoff frequency in Hz, must be below `sfreq / 2`. If None,
        no highpass is applied, by default 0.5.
    h_freq : float or None, optional
        Upper cutoff frequency in Hz, must be below `sfreq / 2`. If None,
        no lowpass is applied, by default 40.
    method : {"fir", "iir"}, optional
        Filter design. "fir" is zero-phase FIR, "iir" is a zero-phase
        Butterworth filter, by default "fir".
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 1.

    Raises
    ------
    ValueError
        If `sfreq` is not a finite positive number.
        If both `l_freq` and `h_freq` are None.
        If `l_freq` is not positive or not below `sfreq / 2`.
        If `h_freq` is not below `sfreq / 2`.
        If `l_freq` is not less than `h_freq`.
        If `method` is not "fir" or "iir".
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        sfreq,
        l_freq=0.5,
        h_freq=40,
        method="fir",
        n_jobs=1,
    ):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        self.n_jobs = n_jobs
        super().__init__(axis=1)  # (n_channels, n_timepoints)

    def _transform(self, X, y=None):
        """Bandpass filter the input series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Input time series.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_channels, n_timepoints)
            Filtered time series.
        """
        # checks
        n_jobs = check_n_jobs(self.n_jobs)

        if (
            not isinstance(self.sfreq, Real)
            or isinstance(self.sfreq, bool)
            or not np.isfinite(self.sfreq)
            or self.sfreq <= 0
        ):
            raise ValueError("sfreq must be a finite positive number.")

        nyquist_freq = self.sfreq / 2

        if self.l_freq is None and self.h_freq is None:
            raise ValueError("At least one of l_freq or h_freq must be set.")

        if self.l_freq is not None and self.l_freq <= 0:
            raise ValueError("l_freq must be greater than 0 Hz.")

        if self.l_freq is not None and self.l_freq >= nyquist_freq:
            raise ValueError(f"l_freq must be less than {nyquist_freq} Hz.")

        if self.h_freq is not None and self.h_freq >= nyquist_freq:
            raise ValueError(f"h_freq must be less than {nyquist_freq} Hz.")

        if (
            self.l_freq is not None
            and self.h_freq is not None
            and self.l_freq >= self.h_freq
        ):
            raise ValueError("l_freq must be less than h_freq.")

        if self.method not in ("fir", "iir"):
            raise ValueError("method must be 'fir' or 'iir'.")

        filtered_data = filter_data(
            X.astype(np.float64),
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method=self.method,
            n_jobs=n_jobs,
            verbose="error",
        )

        return filtered_data

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set to return.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
        """
        return {"sfreq": 100}
