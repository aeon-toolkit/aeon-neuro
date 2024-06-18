"""Estimate power spectral density using Welch's method."""

__maintainer__ = [""]
__all__ = ["PSDWelchTransformer"]

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer
from scipy.signal import welch


class PSDWelchTransformer(BaseSeriesTransformer):
    """Estimate power spectral density using Welch's method.

    Provides a simple wrapper around ``scipy.signal.welch``.

    Parameters
    ----------
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    axis : int, optional
        Axis along which the periodogram is computed; the default is
        over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. Defaults to 'mean'.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    If `noverlap` is 0, this method is equivalent to Bartlett's method
    [2]_.

    Examples
    --------
    >>> from aeon_neuro._wip.transformations.series._psd_welch import PSDWelchTransformer
    >>> from aeon.datasets import load_arrow_head
    >>> import numpy as np
    >>> X = load_arrow_head()
    >>> X_sample = X[0].squeeze()[:3,:]
    >>> transformer = PSDWelchTransformer(nperseg=100)
    >>> X_hat = transformer.fit_transform(X_sample)

    References
    ----------
    # noqa: E501
    .. [1] https://docs.scipy.org/doc//scipy-1.12.0/reference/generated/scipy.signal.welch.html
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        fs=1.0,
        window="hann",
        nperseg=None,
        noverlap=None,
        nfft=None,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="mean",
    ):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.average = average
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X: np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        freqs : np.ndarray
            Array of sample frequencies corresponding to the PSD estimates.
        Xt : Power spectral density or power spectrum of x.
            shape (n_channels, n_frequencies)
        """
        n_channels = X.shape[0]
        freqs = []
        Xt = []
        for i in range(n_channels):
            freq, Pxx = welch(
                X[i],
                fs=self.fs,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                detrend=self.detrend,
                return_onesided=self.return_onesided,
                scaling=self.scaling,
                axis=self.axis,
                average=self.average,
            )
            freqs.append(freq)
            Xt.append(Pxx)
        freqs = np.array(freqs)
        Xt = np.array(Xt)
        # return freqs, Xt
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {
            "fs": 1.0,
            "window": "hann",
            "nperseg": None,
            "noverlap": None,
            "nfft": None,
            "detrend": "constant",
            "return_onesided": True,
            "scaling": "density",
            "axis": -1,
            "average": "mean",
        }

        return params
