"""Cross spectral matrix transformer."""

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer
from mne.time_frequency.csd import _csd_fourier, _vector_to_sym_mat
from scipy.fft import rfftfreq


class CrossSpectralMatrix(BaseSeriesTransformer):
    """Cross spectral (density) matrix transformer.

    Estimate the pairwise cross spectral matrix between channels in the frequency
    domain -- i.e., between power spectral densities. The result is a hermitian
    positive definite (HPD) complex-valued matrix of shape (n_channels, n_channels).
    The magnitude of the HPD matrix is real-valued.
    Matrices are computed as a function of frequency using the short-time fourier
    algorithm, then averaged over the specified frequency range
    (by default 0-60Hz for EEG).

    Parameters
    ----------
    sfreq : int or float, optional
        Sampling frequency in Hz, by default 120.
    fmin : int or float, optional
        Minimum frequency of interest in Hz, by default 0.
    fmax : int or float, optional
        Maximum frequency of interest in Hz, by default 60.
    magnitude : bool, optional
        If True, return the magnitude of the cross spectral matrix (real-valued).
        If False, return the complex-valued matrix, by default False.

    Examples
    --------
    >>> from aeon_neuro._wip.transformations.series._power_spectrum import (
    ...     CrossSpectralMatrix)
    >>> import numpy as np
    >>> n_channels, n_timepoints = 5, 360
    >>> X = np.random.standard_normal(size=(n_channels, n_timepoints))
    >>> transformer = CrossSpectralMatrix()
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape == (n_channels, n_channels)
    True
    >>> np.iscomplexobj(X_transformed)
    True
    >>> X_transformed = transformer.set_params(magnitude=True).fit_transform(X)
    >>> np.iscomplexobj(X_transformed)
    False
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, sfreq=120, fmin=0, fmax=60, magnitude=False):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.magnitude = magnitude
        super().__init__(axis=1)  # (n_channels, n_timepoints)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # frequency mask
        n_timepoints = X.shape[-1]
        n_fft = n_timepoints
        freq_fft = rfftfreq(n_fft, 1.0 / self.sfreq)
        freq_mask = (freq_fft > 0) & (freq_fft >= self.fmin) & (freq_fft <= self.fmax)

        if freq_mask.sum() == 0:
            raise ValueError(
                f"FFT cannot be computed for sfreq={self.sfreq} \
                within the frequency range={self.fmin}-{self.fmax}. \
                Please increase the frequency range."
            )

        # CSD matrix by frequency (stored as vectors)
        X_vector = _csd_fourier(
            X, sfreq=self.sfreq, n_times=n_timepoints, freq_mask=freq_mask, n_fft=n_fft
        )
        # average over frequencies
        # convert vector to symmetric matrix of shape (n_channels, n_channels)
        X_matrix = _vector_to_sym_mat(X_vector.mean(axis=1))

        if self.magnitude:
            return np.abs(X_matrix)
        else:
            return X_matrix
