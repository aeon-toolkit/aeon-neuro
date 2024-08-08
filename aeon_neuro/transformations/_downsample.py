"""Downsample series by frequency."""

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer

__all__ = ["DownsampleCollectionTransformer"]


class DownsampleCollectionTransformer(BaseCollectionTransformer):
    """Downsample the time dimension of a collection of time series.

    Parameters
    ----------
    source_sfreq : int or float
        The source/input sampling frequency in Hz.
    target_sfreq : int or float
        The target/output sampling frequency in Hz.

    Raises
    ------
    ValueError
        If source_sfreq < target_sfreq.
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(self, source_sfreq=2.0, target_sfreq=1.0):
        super().__init__()
        if source_sfreq < target_sfreq:
            raise ValueError("source_sfreq must be >= target_sfreq")
        self.source_sfreq = source_sfreq
        self.target_sfreq = target_sfreq

    def _transform(self, X, y=None):
        """Transform the input collection to downsampled collection.

        Parameters
        ----------
        X : list or np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Input time series collection where n_timepoints can vary over cases.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        list of 2D numpy arrays of shape [(n_channels, n_timepoints_downsampled), ...]
        or np.ndarray of shape (n_cases, n_channels, n_timepoints_downsampled)
            Downsampled time series collection.
        """
        step = int(self.source_sfreq / self.target_sfreq)
        X_downsampled = []
        for x in X:
            n_timepoints = x.shape[-1]
            indices = np.arange(0, n_timepoints, step)
            X_downsampled.append(x[:, indices])

        if isinstance(X, np.ndarray):
            return np.asarray(X_downsampled)
        else:
            return X_downsampled
