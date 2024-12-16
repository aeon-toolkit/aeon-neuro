"""Downsample series by frequency."""

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer

__all__ = ["DownsampleCollectionTransformer"]


class DownsampleCollectionTransformer(BaseCollectionTransformer):
    """Downsample the time dimension of a collection of time series.

    Parameters
    ----------
    downsample_by : str, optional
        The method to downsample by, either "frequency" or "proportion",
        by default "frequency".
    source_sfreq : int or float, optional
        The source sampling frequency in Hz.
        Required if `downsample_by = "frequency"`, by default 2.0.
    target_sfreq : int or float, optional
        The target sampling frequency in Hz.
        Required if `downsample_by = "frequency"`, by default 1.0.
    proportion : float, optional
        The proportion between 0-1 to downsample by.
        Required if `downsample_by = "proportion"`, by default None.

    Raises
    ------
    ValueError
        If `downsample_by` is not "frequency" or "proportion".
        If `source_sfreq` < `target_sfreq` when `downsample_by = "frequency"`.
        If `proportion` is not between 0-1 when `downsample_by = "proportion"`.
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        downsample_by="frequency",
        source_sfreq=2.0,
        target_sfreq=1.0,
        proportion=None,
    ):
        # checks
        if downsample_by not in ["frequency", "proportion"]:
            raise ValueError('downsample_by must be either "frequency" or "proportion"')

        if downsample_by == "frequency":
            if source_sfreq is None or target_sfreq is None:
                raise ValueError("source_sfreq and target_sfreq must be provided")
            if source_sfreq < target_sfreq:
                raise ValueError("source_sfreq must be > target_sfreq")

        if downsample_by == "proportion":
            if proportion is None or not (0 < proportion < 1):
                raise ValueError("proportion must be provided and between 0-1.")

        super().__init__()
        self.downsample_by = downsample_by
        self.source_sfreq = source_sfreq
        self.target_sfreq = target_sfreq
        self.proportion = proportion

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
        if self.downsample_by == "frequency":
            step = int(self.source_sfreq / self.target_sfreq)
        elif self.downsample_by == "proportion":
            step = int(1 / (1 - self.proportion))

        X_downsampled = []
        for x in X:
            n_timepoints = x.shape[-1]
            indices = np.arange(0, n_timepoints, step)
            X_downsampled.append(x[:, indices])

        if isinstance(X, np.ndarray):
            return np.asarray(X_downsampled)
        else:
            return X_downsampled
