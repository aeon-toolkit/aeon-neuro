"""Epoching transformations."""

import numpy as np
from aeon.transformations.series.base import BaseSeriesTransformer

__all__ = ["EpochSeriesTransformer"]


class EpochSeriesTransformer(BaseSeriesTransformer):
    """Epoch series transformer.

    Transform a series into a collection of smaller segments,
    epoched by time or percent.

    Parameters
    ----------
    sfreq : int or float, optional
        Sampling frequency in Hz, by default 1.0.
    epoch_size : int or float, optional
        Length of each epoch in milliseconds, by default None.
    percent : float, optional
        Percent of the total length for each epoch, by default None.

    Raises
    ------
    ValueError
        If `epoch_size` or `percent` are not provided.
        If both `epoch_size` and `percent` are provided.
        If `epoch_size` is shorter than one sample at `sfreq` (transform time).
        If an epoch covers fewer than one sample or more than the series
        length (transform time).

    Examples
    --------
    >>> import numpy as np
    >>> from aeon_neuro.transformations.series import EpochSeriesTransformer
    >>> X = np.zeros((1, 1000))
    >>> t = EpochSeriesTransformer(sfreq=100, epoch_size=100)
    >>> Xt = t.fit_transform(X)
    >>> Xt.shape
    (100, 1, 10)
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, sfreq=1.0, epoch_size=None, percent=None):
        super().__init__(axis=1)  # (n_channels, n_timepoints)
        self.sfreq = sfreq
        self.epoch_size = epoch_size
        self.percent = percent
        if epoch_size is None and percent is None:
            raise ValueError("Either 'epoch_size' or 'percent' must be provided.")
        if epoch_size is not None and percent is not None:
            raise ValueError(
                "Only one of 'epoch_size' or 'percent' should be provided."
            )

    def _transform(self, X, y=None):
        """Transform the input series to epoched collection.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Input time series.
        y : None
            Ignored for interface compatibility, by default None

        Returns
        -------
        np.ndarray of shape (n_epochs, n_channels, n_timepoints_per_epoch)
            Collection of the epoched series.
        """
        n_channels, n_timepoints = X.shape

        if self.epoch_size is not None:
            n_timepoints_per_epoch = int((self.epoch_size / 1000) * self.sfreq)
        elif self.percent is not None:
            n_timepoints_per_epoch = int(n_timepoints * (self.percent / 100))
        else:  # unreachable: __init__ requires one of the two
            raise ValueError("Either 'epoch_size' or 'percent' must be provided.")

        if n_timepoints_per_epoch < 1:
            if self.epoch_size is not None:
                raise ValueError(
                    f"'epoch_size' of {self.epoch_size} ms is shorter than one "
                    f"sample at 'sfreq' of {self.sfreq} Hz."
                )
            raise ValueError(
                "Each epoch must cover at least one sample: "
                f"got {n_timepoints_per_epoch} timepoints per epoch."
            )
        if n_timepoints_per_epoch > n_timepoints:
            raise ValueError(
                f"Each epoch covers {n_timepoints_per_epoch} timepoints, "
                f"more than the series length of {n_timepoints}."
            )

        n_epochs = n_timepoints // n_timepoints_per_epoch
        X_transformed = np.zeros((n_epochs, n_channels, n_timepoints_per_epoch))

        for epoch in range(n_epochs):
            idx = epoch * n_timepoints_per_epoch
            X_transformed[epoch] = X[:, idx : idx + n_timepoints_per_epoch]

        return X_transformed


def epoch_dataset(X, y, sfreq, epoch_size):
    """Transform collection and labels into epoched collection by time.

    Parameters
    ----------
    X : list or np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Input time series collection.
    y : list or np.ndarray of shape (n_cases, )
        Case labels.
    sfreq : int or float
        Sampling frequency in Hz (samples per second).
    epoch_size : int of float
        Desired length of each epoch in milliseconds.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - Collection of the epoched series,
          of shape (n_epochs_total, n_channels, n_timepoints_per_epoch)
        - Repeated labels corresponding to each epoch, of shape (n_epochs_total, )

    Notes
    -----
    Raises if any case is shorter than one epoch; such cases were previously
    skipped silently.
    """
    transformer = EpochSeriesTransformer(sfreq=sfreq, epoch_size=epoch_size)
    epoched_data = []
    new_labels = []
    for i in range(len(X)):
        new_data = transformer.fit_transform(X[i])
        for j in new_data:  # type: ignore
            epoched_data.append(j)
            new_labels.append(y[i])
    return np.asarray(epoched_data), np.asarray(new_labels)
