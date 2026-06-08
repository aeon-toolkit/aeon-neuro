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
    percent : _type_, optional
        Percent of the total length for each epoch, by default None.

    Raises
    ------
    ValueError
        If `epoch_size` or `percent` are not provided.
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

        if self.epoch_size:
            n_timepoints_per_epoch = int((self.epoch_size / 1000) * self.sfreq)
        elif self.percent:
            n_timepoints_per_epoch = int(n_timepoints * (self.percent / 100))

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
