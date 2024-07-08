"""Epoching transformations."""

import numpy as np


def epoch_series_by_percentage(series, percent):
    """Transform a series into a collection of smaller segments (epoched by percent).

    Parameters
    ----------
    series : np.ndarray of shape (n_channels, n_timepoints)
        Input time series.
    percent : int or float
        Percent of the total length for each epoch.

    Returns
    -------
    np.ndarray of shape (n_epochs, n_channels, n_timepoints_per_epoch)
        Collection of the epoched series.
    """
    n_channels, n_timepoints = series.shape
    instances_per_epoch = int(n_timepoints * (percent / 100))
    epoched_data = []
    index = 0
    while index + instances_per_epoch <= n_timepoints:
        epoch = series[:, index : index + instances_per_epoch]
        epoched_data.append(epoch)
        index += instances_per_epoch
    return np.asarray(epoched_data)


def epoch_series_by_time(series, sfreq, epoch_size):
    """Transform a series into a collection of smaller segments (epoched by time).

    Parameters
    ----------
    series : np.ndarray of shape (n_channels, n_timepoints)
        Input time series.
    sfreq : int or float
        Sampling frequency in Hz (samples per second).
    epoch_size : int or float
        Desired length of each epoch in milliseconds.

    Returns
    -------
    np.ndarray of shape (n_epochs, n_channels, n_timepoints_per_epoch)
        Collection of the epoched series.
    """
    n_channels, n_timepoints = series.shape
    instances_per_epoch = int((epoch_size / 1000) * sfreq)
    epoched_data = []
    index = 0
    while index + instances_per_epoch <= n_timepoints:
        epoch = series[0:n_channels, index : index + instances_per_epoch]
        epoched_data.append(epoch)
        index += instances_per_epoch
    return np.asarray(epoched_data)


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
    epoched_data = []
    new_labels = []
    for i in range(len(X)):
        new_data = epoch_series_by_time(X[i], sfreq, epoch_size)
        for j in new_data:
            epoched_data.append(j)
            new_labels.append(y[i])
    return np.asarray(epoched_data), np.asarray(new_labels)
