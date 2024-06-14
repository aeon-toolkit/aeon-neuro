"""Downsample series by frequency."""

import numpy as np


def downsample_series(X, sfreq, target_sample_rate):
    """Downsample a time series.

    Parameters
    ----------
    X : a numpy array of shape = [n_cases, n_channels, n_timepoints]
        The input time series data to be downsampled.
    sfreq : int
        The original sampling frequency of the time series data.
    target_sample_rate : int
        The target sampling frequency after downsampling.

    Returns
    -------
    downsampled : a numpy array of
        shape = [n_cases, n_channels, updated_timepoints]
        The downsampled time series data.
    """
    new_ratio = int(sfreq / target_sample_rate)
    n_cases, n_channels, n_timepoints = np.shape(X)
    updated_timepoints = int(np.ceil(n_timepoints / new_ratio))
    downsampled_data = np.zeros((n_cases, n_channels, updated_timepoints))
    for i in range(n_cases):
        for j in range(n_channels):
            updated_index = 0
            for k in range(0, n_timepoints, new_ratio):
                downsampled_data[i][j][updated_index] = X[i][j][k]
                updated_index += 1
    return downsampled_data
