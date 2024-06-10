"""downsample_series by frequency."""

import numpy as np


def downsample_series(X, sfreq, target_sample_rate):
    """Downsample a time series.

    Parameters
    ----------
    X : a numpy array of shape = [n_instances, n_dimensions, n_timepoints]
        The input time series data to be downsampled.
    sfreq : int
        The original sampling frequency of the time series data.
    target_sample_rate : int
        The target sampling frequency after downsampling.

    Returns
    -------
    downsampled : a numpy array of
        shape = [n_instances, n_dimensions, updated_timepoints]
        The downsampled time series data.
    """
    new_ratio = int(sfreq / target_sample_rate)
    n_instances, n_dimensions, n_timepoints = np.shape(X)
    updated_timepoints = int(np.ceil(n_timepoints / new_ratio))
    downsampled_data = np.zeros((n_instances, n_dimensions, updated_timepoints))
    for i in range(n_instances):
        for j in range(n_dimensions):
            updated_index = 0
            for k in range(0, n_timepoints, new_ratio):
                downsampled_data[i][j][updated_index] = X[i][j][k]
                updated_index += 1
    return downsampled_data
