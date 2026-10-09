"""Utilities for converting between numpy arrays and MNE objects."""

import mne


def narray_to_mne(series, sfreq, channels=None):
    """Convert a single EEG series stored as a numpy array to an MNE Raw object.

    Parameters
    ----------
    series : np.ndarray of shape (n_channels, n_timepoints)
        The EEG series.
    sfreq : float
        Sampling frequency in Hz.
    channels : list of str or None, default=None
        Channel names passed to ``mne.create_info``.

    Returns
    -------
    raw : mne.io.RawArray
        The series as an MNE Raw object with all channels typed as EEG.
    """
    n_dimensions, n_timepoints = series.shape
    info = mne.create_info(channels, ch_types=["eeg"] * n_dimensions, sfreq=sfreq)
    raw = mne.io.RawArray(series, info)
    return raw
