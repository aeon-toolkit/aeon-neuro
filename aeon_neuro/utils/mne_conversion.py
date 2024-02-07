import mne
import numpy as np
import pyprep

from aeon_neuro.utils.data_io import load_auxiliary_info


def narray_to_mne(series, sfreq, channels=None):
    n_dimensions, n_timepoints = series.shape
    info = mne.create_info(channels, ch_types=["eeg"] * n_dimensions, sfreq=sfreq)
    raw = mne.io.RawArray(series, info)
    return raw
