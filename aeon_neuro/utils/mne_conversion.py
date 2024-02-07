import mne
from aeon_neuro.utils.data_io import load_auxiliary_info
import pyprep
import numpy as np

def narray_to_mne(series,sfreq,channels=None):
    n_dimensions,n_timepoints = series.shape
    info = mne.create_info(channels,ch_types=["eeg"]*n_dimensions,sfreq=sfreq)
    raw = mne.io.RawArray(series,info)
    return raw

