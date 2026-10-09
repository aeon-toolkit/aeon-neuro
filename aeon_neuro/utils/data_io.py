"""Read EEG recordings and auxiliary dataset information."""

import json
import warnings

import mne
import numpy as np


def load_auxiliary_info(path, dataset_name):
    """Read auxiliary JSON, warning and returning None if it cannot be loaded.

    Parameters
    ----------
    path : str
        Directory prefix, including its trailing separator.
    dataset_name : str
        Auxiliary filename without the .json extension.

    Returns
    -------
    aux_data : object or None
        Decoded JSON value, or None for file access or JSON decoding errors.
    """
    full_path = path + dataset_name + ".json"
    try:
        with open(full_path) as f:
            return json.load(f)
    except (OSError, ValueError) as error:
        # Keep optional metadata failures non-fatal without swallowing interrupts.
        warnings.warn(
            f"Unable to load auxiliary information at {full_path}: {error}",
            stacklevel=2,
        )
        return None


def load_brainvision_to_mne(path, *, preload=False):
    """Load a BrainVision recording while retaining its MNE metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .vhdr file, with its linked signal and marker files available.
    preload : bool or str, default=False
        Whether to load samples into memory. A string specifies an MNE memory-map
        file. The default leaves samples on disk for subsequent BIDS writing.

    Returns
    -------
    raw : mne.io.BaseRaw
        Recording with all channels, sampling information and annotations as
        read by MNE. No filtering, channel selection or epoching is applied.

    Notes
    -----
    Channel types follow MNE's BrainVision reader. Existing BIDS sidecars are
    not read; use MNE-BIDS when their additional metadata is needed.
    """
    return mne.io.read_raw_brainvision(path, preload=preload)


def load_brainvision_to_numpy(path, remove_non_EEG=True):
    """Load a BrainVision recording as a NumPy array.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .vhdr file, with its linked signal and marker files available.
    remove_non_EEG : bool, default=True
        Retain only channels identified as EEG by MNE when True.

    Returns
    -------
    numpy_data : np.ndarray
        Signal values with shape (n_channels, n_timepoints), without metadata.
    """
    # Preserve the array interface while allowing metadata-aware callers to use Raw.
    mne_data = load_brainvision_to_mne(path)
    if remove_non_EEG:
        mne_data = mne_data.pick_types(eeg=True)
    data = mne_data.load_data()
    return np.asarray(data.get_data())
