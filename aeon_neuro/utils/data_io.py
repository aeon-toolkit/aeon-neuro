"""Read EEG recordings and auxiliary dataset information."""

import json
import warnings
from pathlib import Path

import mne
import numpy as np


def load_eeg_to_mne(path, *, preload=False, **reader_kwargs):
    """Read a native EEG recording without analysis preprocessing.

    Parameters
    ----------
    path : str or os.PathLike
        BrainVision .vhdr, EDF/EDF+ .edf, BioSemi BDF/BDF+ .bdf or continuous
        EEGLAB .set file. Keep any linked signal and marker files alongside it.
    preload : bool or str, default=False
        Whether to load samples into memory, or an MNE memory-map filename.
        Deferred reading lets MNE-BIDS copy supported native files unchanged.
        Some EEGLAB files with embedded samples are necessarily preloaded.
    **reader_kwargs : dict
        Options passed to the selected MNE reader, such as ``eog``, ``misc`` or
        ``uint16_codec``. Available options depend on the recording format.

    Returns
    -------
    raw : mne.io.BaseRaw
        All channels, acquisition metadata and annotations decoded by MNE.

    Raises
    ------
    ValueError
        If the filename extension is unsupported, or the MNE reader rejects it.

    Notes
    -----
    No filtering, referencing, channel selection or epoching is applied. MNE
    reader behaviour still applies: mixed-rate EDF/BDF channels are represented
    at one sampling rate, and discontinuities are not independently validated.
    Status-channel triggers remain in the signal; use ``mne.find_events`` with
    acquisition-appropriate settings when an explicit event table is needed.
    Native reading does not import existing BIDS sidecars or repair metadata.
    """
    readers = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
    }
    extension = Path(path).suffix.lower()
    if extension not in readers:
        raise ValueError(
            "Supported EEG filenames are .vhdr, .edf, .bdf and continuous .set; "
            "use a format-specific MNE reader for other recordings."
        )
    # Reuse native decoders so calibration, companions and reader errors stay in MNE.
    return readers[extension](path, preload=preload, **reader_kwargs)


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
