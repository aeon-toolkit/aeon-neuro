"""Load BIDS recordings through the verified source adapter."""

import numpy as np

from aeon_neuro.datasets._bids_input import inspect_bids_eeg, open_bids_eeg_recording
from aeon_neuro.datasets._dataset_info import EEGDatasetInfo


def load_bids_dataset(
    root, *, subjects=None, sessions=None, tasks=None, acqs=None, runs=None
):
    """Load supported BIDS recordings with their inspected source metadata.

    Parameters
    ----------
    root : path-like
        Local BIDS root containing EDF/BDF sources. No acquisition occurs.
    subjects, sessions, tasks, acqs, runs : str, list of str or None, default=None
        Exact BIDS selections passed to ``inspect_bids_eeg``.

    Returns
    -------
    X : np.ndarray or list of np.ndarray
        Complete recordings in source order. Equal shapes, channel names/order
        and sampling rates produce a 3D array; otherwise a list of 2D arrays.
    info : EEGDatasetInfo
        Aligned metadata, source facts and events, without inferring a target.

    Notes
    -----
    This explicit operation reads complete signals into memory and closes each
    owned source handle on success or failure. It reuses inspected snapshots and
    the source opener. It does not modify the legacy ``load_bids_data`` loader,
    expand companion-file support or apply preprocessing/montage corrections.
    """
    info = EEGDatasetInfo(
        inspect_bids_eeg(
            root,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            acqs=acqs,
            runs=runs,
        )
    )
    recordings = []
    for descriptor in info.metadata:
        with open_bids_eeg_recording(root, descriptor) as raw:
            recordings.append(raw.get_data())
    info.validate_layout(recordings)
    reference = info.recordings[0]
    compatible = all(
        values.shape == recordings[0].shape
        and recording.channel_names == reference.channel_names
        and recording.sfreq == reference.sfreq
        for values, recording in zip(recordings, info.recordings, strict=True)
    )
    return (np.stack(recordings) if compatible else recordings), info
