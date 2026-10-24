"""Simple example of loading data from BIDS format."""

import os

import mne
import numpy as np
from mne_bids import find_matching_paths, read_raw_bids


def load_basic_classification_problem(path="../"):
    """Load data and format to classification problem.

    Toy EEG recorded by Aiden Rushbrooke and stored in BIDS format to use as an example
    for the aeon_neuro package. Loads the data, pre-processes, segments into instances.

    Parameters
    ----------
    path : str
        Relative path to the directory "example_raw_eeg".

    Returns
    -------
    X_train : np.ndarray
        First 20 recordings of shape (240,32,1000)
    y_train : np.ndarray
        Train labels
    X_test : np.ndarray
        Last 20 recordings of shape (240,32,1000)
    y_test : np.ndarray
        Labels for the last 20 recordings
    """
    data_path = path + "example_raw_eeg/basic_classification_task"
    tasks = ["task", "rest"]
    runs = ["01", "02", "03", "04", "05", "06"]

    X = []
    y = []

    # Load each data file from BIDS format
    for task in tasks:
        for run in runs:
            raw = mne.io.read_raw_brainvision(
                data_path
                + "/sub-01/ses-01/eeg/sub-01_ses-01_task-"
                + task
                + "_run-"
                + run
                + "_eeg.vhdr"
            )
            raw.load_data()
            raw.drop_channels(["ACC_X", "ACC_Y", "ACC_Z", "DeviceTrigger"])
            raw = raw.copy().filter(l_freq=0.5, h_freq=100)
            if task == "task":
                for trial in raw.annotations:
                    trial_data = raw.copy().crop(
                        tmin=trial["onset"] - 0.2, tmax=trial["onset"] + 0.800
                    )
                    data = trial_data.get_data()
                    X.append(data[:, :1000])
                    y.append("task")
            else:
                for timepoint in np.arange(30, 90, 1.5):
                    trial_data = raw.copy().crop(tmin=timepoint, tmax=timepoint + 1)
                    data = trial_data.get_data()
                    X.append(data[:, :1000])
                    y.append("rest")

    X_train, X_test = [], []
    y_train, y_test = [], []
    n_instances, _, _ = np.shape(X)
    subject_instances = int(n_instances / 40)

    # Reformat data into a classification problem
    for i in range(subject_instances):
        for j in range(40):
            loc = i * 40 + j
            if j < 20:
                X_train.append(X[loc])
                y_train.append(y[loc])
            else:
                X_test.append(X[loc])
                y_test.append(y[loc])

    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    return X_train, y_train, X_test, y_test


def load_bids_data(
    path="../",
    subjects=None,
    sessions=None,
    tasks=None,
    acqs=None,
    runs=None,
    formatts=(".vhdr", ".bdf", ".edf"),
):
    """Load EEG recordings from a BIDS dataset into an aeon-compatible collection.

    Recordings are loaded using MNE and filtered by the specified subject,
    session, task, acquisition, and run identifiers.

    Returns a 3D NumPy array if all recordings have the same shape,
    same sampling frequency and same channel names and order;
    otherwise, returns a list of 2D NumPy arrays.

    Parameters
    ----------
    path : str, default="../"
        Path to the root directory of the BIDS dataset.
    subjects : list of str or None, default=None
        Subject identifiers, excluding the "sub-" prefix.
    sessions : list of str or None, default=None
        Session identifiers, excluding the "ses-" prefix.
        If None, discover available sessions automatically.
    tasks : list of str or None, default=None
        Task identifiers to include. If None, include all tasks.
    acqs : list of str or None, default=None
        Acquisition identifiers to include. If None, include all acquisitions.
    runs : list of str or None, default=None
        Run identifiers to include. If None, include all runs.
    formatts : tuple of str, default=(".vhdr", ".bdf", ".edf")
        EEG file extensions to consider.

    Returns
    -------
    X : np.ndarray or list of np.ndarray
        EEG recordings. A 3D array of shape
        (n_recordings, n_channels, n_times) is returned when all
        recordings have the same shape, sampling rate, channel names and order.
        Otherwise, returns a list of 2D arrays, each with shape (n_channels, n_times).

    metadata : list of dict
        Metadata for each recording, in the same order as X.
        Each entry identifies the source recording, channel names,
        sampling frequency, subject, session, task, acquisition and run.
    """
    path = os.fspath(path)
    formatts = tuple(ext.lower() for ext in formatts)

    # Discover BIDS recordings
    bids_paths = find_matching_paths(
        path,
        subjects=subjects,
        sessions=sessions,
        tasks=tasks,
        acquisitions=acqs,
        runs=runs,
        datatypes=["eeg"],
        extensions=list(formatts),
        check=False,
    )

    recordings = []
    metadata = []

    for bids_path in bids_paths:
        # Load BIDS
        raw = read_raw_bids(bids_path, verbose=False)

        recordings.append(raw.get_data())

        metadata.append(
            {
                "filepath": os.path.relpath(bids_path.fpath, path),
                "filename": bids_path.fpath.name,
                "subject": bids_path.subject,
                "session": bids_path.session,
                "task": bids_path.task,
                "acq": bids_path.acquisition,
                "run": bids_path.run,
                "channel_names": list(raw.ch_names),
                "sampling_frequency": float(raw.info["sfreq"]),
            }
        )

    if not recordings:
        raise ValueError(
            f"No EEG recordings found in {path!r} "
            "matching the specified selection criteria."
        )

    reference_channels = metadata[0]["channel_names"]
    reference_sfreq = metadata[0]["sampling_frequency"]
    reference_shape = recordings[0].shape

    compatible = all(
        recording.shape == reference_shape
        and item["channel_names"] == reference_channels
        and np.isclose(item["sampling_frequency"], reference_sfreq)
        for recording, item in zip(recordings, metadata)
    )

    if compatible:
        X = np.stack(recordings, axis=0)
    else:
        X = recordings

    return X, metadata
