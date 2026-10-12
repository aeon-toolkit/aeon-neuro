"""Simple example of loading data from BIDS format."""
import os
import re
import mne
import numpy as np
from sklearn.model_selection import train_test_split

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

def load_bids_data(path="../", 
                   subjects = None, sessions = None, tasks = None, acqs = None, runs = None, 
                   include_channels = None, exclude_channels = None,
                   l_freq = 0.5, h_freq = 100,
                   train_ratio = 0.8, 
                   formatts=[".vhdr", ".bdf", ".edf"]
                   ):
    """
        Loads EEG recordings from a BIDS-formatted dataset, optionally filters
        subjects, sessions, tasks, acquisitions, and runs, and selects or excludes
        specified channels. The recordings are band-pass filtered and segmented
        into trials. Trials are labeled according to their task or resting-state
        condition, and the resulting dataset is split into training and test sets.

        Parameters
        ----------
        path : str, default="../"
            Path to the root directory of the BIDS dataset.
        subjects : list of str or None, default=None
            Subject identifiers to load, excluding the ``"sub-"`` prefix.
            If None, all available subjects are considered.
        sessions : list of str or None, default=None
            Session identifiers to load, excluding the ``"ses-"`` prefix.
            If None, all available sessions are considered. If no sessions
            exist, the subject-level EEG directory is used.
        tasks : list of str or None, default=None
            Task identifiers to include. If None, all tasks are considered.
        acqs : list of str or None, default=None
            Acquisition identifiers to include. If None, all acquisitions
            are considered.
        runs : list of str or None, default=None
            Run identifiers to include. If None, all runs are considered.
        include_channels : list of str or None, default=None
            Channel names to retain. If None, all channels are retained.
        exclude_channels : list of str or None, default=None
            Channel names to exclude from the recordings. If None, no channels
            are excluded.
        l_freq : float or None, default=0.5
            Lower cutoff frequency, in Hz, for the band-pass filter.
            If None, no high-pass filtering is applied.
        h_freq : float or None, default=100
            Upper cutoff frequency, in Hz, for the band-pass filter.
            If None, no low-pass filtering is applied.
        train_ratio : float, default=0.8
            Proportion of the dataset to use for training. The remaining
            samples are used for testing. The split is stratified by class.
        formatts : list of str, default=[".vhdr", ".bdf", ".edf"]
            File extensions to consider when searching for EEG recordings.
            Supported formats are BrainVision (``.vhdr``), BioSemi (``.bdf``),
            and European Data Format (``.edf``).

        Returns
        -------
        X_train : np.ndarray
            Training EEG data, with shape
            (n_train_samples, n_channels, n_times).
        y_train : np.ndarray
            Labels for the training samples, containing task or resting-state
            class labels.
        X_test : np.ndarray
            Test EEG data, with shape
            (n_test_samples, n_channels, n_times).
        y_test : np.ndarray
            Labels for the test samples, containing task or resting-state
            class labels.
        """


    X = []
    y = []

    # Gets the subjects
    if subjects is None:
        subjects = [x[4:] for x in os.listdir(path) if x.startswith("sub")]

    for subject in subjects:
        subject_path = os.path.join(path, "sub-" + subject)

        if not os.path.isdir(subject_path):
            continue

        # Find sessions only if none are specified
        if sessions is None:
            available_sessions = [x[4:] for x in os.listdir(subject_path) if x.startswith("ses-")]
        else:
            available_sessions = sessions

        # Choose EEG directories
        if available_sessions:
            eeg_dirs = [os.path.join(subject_path, "ses-" + session, "eeg") for session in available_sessions]
        elif sessions is None:
            eeg_dirs = [os.path.join(subject_path, "eeg")]
        else:
            continue

        # Find and load recordings
        for eeg_dir in eeg_dirs:
            if not os.path.isdir(eeg_dir):
                continue

            for filename in os.listdir(eeg_dir):
                if not filename.lower().endswith(tuple(formatts)):
                    continue

                # Only load files ending with _eeg.
                if not re.search(r"_eeg\.[^.]+$", filename):
                    continue

                # Extract and filter task, acq, and run
                task_match = re.search(r"_task-(.*?)_", filename)
                acq_match = re.search(r"_acq-(.*?)_", filename)
                run_match = re.search(r"_run-(.*?)_", filename)

                if tasks is not None:
                    if not task_match or task_match.group(1) not in tasks:
                        continue

                if acqs is not None:
                    if not acq_match or acq_match.group(1) not in acqs:
                        continue

                if runs is not None:
                    if not run_match or run_match.group(1) not in runs:
                        continue

                filepath = os.path.join(eeg_dir, filename)
                ext = os.path.splitext(filename)[1].lower()

                # Add more extensions if needed
                if ext == ".vhdr":
                    raw = mne.io.read_raw_brainvision(filepath)
                elif ext == ".bdf":
                    raw = mne.io.read_raw_bdf(filepath)
                elif ext == ".edf":
                    raw = mne.io.read_raw_edf(filepath)
                else:
                    continue

                raw.load_data()

                # Pick channels if given, exclude channels if given
                if include_channels is not None:
                    raw.pick(include_channels)
                if exclude_channels is not None:
                    raw.drop_channels(exclude_channels)

                # Filters channels between l_freq and h_freq, defaults to 0.5 and 100 respectivly
                raw = raw.copy().filter(l_freq, h_freq)

                # Figure out how to get class labels, can never always be task matching
                if task_match and task_match.group(1) == "task":
                    for trial in raw.annotations:
                        trial_data = raw.copy().crop(tmin=trial["onset"] - 0.2, tmax=trial["onset"] + 0.800)
                        data = trial_data.get_data()
                        X.append(data[:, :1000])
                        y.append("task")
                else:
                    for timepoint in np.arange(30, 90, 1.5):
                        trial_data = raw.copy().crop(tmin=timepoint, tmax=timepoint + 1)
                        data = trial_data.get_data()
                        X.append(data[:, :1000])
                        y.append("rest")

    #Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_ratio, stratify = y)
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    return X_train, y_train, X_test, y_test
