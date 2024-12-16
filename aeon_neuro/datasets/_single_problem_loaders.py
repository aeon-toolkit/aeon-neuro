"""Utilities for loading datasets."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["load_kdd_example"]

import mne
import numpy as np
from aeon.datasets import load_classification


def load_kdd_example(split=None, return_metadata=False):
    """Load the preprocessed EEG KDD dataset used in the 2024 SIGKDD Tutorial.

    On time series classification and regression. Slides and code [are available]
    (https://github.com/aeon-toolkit/aeon-tutorials/tree/main/KDD-2024).

    The full dataset is loadable directly from the BIDS files using
    ``load_kdd_full``. This data contains four channels and is downsampled to 100
    time points.


    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        Collection of EEG recordings.
    y: np.ndarray
        The class labels for each EEG in X.

    Notes
    -----
    Number of time points:  100
    Number of channels:     4
    Train cases:            20
    Test cases:             20
    Number of classes:      2   ("task", "rest")
    Details: TBC
    """
    return load_classification(
        extract_path="data/KDD_Example/KDD_Example/",
        name="KDD_MTSC",
        split=split,
        return_metadata=return_metadata,
    )


def load_kdd_full_example(split=None, verbose=False):
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
    time series classification and regression. Slides and code [are available]
    (https://github.com/aeon-toolkit/aeon-tutorials/tree/main/KDD-2024).

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, X, X)
    y: np.ndarray
        1D array of length XX or 300. The class labels for each time series instance
        in X.

    Examples
    --------
    >>> from aeon_neuro.datasets import load_kdd_full_example
    >>> X, y = load_kdd_full_example()

    Notes
    -----
    Number of time points:      150
    Number of channels:
    Train cases:        X
    Test cases:         X
    Number of classes:  X
    Details: TBC
    """
    data_path = "../../example_raw_eeg/basic_classification_task"
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
                + "_eeg.vhdr",
                verbose=verbose,
            )
            raw.load_data(verbose=verbose)
            raw.drop_channels(["ACC_X", "ACC_Y", "ACC_Z", "DeviceTrigger"])
            raw = raw.copy().filter(l_freq=0.5, h_freq=100, verbose=verbose)
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
    if split == "TRAIN":
        return X_train, y_train
    elif split == "TEST":
        return X_test, y_test
    return np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test))
