import json

import mne
import numpy as np


def load_auxiliary_info(path, dataset_name):
    full_path = path + dataset_name + ".json"
    try:
        f = open(full_path)
        aux_data = json.load(f)
        return aux_data
    except:
        print("Auxiliary file not found at: " + full_path)


def load_brainvision_to_numpy(path, remove_non_EEG=True):
    """
    Load EEG data from brainvision format into numpy array, removing non-EEG channels

    Parameters
    ----------
    _____________
    path : the file path to the vhdr file
    remove_non_EEG : if non-EEG channels should be removed prior to returning
                     Default: True

    Returns
    -------
    _____________
    numpy_data: numpy.array
        a numpy array containing the raw EEG data in the form [num_channels][num_timepoints]


    """
    mne_data = mne.io.read_raw_brainvision(path)
    if remove_non_EEG:
        mne_data = mne_data.pick_types(eeg=True)
    data = mne_data.load_data()
    return np.asarray(data.get_data())
