import numpy as np
import mne

def load_brainvision_to_numpy(path):
    """
    Load EEG data from brainvision format into numpy array, removing non-EEG channels

    Parameters
    _____________
    path : the file path to the vhdr file

    Returns
    _____________
    numpy_data: numpy.array
        a numpy array containing the raw EEG data in the form num_channels*num_timepoints


    """
    raw_data = mne.io.read_raw_brainvision(path)
    selected_EEG = raw_data.pick_types(eeg=True)
    data = selected_EEG.load_data()
    return np.asarray(data.get_data())
    

