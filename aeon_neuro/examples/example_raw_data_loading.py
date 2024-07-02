import mne
import numpy as np
from scipy.linalg import logm
import math

def load_example_dataset():
    #Returns two instances of raw EEG data, the first being resting state eyes closed, the second resting state eyes open
    instances=[]
    instances.append(
        mne.io.read_raw_brainvision(
            "./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-EC_run-01_eeg.vhdr",
            preload=True
        ).get_data()
    )
    instances.append(
        mne.io.read_raw_brainvision(
            "./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-EO_run-01_eeg.vhdr"
            ,preload=True
        ).get_data()
    )
    return instances
    

