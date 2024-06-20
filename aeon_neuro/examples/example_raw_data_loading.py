import mne
import numpy as np

raw  = mne.io.read_raw_brainvision("./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-example_run-01_eeg.vhdr",preload=True)
data = raw.get_data()
channels,timepoints = np.shape(data)

