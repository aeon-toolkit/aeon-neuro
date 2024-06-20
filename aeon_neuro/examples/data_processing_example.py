import mne
import numpy as np
raw  = mne.io.read_raw_brainvision("./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-example_run-01_eeg.vhdr",preload=True)
data = raw.get_data()
print(np.shape(data))
print(raw.info)
#raw.plot()

#Remove non-EEG channels
raw.drop_channels(["ACC_X","ACC_Y","ACC_Z","DeviceTrigger"])
raw.set_montage(mne.channels.make_standard_montage("standard_1020"))

raw_filtered = raw.copy().filter(l_freq=0.5,h_freq=50)

ica= mne.preprocessing.ICA(n_components=32, max_iter="auto", random_state=1)
ica.fit(raw_filtered)
ica.plot_sources(raw_filtered,block=False)
ica.plot_components(nrows=4,ncols=8)
input()