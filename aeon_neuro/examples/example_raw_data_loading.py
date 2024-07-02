"""Example for loading raw BIDS formatted EEG using MNE."""

import mne


def load_example_dataset():
    """Get two instances of raw EEG data.

    The first being resting state eyes closed, the second resting state eyes open.

    Returns
    -------
    instances: a list containing of 2 raw EEG instances

    """
    instances = []
    instances.append(
        mne.io.read_raw_brainvision(
            """./aeon_neuro/data/example_dataset/sub-01/
            ses-01/eeg/sub-01_ses-01_task-EC_run-01_eeg.vhdr""",
            preload=True,
        ).get_data()
    )
    instances.append(
        mne.io.read_raw_brainvision(
            """./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/
            sub-01_ses-01_task-EO_run-01_eeg.vhdr""",
            preload=True,
        ).get_data()
    )
    return instances
