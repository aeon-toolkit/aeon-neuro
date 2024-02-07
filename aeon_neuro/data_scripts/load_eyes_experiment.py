import mne
import numpy as np
from aeon.datasets import write_to_tsfile

# Loads each baseline EEG datafile for the VIPA study
# Splits into eyes closed and eyes open portions
# truncates to equal length then outputs to .ts file format

if __name__ == "__main__":
    # Subject files with valid markers for closed/open eye segements
    datasets = [
        "_V5e2_",
        "_V6_E2_rs",
        "_V8exp2_",
        "_V9e2_",
        "_v10e2_",
        "_V11e2_",
        "_V13e2_",
        "_V14e2_",
        "_V15e2_",
        "_V16e2_",
        "_V21e2_",
        "_V23e2_",
        "_V27e2_",
        "_V28e2_",
        "_V34e2_",
        "_V35e2_",
        "_V37e2_",
    ]

    data = []
    classes = []
    classNames = ["closed", "open"]
    current_min = np.Inf

    # Set path to EEG data files and output path
    dataset_location = ""
    output_location = "."

    # Load and add each dataset
    for filename in datasets:
        # Load data into MNE
        raw = mne.io.read_raw_brainvision(
            dataset_location + "VIPA" + filename + "baseline.vhdr"
        )

        # Uncomment line below to remove any non-EEG channels
        # raw = raw.pick_types(eeg=True)

        # Make copies as most mne functions work inplace
        closed = raw.copy()
        opened = raw.copy()
        closedStart = 0
        openStart = 0

        # Select data based on eyes closed and open markers
        for annotation in raw.annotations:
            if annotation["description"] == "Comment/eyes closed":
                closedStart = annotation["onset"]
            elif annotation["description"] == "Comment/eyes open":
                openStart = annotation["onset"]
        closed.crop(tmin=closedStart, tmax=openStart)
        opened.crop(tmin=openStart)

        closed = np.asarray(closed.get_data())
        opened = np.asarray(opened.get_data())

        # Keep track of the current shortest instance
        if np.shape(closed)[1] < current_min:
            current_min = np.shape(closed)[1]
        if np.shape(opened)[1] < current_min:
            current_min = np.shape(opened)[1]

        # Add data and relevent class to an array
        data.append(closed)
        classes.append("closed")
        data.append(opened)
        classes.append("open")

    # Truncate each instance to length of shortest instance
    for i in range(len(data)):
        data[i] = data[i][:, :current_min]
        # print(np.shape(data[i]))

    # print(np.shape(np.asarray(data)))
    # print(classes)
    write_to_tsfile(
        np.asarray(data), output_location, "VIPA_Study", classNames, np.asarray(classes)
    )
