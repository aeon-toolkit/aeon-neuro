import mne
import numpy as np
from aeon.datasets import write_to_tsfile

if __name__ == "__main__":
    datapath = "D:/PhD Files/matchingpennies/raw/"
    writeloc = "D:/PhD Files/matchingpennies/"
    shortest = np.Inf
    dataset = []
    labels = []
    subjects = ["05", "06", "07", "08", "09", "10", "11"]
    classNames = ["S1", "S2"]
    for subject in subjects:
        data = mne.io.read_raw_brainvision(
            datapath
            + "sub-"
            + subject
            + "/eeg/sub-"
            + subject
            + "_task-matchingpennies_eeg.vhdr"
        )
        prevOnset = 0
        for annotation in data.annotations:
            if prevOnset == 0:
                prevOnset = annotation["onset"]
            else:
                datacopy = data.copy()
                description = annotation["description"]
                classVal = description.split("/")
                currentOnset = annotation["onset"]
                instance = datacopy.crop(tmin=prevOnset, tmax=currentOnset)
                prevOnset = currentOnset
                fulldata = instance.get_data()
                _, timepoints = np.shape(fulldata)
                dataset.append(fulldata)
                labels.append(classVal)
                if timepoints < shortest:
                    shortest = timepoints
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:, :shortest]

    write_to_tsfile(
        np.asarray(dataset), writeloc, "matchingpennies", classNames, np.asarray(labels)
    )
