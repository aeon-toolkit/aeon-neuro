"""Preprocessing of VIPA dataset."""

import numpy as np
from aeon.datasets import load_from_tsfile, write_to_tsfile

from aeon_neuro.transformations import DownsampleCollectionTransformer

if __name__ == "__main__":
    dataloc = "D:/PhD Files/VIPA Study/transformed/VIPA_Study/VIPA_Study.ts"
    writeloc = "D:/PhD Files/VIPA Study/transformed/VIPA_Study/"
    data, y = load_from_tsfile(dataloc, return_type="numpy3D")
    transformer = DownsampleCollectionTransformer(source_sfreq=500, target_sfreq=50)
    data = transformer.fit_transform(data)
    # newdata =[]
    # newlabels = []
    # classNames = ["closed","open"]

    trainX = data[:20]
    testX = data[20:]
    trainy = y[:20]
    testy = y[20:]
    classNames = ["closed", "open"]
    write_to_tsfile(
        np.asarray(trainX),
        writeloc,
        "VIPA_Study_downsampled_TRAIN",
        classNames,
        np.asarray(trainy),
    )
    write_to_tsfile(
        np.asarray(testX),
        writeloc,
        "VIPA_Study_downsampled_TEST",
        classNames,
        np.asarray(testy),
    )
