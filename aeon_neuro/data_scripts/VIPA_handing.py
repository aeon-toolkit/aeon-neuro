from aeon.datasets._data_io import load_from_tsfile
from aeon.datasets._data_io import write_ndarray_to_tsfile
from tsmlpy_neuro.transformations import epoching
from tsmlpy_neuro.transformations import downsample
import numpy as np
dataloc = "D:/PhD Files/VIPA Study/transformed/VIPA_Study/VIPA_Study.ts"
writeloc = "D:/PhD Files/VIPA Study/transformed/VIPA_Study/"
data,y = load_from_tsfile(dataloc,return_data_type="numpy3D")
print(np.shape(data))
data = downsample.downsample_series(data,500,50)
#newdata =[]
#newlabels = []
#classNames = ["closed","open"]


trainX = data[:20]
testX = data[20:]
trainy = y[:20]
testy = y[20:]
classNames = ["closed","open"]
write_ndarray_to_tsfile(np.asarray(trainX),writeloc,"VIPA_Study_downsampled_TRAIN",classNames,np.asarray(trainy))
write_ndarray_to_tsfile(np.asarray(testX),writeloc,"VIPA_Study_downsampled_TEST",classNames,np.asarray(testy))
