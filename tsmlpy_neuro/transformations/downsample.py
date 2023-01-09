import numpy as np


def downsample_series(X,sfreq,target_sample_rate):
    new_ratio = int(sfreq/target_sample_rate)
    n_instances,n_dimensions,n_timepoints = np.shape(X)
    updated_timepoints = int(n_timepoints/new_ratio)
    downsampled_data = np.zeros((n_instances,n_dimensions,updated_timepoints+1))
    for i in range(n_instances):
        for j in range(n_dimensions):
            updated_index=0
            for k in range(0,n_timepoints,new_ratio):
                downsampled_data[i][j][updated_index]=X[i][j][k]
                updated_index+=1
    return downsampled_data

