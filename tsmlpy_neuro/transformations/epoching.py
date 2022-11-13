import numpy as np


def epoch_series_by_percentage(series,percent):
    pass

def epoch_series_by_time(series,sfreq,epoch_size):
    n_dimensions,n_timepoints = series.shape
    instances_per_epoch = int((epoch_size/1000)*sfreq)
    epoched_data = []
    index=0
    while index+instances_per_epoch<=n_timepoints:
        epoch = series[0:n_dimensions,index:index+instances_per_epoch]
        epoched_data.append(epoch)
        index+=instances_per_epoch
    return np.asarray(epoched_data)
