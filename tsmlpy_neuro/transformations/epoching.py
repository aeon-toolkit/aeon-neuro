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

def epoch_dataset(X,y,sfreq,epoch_size):
    epoched_data=[]
    new_labels=[]
    for i in range(len(X)):
        new_data = epoch_series_by_time(X[i],sfreq,epoch_size)
        for j in new_data:
            epoched_data.append(new_data[j])
            new_labels.append(y[i])
    return np.asarray(epoched_data),np.asarray(new_labels)
