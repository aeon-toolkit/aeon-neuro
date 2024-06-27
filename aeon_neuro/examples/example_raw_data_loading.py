import mne
import numpy as np
from scipy.linalg import logm
import math

def load_example_dataset():
    #Returns two instances of raw EEG data, the first being resting state eyes closed, the second resting state eyes open
    instances=[]
    instances.append(mne.io.read_raw_brainvision("./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-EC_run-01_eeg.vhdr",preload=True).get_data())
    instances.append(mne.io.read_raw_brainvision("./aeon_neuro/data/example_dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-EO_run-01_eeg.vhdr",preload=True).get_data())
    return instances

def riemannian_distance(A,B):
    print(np.shape(A))
    print(np.shape(B))
    a = np.trace(A)
    b = np.trace(B)
    a_root = logm(A)
    c_temp = logm(np.matmul(a_root,np.matmul(B,a_root)))
    c = 2*np.trace(c_temp)
    return math.sqrt(a+b-c)






if __name__ == "__main__":
    instances = load_example_dataset()
    print(np.shape(instances[0]))
    print(np.shape(instances[1]))
    print(riemannian_distance(np.cov(instances[0]),np.cov(instances[1])))
    

