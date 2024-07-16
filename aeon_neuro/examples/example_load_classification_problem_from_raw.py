"""Example for loading and formatting a raw dataset into a classifcation problem"""

import mne
import numpy as np
from aeon.classification.convolution_based import RocketClassifier


def load_basic_classification_problem():
    """Load data and format to classification problem

    Loads the data, pre-processes, segments into instances and runs a basic classifier


    """
    data_path = "./aeon_neuro/data/basic_classification_task"
    tasks = ["task","rest"]
    runs=["01","02","03","04","05","06"]

    X = []
    y=[]

    #Load each data file from BIDS format 
    for task in tasks:
        for run in runs:
            raw = mne.io.read_raw_brainvision(
                data_path+"/sub-01/ses-01/eeg/sub-01_ses-01_task-"+
                task+"_run-"+run+"_eeg.vhdr"
            )
            raw.load_data()
            raw.drop_channels(["ACC_X","ACC_Y","ACC_Z","DeviceTrigger"])
            raw = raw.copy().filter(l_freq=0.5,h_freq=100)
            if task == "task":
                for trial in raw.annotations:
                    trial_data = raw.copy().crop(tmin = trial["onset"]-0.2,tmax=trial["onset"]+0.800)
                    data = trial_data.get_data()
                    X.append(data[:,:1000])
                    y.append("task")
            else:
                for timepoint in np.arange (30, 90, 1.5):
                    trial_data = raw.copy().crop(tmin = timepoint,tmax=timepoint+1)
                    data = trial_data.get_data()
                    X.append(data[:,:1000])
                    y.append("rest")

    X_train,X_test = [],[]
    y_train,y_test=[],[]           
    n_instances,_,_ = np.shape(X)
    subject_instances = int(n_instances/40)

    #Reformat data into a classification problem
    for i in range(subject_instances):
        for j in range(40):
            loc = i*40+j
            if j<20:
                X_train.append(X[loc])
                y_train.append(y[loc])
            else:
                X_test.append(X[loc])
                y_test.append(y[loc])

    X_train,X_test = np.array(X_train),np.array(X_test)
    y_train,y_test = np.array(y_train),np.array(y_test)

    #Run minirocket on example dataset
    cls = RocketClassifier(rocket_transform="minirocket")
    cls.fit(X_train,y_train)
    print(cls.score(X_test,y_test))

if __name__=="__main__":
    load_basic_classification_problem()
