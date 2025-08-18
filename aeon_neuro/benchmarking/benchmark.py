from aeon.datasets import load_classification
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
from pyriemann.utils.covariance import covariances
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.deep_learning import TimeCNNClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from pyriemann.classification import KNearestNeighbor,MDM
from sklearn.svm import SVC
import zenodo_get

datasets = {
    "Alzheimers": 15500801,
    "ButtonPress": 15522930,
    "FaceDetection": 11206216,
    "FingerMovements":/11206220,
    "HandMovementDirection":11206224,
    "ImaginedOpenCloseFist":15493010,
    "ImaginedFeetHands":15493041,
    "InnerSpeech":15425020,
    "LongIntervalTask":15427724,
    "LowCost":15523038,
    "MatchingPennies":15523058,
    "MindReading":15523138,
    "MotorImagery":11206246,
    "OpenCloseFist":15492985,
    "FeetHands":15495750,
    "PhotoStimulation":15500781,
    "PronouncedSpeech":15392755",
    "SelfRegulationSCP1":11206265,
    "SelfRegulationSCP2":11206269,
    "ShortIntervalTask":15491177,
    "SitStand":15500726,
    "Sleep":15526565,
    "SongFamiliarity": 15496080,
    "VisualSpeech":15366803,
}

def download_dataset(dataset):#
    try:
        zenodo_get.download(datasets[dataset],output_dir="aeon_neuro/datasets/data/"+dataset)
        return True
    except:
        return False

dataset="HandMovementDirection"
classifier = "MDM" # Set desired classifier

data_path="aeon_neuro/datasets/data/"


if not os.path.isdir(data_path+dataset):
    download_dataset(dataset)
X_train, y_train =load_classification(dataset,extract_path=data_path, split="TRAIN")
X_test, y_test = load_classification(dataset,extract_path=data_path, split="test")

def get_classifier(classifier): #Import and set specified classifier
    if classifier=="HC2":
        return HIVECOTEV2()
    elif classifier=="MultiRocketHydra":
        return MultiRocketHydraClassifier()
    elif classifier=="CNN":
        return TimeCNNClassifier()
    elif classifier=="InceptionTime":
        return InceptionTimeClassifier()
    elif classifier=="R-KNN":
        return KNearestNeighbor(n_neighbors=1)
    elif classifier=="MDM":
        return MDM()
    elif classifier=="SVM" or classifier=="CSP-SVM":
        return SVC(probability=True)
        
cls = get_classifier(classifier=classifier)

if classifier=="CSP-SVM": #Apply CSP transform
    transform = CSP(transform_into="csp_space")
    X_train = transform.fit_transform(X_train, y_train)
    X_test = transform.transform(X_test)
if classifier == "R-KNN" or classifier == "MDM":
     X_train=covariances(X_train,estimator="lwf")
     X_test=covariances(X_test,estimator="lwf")  
if classifier=="SVM" or classifier=="SVM-CSP": #Concatenate channels
    X_train=X_train.reshape(X_train.shape[0], -1)
    X_test=X_test.reshape(X_train.shape[0], -1)
    
start = int(round(time.time()*1000))
cls.fit(X_train,y_train) #Fit classifier
fit_time = (int(round(time.time()*1000))-start)
start = int(round(time.time() * 1000))
test_probs = cls.predict_proba(X_test) #Predict values
test_time = (int(round(time.time() * 1000))- start)
test_preds = cls.classes_[np.argmax(test_probs, axis=1)]
test_acc = accuracy_score(y_test, test_preds)
print(test_acc,fit_time,test_time)
