"""Imports, wrappers and implementations of EEG classification algorithms.

aeon imports:
    HIVECOTEV2 (HC2)
    MultiRocketHydraClassifier (MRHydra)
    InceptionTimeClassifier (IT)

wrappers:

1. Braindecode
    EEGNet
    DeepConvNet (DCN)
    EEGConformer
2. pyriemann
Riemannian-MDM (RMD)
3. MNE+scikit-learn
    CSP+SVM, CSP+LDA (CSPClassifier)


"""

__all__ = [
    "HIVECOTEV2",
    "MultiRocketHydraClassifier",
    "InceptionTimeClassifier",
    "DeepConvNetClassifier",
    "EEGConformerClassifier",
    "EEGNetClassifier",
    "RiemannianMDMClassifier",
    "RiemannianKNNClassifier",
    "TimeCNNClassifier",
    "CSPClassifier",
]

from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier, TimeCNNClassifier
from aeon.classification.hybrid import HIVECOTEV2

from aeon_neuro.classification.deep_learning import (
    DeepConvNetClassifier,
    EEGConformerClassifier,
    EEGNetClassifier,
)
from aeon_neuro.classification.distance_based import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)
from aeon_neuro.classification.feature_based import CSPClassifier
