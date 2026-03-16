"""Imports, wrappers and implementations of EEG classification algorithms.

aeon imports:
    HIVECOTEV2 (HC2)
    MultiRocketHydraClassifier (MRHydra)
    InceptionTimeClassifier (IT)

wrappers:

1. Braindecode
    EEGNet
    DeepConvNet (DCN)
2. pyrieman
Riemannian-MDM (RMD)
3. MNE+sckitkit-learn


"""

__all__ = [
    "HIVECOTEV2",
    "MultiRocketHydraClassifier",
    "InceptionTimeClassifier",
    "DeepConvNetClassifier",
    "EEGNetClassifier",
    "RiemannianMDMClassifier",
    "RiemannianKNNClassifier",
    "TimeCNNClassifier",
]

from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier, TimeCNNClassifier
from aeon.classification.hybrid import HIVECOTEV2

from aeon_neuro.classification.deep_learning import (
    DeepConvNetClassifier,
    EEGNetClassifier,
)
from aeon_neuro.classification.distance_based import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)
