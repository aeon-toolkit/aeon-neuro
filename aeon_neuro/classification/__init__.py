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
]

from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.hybrid import HIVECOTEV2

from aeon_neuro.classification._deepconvnet import DeepConvNetClassifier
from aeon_neuro.classification._eegnet import EEGNetClassifier
