"""Deep learning based EEG classifiers."""

__all__ = [
    "CBraModClassifier",
    "DeepConvNetClassifier",
    "EEGNetClassifier",
    "EEGPTClassifier",
]


from aeon_neuro.classification.deep_learning._cbramod import CBraModClassifier
from aeon_neuro.classification.deep_learning._deepconvnet import DeepConvNetClassifier
from aeon_neuro.classification.deep_learning._eegnet import EEGNetClassifier
from aeon_neuro.classification.deep_learning._eegpt import EEGPTClassifier
