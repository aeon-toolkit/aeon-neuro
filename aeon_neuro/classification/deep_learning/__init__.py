"""Deep learning based EEG classifiers."""

__all__ = [
    "DeepConvNetClassifier",
    "EEGNetClassifier",
]


from aeon_neuro.classification.deep_learning._deepconvnet import DeepConvNetClassifier
from aeon_neuro.classification.deep_learning._eegnet import EEGNetClassifier
