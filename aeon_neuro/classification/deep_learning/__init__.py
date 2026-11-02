"""Deep learning based EEG classifiers."""

__all__ = [
    "CBraModClassifier",
    "DeepConvNetClassifier",
    "EEGConformerClassifier",
    "EEGNetClassifier",
    "LabramClassifier",
]


from aeon_neuro.classification.deep_learning._cbramod import CBraModClassifier
from aeon_neuro.classification.deep_learning._deepconvnet import DeepConvNetClassifier
from aeon_neuro.classification.deep_learning._eegconformer import (
    EEGConformerClassifier,
)
from aeon_neuro.classification.deep_learning._eegnet import EEGNetClassifier
from aeon_neuro.classification.deep_learning._labram import LabramClassifier
