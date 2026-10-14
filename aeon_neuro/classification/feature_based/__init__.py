"""Feature based EEG classifiers."""

__all__ = [
    "CSPLDAClassifier",
    "CSPSVMClassifier",
]
from aeon_neuro.classification.feature_based._csplda import CSPLDAClassifier
from aeon_neuro.classification.feature_based._cspsvm import CSPSVMClassifier
