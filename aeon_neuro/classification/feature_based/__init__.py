"""Feature based EEG classifiers."""

__all__ = ["CSPClassifier", "XDAWNClassifier"]
from aeon_neuro.classification.feature_based._csp import CSPClassifier
from aeon_neuro.classification.feature_based._xdawn import XDAWNClassifier
