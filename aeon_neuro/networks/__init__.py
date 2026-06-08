"""Deep learning networks."""

__all__ = ["BaseDeepLearningNetwork", "EEGNetNetwork"]

from aeon_neuro.networks._eegnet import EEGNetNetwork
from aeon_neuro.networks.base import BaseDeepLearningNetwork
