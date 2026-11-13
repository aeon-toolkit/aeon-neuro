"""Experimental classifiers used for benchmark comparisons.

Not part of the public API: these may change or be removed once the
experiments that use them are finished.
"""

__all__ = [
    "CBraModFineTuneClassifier",
    "LabramFineTuneClassifier",
    "MOABBPipelineClassifier",
    "MOABBTSLRClassifier",
    "NeuralBenchCoSpectraLRClassifier",
    "NeuralBenchCTNetClassifier",
    "NeuralBenchDeepClassifier",
    "NeuralBenchPreprocessor",
    "NeuralBenchTSLRClassifier",
    "NeuralBenchXdawnTSLRClassifier",
]

from aeon_neuro.classification.experimental._foundation import (
    CBraModFineTuneClassifier,
    LabramFineTuneClassifier,
)
from aeon_neuro.classification.experimental._neuralbench import (
    MOABBPipelineClassifier,
    MOABBTSLRClassifier,
    NeuralBenchCoSpectraLRClassifier,
    NeuralBenchCTNetClassifier,
    NeuralBenchDeepClassifier,
    NeuralBenchPreprocessor,
    NeuralBenchTSLRClassifier,
    NeuralBenchXdawnTSLRClassifier,
)
