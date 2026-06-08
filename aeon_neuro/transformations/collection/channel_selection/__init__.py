"""EEG related channel selection."""

__all__ = [
    "BPSO",
    "Riemannian",
    "DetachRocket",
]

from aeon_neuro.transformations.collection.channel_selection._bpso import BPSO
from aeon_neuro.transformations.collection.channel_selection._detach_rocket import (
    DetachRocket,
)
from aeon_neuro.transformations.collection.channel_selection._riemannian import (
    Riemannian,
)
