"""EEG related channel selection."""

__all__ = [
    "BPSO",
    "CLeVerCluster",
    "CLeVerHybrid",
    "CLeVerRank",
    "DetachRocket",
    "DetachRocketChannelSelector",
    "Riemannian",
    "CaseTimeReducer",
    "UMAP",
]

from aeon_neuro.transformations.collection.channel_selection._bpso import BPSO
from aeon_neuro.transformations.collection.channel_selection._clever import (
    CLeVerCluster,
    CLeVerHybrid,
    CLeVerRank,
)
from aeon_neuro.transformations.collection.channel_selection._detach_rocket import (
    DetachRocket,
    DetachRocketChannelSelector,
)
from aeon_neuro.transformations.collection.channel_selection._reducer import (
    CaseTimeReducer,
)
from aeon_neuro.transformations.collection.channel_selection._riemannian import (
    Riemannian,
)
from aeon_neuro.transformations.collection.channel_selection._umap import UMAP
