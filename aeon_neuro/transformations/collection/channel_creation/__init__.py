"""Channel creation algorithms."""

__all__ = [
    "CommonSpatialPatterns",
    "TRCommonSpatialPatterns",
    "UMAPChannelCreator",
    "XDAWN",
]

from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._trcsp import (
    TRCommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._umap import (
    UMAPChannelCreator,
)
from aeon_neuro.transformations.collection.channel_creation._xdawn import XDAWN
