"""Channel creation algorithms."""

__all__ = ["CommonSpatialPatterns", "UMAPChannelCreator", "Xdawn"]

from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._umap import (
    UMAPChannelCreator,
)
from aeon_neuro.transformations.collection.channel_creation._xdawn import Xdawn
