"""Channel creation algorithms."""

__all__ = ["CommonSpatialPatterns", "UMAPChannelCreator"]

from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._umap import (
    UMAPChannelCreator,
)
