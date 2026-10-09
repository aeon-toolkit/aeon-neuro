"""Channel creation algorithms."""

__all__ = ["CommonSpacialPatterns", "UMAPChannelCreator"]

from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpacialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._umap import (
    UMAPChannelCreator,
)
