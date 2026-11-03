"""Channel creation algorithms."""

__all__ = ["CommonSpatialPatterns", "ERPCovariances", "UMAPChannelCreator", "XDAWN"]

from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._erpcovariance import (
    ERPCovariances,
)
from aeon_neuro.transformations.collection.channel_creation._umap import (
    UMAPChannelCreator,
)
from aeon_neuro.transformations.collection.channel_creation._xdawn import XDAWN
