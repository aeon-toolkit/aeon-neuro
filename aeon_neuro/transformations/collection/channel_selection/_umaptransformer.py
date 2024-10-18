"""Transforms a collection of series into features using UMAP."""

import numpy as np
import umap
from aeon.transformations.collection.base import BaseCollectionTransformer

__all__ = ["UMAPTransformer"]


class UMAPTransformer(BaseCollectionTransformer):
    """UMAP transformer.

    Reduces the the number of channels to a given number of features
    for a collection of series using UMAP.

    Parameters
    ----------
    n_ neighbours : int, optional
        Number of neighbours considered, by default 3
    metric: string, optional
        Distance measure used, by default "cosine"
    min_dist: float, optional
        Minimum distance between points in low dimenonal representation, default 0.1
    n_components: int, optional
        Number of dimensions in final representation, default 2

    Raises
    ------
    ValueError
        If `min_dist` is not between 0-1.
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, n_neighbours=3, metric="cosine", min_dist=0.1, n_components=2):

        if min_dist < 0 or min_dist > 1:
            raise ValueError("min_dist must be between 0 and 1")

        super().__init__()
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.min_dist = min_dist
        self.n_components = n_components

    def _transform(self, X, y=None):
        """Reduces the dimenensionality of the collection using UMAP.

        Parameters
        ----------
        X : list or np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Input time series collection.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_cases, n_components, n_timepoints)
        """
        _, _, n_timepoints = np.shape(X)
        X_t = np.transpose(X, (2, 0, 1))
        relations = [
            {j: j for j in range(n_timepoints)} for _ in range(n_timepoints - 1)
        ]

        aligned_mapper = umap.AlignedUMAP(
            metric=self.metric,
            min_dist=self.min_dist,
            n_neighbors=self.n_neighbours,
            n_components=self.n_components,
        ).fit(X_t, relations=relations)

        X = np.transpose(aligned_mapper.embeddings_, (1, 2, 0))

        return X
