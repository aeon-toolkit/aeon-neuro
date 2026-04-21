"""UMAP embedding transformer for univariate collections of time series."""

from __future__ import annotations

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer

__all__ = ["UMAP"]


class UMAP(BaseCollectionTransformer):
    """Embed a univariate collection of time series into a feature space.

    Transforms a collection of equal-length univariate time series into a 2D
    tabular feature representation using Uniform Manifold Approximation and
    Projection (UMAP) [1]. Each case is treated as a single sample and each
    time point as a feature, so the input collection is reshaped from
    ``(n_cases, 1, n_timepoints)`` to ``(n_cases, n_timepoints)`` before
    fitting the embedding.

    For input data of shape ``(n_cases, 1, n_timepoints)``, the output has
    shape ``(n_cases, n_components)``.

    This is an embedding method rather than a channel selection or channel
    creation method. The transformed features are nonlinear latent
    representations of complete time series cases.

    Parameters
    ----------
    n_components : int, default=2
        Number of embedding dimensions in the transformed output.
    n_neighbors : int, default=15
        Size of local neighbourhood used for manifold approximation.
    min_dist : float, default=0.1
        Minimum distance between points in the low-dimensional embedding.
        Must be in the interval ``[0, 1]``.
    metric : str, default="euclidean"
        Distance metric used to compute similarities between cases.
    random_state : int or None, default=None
        Random seed used to initialise the embedding.

    Raises
    ------
    ValueError
        If `n_components` is not a positive integer.
    ValueError
        If `n_neighbors` is not a positive integer.
    ValueError
        If `min_dist` is not in the interval ``[0, 1]``.

    Notes
    -----
    UMAP is a nonlinear dimensionality reduction technique that operates on
    tabular data of shape ``(n_samples, n_features)``. In this transformer,
    each univariate time series is treated as a single sample and the temporal
    dimension is treated as the feature dimension.

    References
    ----------
    .. [1] McInnes, L., Healy, J., and Melville, J. (2020). UMAP: Uniform
           Manifold Approximation and Projection for Dimension Reduction.
           IEEE Transactions on Visualization and Computer Graphics, 26(1),
           1105-1115.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "output_data_type": "Tabular",
    }

    def __init__(
        self,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the embedding model.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, 1, n_timepoints)
            Collection of univariate time series to fit the embedding on.
        y : None, default=None
            Ignored. Included for interface compatibility.

        Returns
        -------
        self : UMAP
            Reference to self.
        """
        self._validate_params()
        X = self._validate_X(X)

        self.n_timepoints_in_ = X.shape[2]

        X_tabular = self._reshape_collection(X)

        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP requires the optional dependency 'umap-learn'. "
                "Install it with: pip install umap-learn"
            ) from exc

        self.embedder_ = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        self.embedder_.fit(X_tabular)

        return self

    def _transform(self, X, y=None):
        """Transform a collection into embedding features.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, 1, n_timepoints)
            Collection of univariate time series to transform.
        y : None, default=None
            Ignored. Included for interface compatibility.

        Returns
        -------
        X_t : np.ndarray of shape (n_cases, n_components)
            Embedded feature representation of the input collection.
        """
        X = self._validate_X(X)

        if X.shape[2] != self.n_timepoints_in_:
            raise ValueError(
                "Number of time points in X does not match the data seen in fit. "
                f"Expected {self.n_timepoints_in_}, got {X.shape[2]}."
            )

        X_tabular = self._reshape_collection(X)
        return self.embedder_.transform(X_tabular)

    @staticmethod
    def _reshape_collection(X):
        """Reshape a univariate collection into tabular form."""
        return X[:, 0, :]

    @staticmethod
    def _validate_X(X):
        """Validate input collection."""
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D numpy array of shape " "(n_cases, 1, n_timepoints)."
            )

        if X.shape[1] != 1:
            raise ValueError(
                "UMAP only supports univariate collections. "
                f"Expected 1 channel, got {X.shape[1]}."
            )

        if X.shape[2] < 1:
            raise ValueError("X must contain at least one time point.")

        return X

    def _validate_params(self):
        """Validate transformer parameters."""
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer.")

        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        if not (0 <= self.min_dist <= 1):
            raise ValueError("min_dist must be in the interval [0, 1].")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {
                "n_components": 2,
                "n_neighbors": 3,
                "random_state": 0,
            },
            {
                "n_components": 3,
                "n_neighbors": 5,
                "metric": "cosine",
                "random_state": 1,
            },
        ]
