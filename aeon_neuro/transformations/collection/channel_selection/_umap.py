"""UMAP-based channel creation."""

from __future__ import annotations

from aeon.transformations.collection.base import BaseCollectionTransformer

__all__ = ["UMAP"]


class UMAP(BaseCollectionTransformer):
    """Create latent channels using Uniform Manifold Approximation and Projection.

    The input is reshaped so that each case-timepoint pair is a sample and the
    channels are features. A single UMAP model is fitted on all TRAIN channel
    vectors and reused to transform new collections. The output therefore has
    shape ``(n_cases, n_components, n_timepoints)``.

    Parameters
    ----------
    n_neighbors : int, default=3
        Size of the local neighbourhood used for manifold approximation.
    metric : str, default="cosine"
        Distance metric used to compare channel vectors.
    min_dist : float, default=0.1
        Minimum distance between points in the low-dimensional representation.
        Must be in the interval ``[0, 1]``.
    n_components : int, default=2
        Number of latent channels in the transformed collection.
    random_state : int or None, default=None
        Random seed used to initialise UMAP.

    Attributes
    ----------
    embedder_ : umap.UMAP
        Fitted UMAP model.
    n_channels_in_ : int
        Number of channels in the fitted collection.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        n_neighbors=3,
        metric="cosine",
        min_dist=0.1,
        n_components=2,
        random_state=None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit UMAP on the channel vectors in a collection.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training time series.
        y : None, default=None
            Ignored. Included for interface compatibility.

        Returns
        -------
        self
            Fitted transformer.
        """
        self._validate_parameters()
        self.n_channels_in_ = X.shape[1]

        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP requires the optional dependency 'umap-learn'. "
                "Install it with: pip install umap-learn"
            ) from exc

        self.embedder_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            min_dist=self.min_dist,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.embedder_.fit(self._reshape_collection(X))
        return self

    def _transform(self, X, y=None):
        """Transform channel vectors into latent channels.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Time series to transform.
        y : None, default=None
            Ignored. Included for interface compatibility.

        Returns
        -------
        np.ndarray of shape (n_cases, n_components, n_timepoints)
            Collection containing the latent channels.
        """
        if X.shape[1] != self.n_channels_in_:
            raise ValueError(
                "Number of channels in X does not match the data seen in fit. "
                f"Expected {self.n_channels_in_}, got {X.shape[1]}."
            )

        n_cases, _, n_timepoints = X.shape
        transformed = self.embedder_.transform(self._reshape_collection(X))
        return transformed.reshape(n_cases, n_timepoints, self.n_components).transpose(
            0, 2, 1
        )

    @staticmethod
    def _reshape_collection(X):
        """Reshape a collection into case-timepoint channel vectors."""
        return X.transpose(0, 2, 1).reshape(-1, X.shape[1])

    def _validate_parameters(self):
        """Validate constructor parameters."""
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 2:
            raise ValueError("n_neighbors must be an integer greater than 1.")
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not 0 <= self.min_dist <= 1:
            raise ValueError("min_dist must be in the interval [0, 1].")

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a lightweight parameter set for estimator tests."""
        return {
            "n_neighbors": 2,
            "n_components": 2,
            "random_state": 0,
        }
