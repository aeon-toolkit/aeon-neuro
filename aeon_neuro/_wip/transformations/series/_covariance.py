"""Covariance matrix transformer."""

from aeon.transformations.series.base import BaseSeriesTransformer


class CovarianceMatrix(BaseSeriesTransformer):
    """Covariance matrix transformer.

    Estimate the (unbiased) pairwise covariance matrix between channels in the
    time domain. The result is a positive semidefinite (PSD) real-valued matrix
    of shape (n_channels, n_channels).

    Parameters
    ----------
    correlation : bool, optional
        Whether to estimate the pairwise correlation (standardized covariance) matrix,
        by default False

    Examples
    --------
    >>> from aeon_neuro._wip.transformations.series._covariance import (
    ...     CovarianceMatrix)
    >>> import numpy as np
    >>> n_channels, n_timepoints = 5, 360
    >>> X = np.random.standard_normal(size=(n_channels, n_timepoints))
    >>> transformer = CovarianceMatrix()
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape == (n_channels, n_channels)
    True
    >>> np.iscomplexobj(X_transformed)
    False
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, correlation=False):
        self.correlation = correlation
        super().__init__(axis=1)  # (n_channels, n_timepoints)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        n_timepoints = X.shape[-1]
        X_centered = X - X.mean(axis=1, keepdims=True)
        if self.correlation:
            X_centered /= X.std(axis=1, keepdims=True)
        return (X_centered @ X_centered.T) / (n_timepoints - 1)
