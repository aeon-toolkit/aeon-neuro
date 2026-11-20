"""Riemannian tangent space mapping, wrapping pyRiemann's TangentSpace."""

from __future__ import annotations

__all__ = ["TangentSpace"]

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer


class TangentSpace(BaseCollectionTransformer):
    """Riemannian tangent space transformer.

    Estimates a covariance matrix for each case and maps it onto the tangent
    space of the Riemannian manifold of symmetric positive definite (SPD)
    matrices. Covariance matrices lie on a curved manifold, so Euclidean
    classifiers such as logistic regression or SVM cannot be applied to them
    directly. During ``fit``, a reference point is computed as the mean of the
    training covariance matrices. During ``transform``, each covariance matrix
    is projected onto the tangent space at that point and its upper triangle is
    vectorised, giving ``n_channels * (n_channels + 1) // 2`` features per case.
    Uses ``pyriemann.estimation.Covariances`` and
    ``pyriemann.tangentspace.TangentSpace``.

    Parameters
    ----------
    estimator : str, default="scm"
        Covariance matrix estimator, see
        ``pyriemann.geometry.covariance.covariances`` for the available options.
        ``"scm"`` needs full rank data, i.e. more timepoints than channels and
        no flat or linearly dependent channels (e.g. average referenced data).
        Regularised estimators such as ``"lwf"`` or ``"oas"`` always give SPD
        matrices.
    metric : str or dict, default="riemann"
        Metric used to compute the reference point and the tangent space
        mapping, see ``pyriemann.tangentspace.TangentSpace`` for the available
        options.
    tsupdate : bool, default=False
        If True, the reference point is recomputed on the data passed to
        ``transform``. This can help with sessions recorded at different times,
        but means test data influences its own features.

    Attributes
    ----------
    covariances_ : pyriemann.estimation.Covariances
        The covariance estimator.
    tangent_space_ : pyriemann.tangentspace.TangentSpace
        The fitted tangent space, holding the reference point.

    See Also
    --------
    ERPCovariances : Covariance matrices for event-related potential data.

    References
    ----------
    .. [1] Barachant, A., Bonnet, S., Congedo, M., and Jutten, C. (2012).
       Multiclass brain-computer interface classification by Riemannian
       geometry. IEEE Transactions on Biomedical Engineering, 59(4), 920-928.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon_neuro.transformations.collection import TangentSpace
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(10, 4, 100))
    >>> TangentSpace().fit_transform(X).shape
    (10, 10)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "fit_is_empty": False,
        "python_dependencies": "pyriemann",
    }

    def __init__(
        self,
        estimator: str = "scm",
        metric: str | dict = "riemann",
        tsupdate: bool = False,
    ):
        self.estimator = estimator
        self.metric = metric
        self.tsupdate = tsupdate
        super().__init__()

    def _fit(self, X: np.ndarray, y=None):
        """Fit the reference point on the covariance matrices of X.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials.
        y : np.ndarray of shape (n_cases,) or None, default=None
            Ignored for interface compatibility.

        Returns
        -------
        self
            Fitted transformer.
        """
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace as PyRiemannTangentSpace

        if not isinstance(self.tsupdate, bool):
            raise ValueError("tsupdate must be a bool.")

        self.covariances_ = Covariances(estimator=self.estimator)
        covs = self._compute_covariances(X)

        self.tangent_space_ = PyRiemannTangentSpace(
            metric=self.metric, tsupdate=self.tsupdate
        )
        self.tangent_space_.fit(covs)
        return self

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Map the covariance matrix of each case onto the tangent space.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials with the same n_channels as in ``fit``.
        y : None
            Ignored for interface compatibility.

        Returns
        -------
        np.ndarray of shape (n_cases, n_channels * (n_channels + 1) // 2)
            Tangent space vectors, one row per case.
        """
        covs = self._compute_covariances(X)
        return self.tangent_space_.transform(covs)

    def _compute_covariances(self, X: np.ndarray) -> np.ndarray:
        """Estimate covariance matrices and check they are SPD."""
        n_channels = X.shape[1]
        n_timepoints = X.shape[2]
        if self.estimator == "scm" and n_timepoints <= n_channels:
            raise ValueError(
                f"estimator='scm' needs more timepoints than channels to give "
                f"positive definite covariance matrices, got n_timepoints="
                f"{n_timepoints} and n_channels={n_channels}. Use a regularised "
                f"estimator such as 'lwf' or 'oas'."
            )

        covs = self.covariances_.transform(X)

        # rank < n_channels catches flat and linearly dependent channels
        ranks = np.linalg.matrix_rank(covs, hermitian=True)
        singular = np.where(ranks < n_channels)[0]
        if singular.size:
            raise ValueError(
                f"Covariance matrices of cases {singular.tolist()} are not "
                f"positive definite, e.g. due to flat or linearly dependent "
                f"(average referenced) channels. Use a regularised estimator "
                f"such as 'lwf' or 'oas'."
            )
        return covs

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
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
        return [{}, {"estimator": "lwf", "metric": "logeuclid"}]
