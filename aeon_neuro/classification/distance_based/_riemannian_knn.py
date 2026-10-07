# aeon_neuro/classification/distance_based/_riemannian.py

"""pyRiemann covariance-based classifiers wrapped for aeon."""

from __future__ import annotations

__all__ = ["RiemannianMDMClassifier", "RiemannianKNNClassifier"]

import numpy as np
from aeon.classification import BaseClassifier


class _BaseRiemannianCovarianceClassifier(BaseClassifier):
    """Base class for pyRiemann covariance-based classifiers.

    This wrapper accepts aeon collection data of shape
    (n_cases, n_channels, n_timepoints), estimates one covariance matrix per case,
    and then applies a pyRiemann classifier to the resulting SPD matrices.

    Parameters
    ----------
    covariance_estimator : str, default="scm"
        Covariance estimator passed to ``pyriemann.estimation.Covariances``.
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance estimator.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "distance",
        "python_dependencies": ["pyriemann"],
    }

    def __init__(
        self,
        covariance_estimator: str = "scm",
        covariance_params: dict | None = None,
    ):
        self.covariance_estimator = covariance_estimator
        self.covariance_params = covariance_params

        self.covariance_transformer_ = None
        self.classifier_ = None

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier."""
        self.covariance_transformer_ = self._build_covariance_transformer()
        X_cov = self.covariance_transformer_.fit_transform(X)
        self.classifier_ = self._build_classifier()
        self.classifier_.fit(X_cov, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        X_cov = self.covariance_transformer_.transform(X)
        return np.asarray(self.classifier_.predict(X_cov))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        X_cov = self.covariance_transformer_.transform(X)
        probs = np.asarray(self.classifier_.predict_proba(X_cov), dtype=float)

        clf_classes = np.asarray(self.classifier_.classes_)
        if np.array_equal(clf_classes, self.classes_):
            return probs

        aligned = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, label in enumerate(clf_classes):
            aligned[:, self._class_dictionary[label]] = probs[:, i]
        return aligned

    def _build_covariance_transformer(self):
        """Construct the pyRiemann covariance transformer."""
        from pyriemann.estimation import Covariances

        kwargs = {} if self.covariance_params is None else dict(self.covariance_params)
        return Covariances(estimator=self.covariance_estimator, **kwargs)

    def _build_classifier(self):
        """Construct the underlying pyRiemann classifier."""
        raise NotImplementedError("abstract method")

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return parameter settings for estimator tests."""
        return {"covariance_estimator": "scm"}


class RiemannianMDMClassifier(_BaseRiemannianCovarianceClassifier):
    """Minimum Distance to Mean classifier on covariance matrices.

    This estimator converts each multivariate time series into a covariance matrix
    and applies ``pyriemann.classification.MDM``.

    Parameters
    ----------
    metric : str or dict, default="riemann"
        Metric passed to ``pyriemann.classification.MDM``.
    n_jobs : int, default=1
        Number of jobs passed to ``pyriemann.classification.MDM``.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to ``pyriemann.estimation.Covariances``.
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance estimator.

    References
    ----------
    .. [1] Barachant, A., Bonnet, S., Congedo, M., and Jutten, C. (2012).
       Multiclass Brain-Computer Interface Classification by Riemannian Geometry.
       IEEE Transactions on Biomedical Engineering, 59(4), 920-928.
    .. [2] Barachant, A., Bonnet, S., Congedo, M., and Jutten, C. (2010).
       Riemannian geometry applied to BCI classification.
       LVA/ICA 2010, LNCS 6365, 629-636.
    """

    def __init__(
        self,
        metric: str | dict = "riemann",
        n_jobs: int = 1,
        covariance_estimator: str = "scm",
        covariance_params: dict | None = None,
    ):
        self.metric = metric
        self.n_jobs = n_jobs
        super().__init__(
            covariance_estimator=covariance_estimator,
            covariance_params=covariance_params,
        )

    def _build_classifier(self):
        """Construct the underlying pyRiemann MDM classifier."""
        from pyriemann.classification import MDM

        return MDM(metric=self.metric, n_jobs=self.n_jobs)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return parameter settings for estimator tests."""
        return {
            "metric": "riemann",
            "n_jobs": 1,
            "covariance_estimator": "scm",
        }


class RiemannianKNNClassifier(_BaseRiemannianCovarianceClassifier):
    """k-nearest-neighbour classifier on covariance matrices.

    This estimator converts each multivariate time series into a covariance matrix
    and applies ``pyriemann.classification.KNearestNeighbor``.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours.
    metric : str or callable, default="riemann"
        Metric passed to ``pyriemann.classification.KNearestNeighbor``.
    n_jobs : int, default=1
        Number of jobs passed to ``pyriemann.classification.KNearestNeighbor``.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to ``pyriemann.estimation.Covariances``.
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance estimator.

    References
    ----------
    .. [1] Barachant, A., Bonnet, S., Congedo, M., and Jutten, C. (2010).
       Riemannian geometry applied to BCI classification.
       LVA/ICA 2010, LNCS 6365, 629-636.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "riemann",
        n_jobs: int = 1,
        covariance_estimator: str = "scm",
        covariance_params: dict | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        super().__init__(
            covariance_estimator=covariance_estimator,
            covariance_params=covariance_params,
        )

    def _build_classifier(self):
        """Construct the underlying pyRiemann k-NN classifier."""
        from pyriemann.classification import KNearestNeighbor

        return KNearestNeighbor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return parameter settings for estimator tests."""
        return {
            "n_neighbors": 3,
            "metric": "riemann",
            "n_jobs": 1,
            "covariance_estimator": "scm",
        }
