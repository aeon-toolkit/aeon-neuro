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
    covariance_type : {"cov", "erp"}, default="cov"
        Which covariance step to use. ``"cov"`` estimates a plain covariance
        matrix per case with ``pyriemann.estimation.Covariances``. ``"erp"``
        uses the supervised
        ``aeon_neuro.transformations.collection.ERPCovariances``, which augments
        each case with per-class ERPs before computing the covariance and
        therefore requires the labels ``y`` at fit time.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to the covariance step (the ``estimator``
        argument of ``Covariances`` or ``ERPCovariances``).
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance step. The valid
        keys depend on ``covariance_type``. For ``"cov"`` they are the remaining
        arguments of ``pyriemann.estimation.Covariances``. For ``"erp"`` they are
        the remaining arguments of
        ``aeon_neuro.transformations.collection.ERPCovariances``: ``classes``
        (list of int or None), ``svd`` (int or None) and ``estimator_params``
        (dict or None).
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "distance",
        "python_dependencies": ["pyriemann", "torch"],
    }

    def __init__(
        self,
        covariance_type: str = "cov",
        covariance_estimator: str = "scm",
        covariance_params: dict | None = None,
    ):
        self.covariance_type = covariance_type
        self.covariance_estimator = covariance_estimator
        self.covariance_params = covariance_params

        self.covariance_transformer_ = None
        self.classifier_ = None

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier."""
        X_cov = self._compute_covariances(X, fit=True, y=y)
        self.classifier_ = self._build_classifier()
        self.classifier_.fit(X_cov, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        X_cov = self._compute_covariances(X)
        return np.asarray(self.classifier_.predict(X_cov))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        X_cov = self._compute_covariances(X)
        probs = np.asarray(self.classifier_.predict_proba(X_cov), dtype=float)

        clf_classes = np.asarray(self.classifier_.classes_)
        if np.array_equal(clf_classes, self.classes_):
            return probs

        aligned = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, label in enumerate(clf_classes):
            aligned[:, self._class_dictionary[label]] = probs[:, i]
        return aligned

    def _build_covariance_transformer(self):
        """Construct the covariance transformer for ``covariance_type``."""
        kwargs = {} if self.covariance_params is None else dict(self.covariance_params)

        if self.covariance_type == "cov":
            from pyriemann.estimation import Covariances

            return Covariances(estimator=self.covariance_estimator, **kwargs)
        if self.covariance_type == "erp":
            from aeon_neuro.transformations.collection import ERPCovariances

            return ERPCovariances(estimator=self.covariance_estimator, **kwargs)
        raise ValueError(
            "covariance_type must be one of {'cov', 'erp'}, "
            f"got {self.covariance_type!r}."
        )

    def _compute_covariances(self, X, fit=False, y=None):
        """Estimate covariance matrices, on GPU if available.

        Plain covariances are computed on CUDA when available, then returned as
        a numpy array for the pyRiemann classifier. The supervised ERP step runs
        on numpy and uses ``y`` at fit time.
        """
        import torch

        if fit:
            self.covariance_transformer_ = self._build_covariance_transformer()

        # The ERP transformer is a numpy/aeon estimator, so only the plain
        # covariance path is moved onto the GPU.
        if self.covariance_type == "cov" and torch.cuda.is_available():
            X = torch.as_tensor(X, dtype=torch.float64, device="cuda")

        if fit:
            X_cov = self.covariance_transformer_.fit_transform(X, y)
        else:
            X_cov = self.covariance_transformer_.transform(X)

        if torch.is_tensor(X_cov):
            X_cov = X_cov.cpu().numpy()
        return X_cov

    def _build_classifier(self):
        """Construct the underlying pyRiemann classifier."""
        raise NotImplementedError("abstract method")

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
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
    covariance_type : {"cov", "erp"}, default="cov"
        Which covariance step to use. ``"cov"`` estimates a plain covariance
        matrix per case; ``"erp"`` uses the supervised
        ``aeon_neuro.transformations.collection.ERPCovariances``.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to the covariance step.
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance step. Valid keys
        depend on ``covariance_type``; for ``"erp"`` these are ``classes``,
        ``svd`` and ``estimator_params``.

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
        covariance_type: str = "cov",
        covariance_estimator: str = "scm",
        covariance_params: dict | None = None,
    ):
        self.metric = metric
        self.n_jobs = n_jobs
        super().__init__(
            covariance_type=covariance_type,
            covariance_estimator=covariance_estimator,
            covariance_params=covariance_params,
        )

    def _build_classifier(self):
        """Construct the underlying pyRiemann MDM classifier."""
        from pyriemann.classification import MDM

        return MDM(metric=self.metric, n_jobs=self.n_jobs)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
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
    covariance_type : {"cov", "erp"}, default="cov"
        Which covariance step to use. ``"cov"`` estimates a plain covariance
        matrix per case; ``"erp"`` uses the supervised
        ``aeon_neuro.transformations.collection.ERPCovariances``.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to the covariance step.
    covariance_params : dict or None, default=None
        Optional keyword arguments forwarded to the covariance step. Valid keys
        depend on ``covariance_type``; for ``"erp"`` these are ``classes``,
        ``svd`` and ``estimator_params``.

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
        covariance_type: str = "cov",
        covariance_estimator: str = "scm",
        covariance_params: dict | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        super().__init__(
            covariance_type=covariance_type,
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
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return parameter settings for estimator tests."""
        return {
            "n_neighbors": 3,
            "metric": "riemann",
            "n_jobs": 1,
            "covariance_estimator": "scm",
        }
