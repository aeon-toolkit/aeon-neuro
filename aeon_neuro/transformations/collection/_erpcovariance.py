"""ERP covariance estimation, wrapping pyRiemann's ERPCovariances."""

from __future__ import annotations

__all__ = ["ERPCovariances"]

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer


class ERPCovariances(BaseCollectionTransformer):
    """Event-related potential (ERP) covariance transformer.

    Computes a covariance matrix for each case that captures how its channels
    relate to each other and how strongly it contains each class's ERP. During
    `fit`, an ERP is estimated for each class by averaging the cases of that
    class at each time point.
    During `transform`, the ERPs are concatenated with the channels of each case
    and the covariance of the combined series is computed. Uses
    `pyriemann.estimation.ERPCovariances`.

    Parameters
    ----------
    classes : list of int or None, optional
        Classes to estimate ERPs for. If None, all classes in `y` are used,
        by default None.
    estimator : str, optional
        Covariance matrix estimator, see
        `pyriemann.geometry.covariance.covariances` for the available options,
        by default "scm".
    svd : int or None, optional
        Number of SVD components used to reduce each class's ERP. If None,
        no reduction is applied, by default None.
    estimator_params : dict or None, optional
        Additional parameters passed to the covariance estimator. If None,
        no additional parameters are passed, by default None.
    """

    _tags = {
        "capability:multivariate": True,
        "requires_y": True,
        "fit_is_empty": False,
    }

    def __init__(self, classes=None, estimator="scm", svd=None, estimator_params=None):
        self.classes = classes
        self.estimator = estimator
        self.svd = svd
        self.estimator_params = estimator_params
        super().__init__()

    def _fit(self, X, y=None):
        from pyriemann.estimation import ERPCovariances as PyRiemannERPCovariances

        if self.classes is not None:
            if len(self.classes) == 0:
                raise ValueError("classes must contain at least one class.")
            if len(set(self.classes)) != len(self.classes):
                raise ValueError("classes must not contain duplicates.")
            labels = set(np.unique(y))
            missing = [c for c in self.classes if c not in labels]
            if missing:
                raise ValueError(f"classes {missing} are not present in y.")

        if self.svd is not None and (
            not isinstance(self.svd, int) or isinstance(self.svd, bool) or self.svd < 1
        ):
            raise ValueError("svd must be None or an integer of at least 1.")

        params = self.estimator_params or {}
        self.erp_covariances_ = PyRiemannERPCovariances(
            classes=self.classes,
            estimator=self.estimator,
            svd=self.svd,
            **params,
        )
        self.erp_covariances_.fit(X, y)
        return self

    def _transform(self, X, y=None):
        """Estimate ERP covariance matrices.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_cases, n_components, n_components)
            ERP covariance matrices. ``n_components`` is
            ``(1 + n_classes) * n_channels`` if ``svd`` is None, otherwise
            ``n_channels + n_classes * min(svd, n_channels)``.
        """
        return self.erp_covariances_.transform(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
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
        return [{}, {"estimator": "lwf", "svd": 1}]
