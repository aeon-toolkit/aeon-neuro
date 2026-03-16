# aeon_neuro/classification/feature_based/_csp_svm.py

"""MNE CSP + scikit-learn SVM classifier wrapper for aeon."""

from __future__ import annotations

__all__ = ["CSPSVMClassifier"]

import numpy as np
from aeon.classification import BaseClassifier


class CSPSVMClassifier(BaseClassifier):
    """Common Spatial Patterns plus SVM classifier.

    This estimator applies MNE's Common Spatial Patterns (CSP) transform to
    multivariate equal-length time series, then classifies the resulting features
    with a scikit-learn support vector classifier.

    The input shape is aeon's standard collection format:
    (n_cases, n_channels, n_timepoints).

    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components.
    reg : float, str or None, default=None
        Regularisation passed to ``mne.decoding.CSP``.
    log : bool or None, default=None
        Log transform setting passed to ``mne.decoding.CSP``.
    cov_est : {"concat", "epoch"}, default="concat"
        Covariance estimation mode for CSP.
    norm_trace : bool, default=False
        Whether to normalise class covariance by trace in CSP.
    cov_method_params : dict or None, default=None
        Extra covariance estimation keyword arguments for CSP.
    component_order : str, default="mutual_info"
        Component ordering strategy for CSP.
    C : float, default=1.0
        SVM regularisation parameter.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        SVM kernel.
    degree : int, default=3
        Degree for the polynomial kernel.
    gamma : {"scale", "auto"} or float, default="scale"
        Kernel coefficient for SVM.
    coef0 : float, default=0.0
        Independent term for poly and sigmoid kernels.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic in SVM.
    tol : float, default=1e-3
        Tolerance for the stopping criterion.
    cache_size : float, default=200
        Kernel cache size in MB.
    class_weight : dict, "balanced", or None, default=None
        Class weights for SVM.
    max_iter : int, default=-1
        Maximum iterations for SVM.
    break_ties : bool, default=False
        Whether to break ties according to confidence values.
    random_state : int or None, default=None
        Random seed for the SVM probability model.

    References
    ----------
    .. [1] Koles, Z. J., Lazar, M. S., and Zhou, S. Z. (1990).
       Spatial patterns underlying population differences in the background EEG.
       Brain Topography, 2(4), 275-284.
    .. [2] Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., and
       Muller, K.-R. (2008). Optimizing spatial filters for robust EEG
       single-trial analysis. IEEE Signal Processing Magazine, 25(1), 41-56.
    .. [3] Grosse-Wentrup, M., and Buss, M. (2008).
       Multiclass common spatial patterns and information theoretic feature
       extraction. IEEE Transactions on Biomedical Engineering, 55(8), 1991-2000.
    .. [4] Cortes, C., and Vapnik, V. (1995).
       Support-vector networks. Machine Learning, 20, 273-297.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "feature",
        "python_dependencies": ["mne", "sklearn"],
    }

    def __init__(
        self,
        n_components: int = 4,
        reg: float | str | None = None,
        log: bool | None = None,
        cov_est: str = "concat",
        norm_trace: bool = False,
        cov_method_params: dict | None = None,
        component_order: str = "mutual_info",
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
        class_weight=None,
        max_iter: int = -1,
        break_ties: bool = False,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.break_ties = break_ties
        self.random_state = random_state

        self.pipeline_ = None
        self.csp_ = None
        self.svc_ = None

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the CSP plus SVM classifier."""
        from mne.decoding import CSP
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        X = np.asarray(X, dtype=np.float64, order="C")

        self.csp_ = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            cov_est=self.cov_est,
            transform_into="average_power",
            norm_trace=self.norm_trace,
            cov_method_params=self.cov_method_params,
            component_order=self.component_order,
        )
        self.svc_ = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=True,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            break_ties=self.break_ties,
            random_state=self.random_state,
        )
        self.pipeline_ = Pipeline(
            [
                ("csp", self.csp_),
                ("svc", self.svc_),
            ]
        )
        self.pipeline_.fit(X, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        X = np.asarray(X, dtype=np.float64, order="C")
        return np.asarray(self.pipeline_.predict(X))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        X = np.asarray(X, dtype=np.float64, order="C")
        probs = np.asarray(self.pipeline_.predict_proba(X), dtype=float)

        clf_classes = np.asarray(self.svc_.classes_)
        if np.array_equal(clf_classes, self.classes_):
            return probs

        aligned = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, label in enumerate(clf_classes):
            aligned[:, self._class_dictionary[label]] = probs[:, i]
        return aligned

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return {
            "n_components": 2,
            "kernel": "linear",
            "C": 1.0,
            "random_state": 0,
        }
