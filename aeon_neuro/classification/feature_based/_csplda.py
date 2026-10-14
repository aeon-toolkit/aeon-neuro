"""CSP + scikit-learn LDA classifier for aeon."""

from __future__ import annotations

__all__ = ["CSPLDAClassifier"]

import numpy as np
from aeon.classification import BaseClassifier

from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)


class CSPLDAClassifier(BaseClassifier):
    """Common Spatial Patterns plus LDA classifier.

    This estimator applies the ``CommonSpatialPatterns`` transformer, which wraps
    ``mne.decoding.CSP``, to multivariate equal-length time series, then
    classifies the resulting log-power features with scikit-learn's
    ``LinearDiscriminantAnalysis``.

    The input shape is aeon's standard collection format:
    (n_cases, n_channels, n_timepoints).

    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components.
    reg : float, str or None, default=None
        Regularisation passed to ``mne.decoding.CSP``.
    log : bool or None, default=True
        Whether to apply log transform to the average power features. Passed
        to ``mne.decoding.CSP``.
    cov_est : {"concat", "epoch"}, default="concat"
        Covariance estimation mode for CSP.
    norm_trace : bool, default=False
        Whether to normalise class covariance by trace in CSP.
    cov_method_params : dict or None, default=None
        Extra covariance estimation keyword arguments for CSP.
    component_order : str, default="mutual_info"
        Component ordering strategy for CSP.
    solver : {"svd", "lsqr", "eigen"}, default="svd"
        Solver for ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis``.
        "svd" does not compute the covariance matrix and does not support
        shrinkage. "lsqr" and "eigen" support shrinkage and
        ``covariance_estimator``.
    shrinkage : "auto", float or None, default=None
        Shrinkage of the LDA covariance estimate. "auto" uses the Ledoit-Wolf
        lemma; a float between 0 and 1 gives fixed shrinkage. Only used when
        ``solver`` is "lsqr" or "eigen".
    priors : array-like of shape (n_classes,) or None, default=None
        Class prior probabilities. If None, they are inferred from the class
        proportions in the training data.
    store_covariance : bool, default=False
        If True, compute the within-class covariance when ``solver="svd"``.
        The other solvers always compute it.
    tol : float, default=1e-4
        Threshold for a singular value of X to be considered significant,
        used to estimate the rank of X. Only used when ``solver="svd"``.
    covariance_estimator : estimator or None, default=None
        Covariance estimator with a ``fit`` method and a ``covariance_``
        attribute, used instead of ``shrinkage``. Only used when ``solver``
        is "lsqr" or "eigen".

    References
    ----------
    .. [1] Ramoser, H., Muller-Gerking, J., and Pfurtscheller, G. (2000).
       Optimal spatial filtering of single trial EEG during imagined hand
       movement. IEEE Transactions on Rehabilitation Engineering, 8(4),
       441-446. https://doi.org/10.1109/86.895946
    .. [2] Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., and Muller, K.-R.
       (2008). Optimizing spatial filters for robust EEG single-trial analysis.
       IEEE Signal Processing Magazine, 25(1), 41-56.
       https://doi.org/10.1109/MSP.2008.4408441
    .. [3] Blankertz, B., Lemm, S., Treder, M., Haufe, S., and Muller, K.-R.
       (2011). Single-trial analysis and classification of ERP components -
       a tutorial. NeuroImage, 56(2), 814-825.
       https://doi.org/10.1016/j.neuroimage.2010.06.048
    .. [4] Ledoit, O., and Wolf, M. (2004). A well-conditioned estimator for
       large-dimensional covariance matrices. Journal of Multivariate
       Analysis, 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        n_components: int = 4,
        reg: float | str | None = None,
        log: bool | None = True,
        cov_est: str = "concat",
        norm_trace: bool = False,
        cov_method_params: dict | None = None,
        component_order: str = "mutual_info",
        solver: str = "svd",
        shrinkage: float | str | None = None,
        priors: np.ndarray | None = None,
        store_covariance: bool = False,
        tol: float = 1e-4,
        covariance_estimator=None,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the CSP plus LDA classifier."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.csp_ = CommonSpatialPatterns(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            cov_est=self.cov_est,
            transform_into="average_power",
            norm_trace=self.norm_trace,
            cov_method_params=self.cov_method_params,
            component_order=self.component_order,
        )
        self.lda_ = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=self.shrinkage,
            priors=self.priors,
            store_covariance=self.store_covariance,
            tol=self.tol,
            covariance_estimator=self.covariance_estimator,
        )
        self.lda_.fit(self._csp_features(X, y), y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return np.asarray(self.lda_.predict(self._csp_features(X)))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        probs = np.asarray(self.lda_.predict_proba(self._csp_features(X)), dtype=float)

        clf_classes = np.asarray(self.lda_.classes_)
        if np.array_equal(clf_classes, self.classes_):
            return probs

        aligned = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, label in enumerate(clf_classes):
            aligned[:, self._class_dictionary[label]] = probs[:, i]
        return aligned

    def _csp_features(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return CSP average power features of shape (n_cases, n_components).

        Fits the CSP transformer when ``y`` is given, otherwise uses the fitted one.
        """
        if y is not None:
            Xt = self.csp_.fit_transform(X, y)
        else:
            Xt = self.csp_.transform(X)
        return Xt[:, :, 0]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return {"n_components": 2}
