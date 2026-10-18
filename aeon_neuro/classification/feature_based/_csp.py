"""CSP + scikit-learn classifier for aeon."""

from __future__ import annotations

__all__ = ["CSPClassifier"]

import numpy as np
from aeon.classification import BaseClassifier

from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)


class CSPClassifier(BaseClassifier):
    """Common Spatial Patterns plus SVM or LDA classifier.

    This estimator applies the ``CommonSpatialPatterns`` transformer, which wraps
    ``mne.decoding.CSP``, to multivariate equal-length time series, then
    classifies the resulting average power features with either a scikit-learn
    support vector classifier (``SVC``) or ``LinearDiscriminantAnalysis``.

    The input shape is aeon's standard collection format:
    (n_cases, n_channels, n_timepoints).

    Parameters
    ----------
    classifier : {"svm", "lda"}, default="svm"
        Classifier fitted on the CSP features. "svm" uses
        ``sklearn.svm.SVC`` with ``kernel="linear"`` and ``probability=True``
        unless overridden in ``classifier_params``. "lda" uses
        ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis``.
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
    classifier_params : dict or None, default=None
        Keyword arguments passed to the scikit-learn classifier, e.g.
        ``{"C": 0.5, "kernel": "rbf"}`` for "svm" or
        ``{"solver": "lsqr", "shrinkage": "auto"}`` for "lda".
    random_state : int or None, default=None
        Random seed for the SVM probability model. Ignored for "lda", which is
        deterministic.

    Attributes
    ----------
    csp_ : CommonSpatialPatterns
        The fitted CSP transformer.
    classifier_ : sklearn.svm.SVC or LinearDiscriminantAnalysis
        The fitted scikit-learn classifier.

    References
    ----------
    .. [1] Koles, Z. J., Lazar, M. S., and Zhou, S. Z. (1990).
       Spatial patterns underlying population differences in the background EEG.
       Brain Topography, 2(4), 275-284.
    .. [2] Ramoser, H., Muller-Gerking, J., and Pfurtscheller, G. (2000).
       Optimal spatial filtering of single trial EEG during imagined hand
       movement. IEEE Transactions on Rehabilitation Engineering, 8(4),
       441-446. https://doi.org/10.1109/86.895946
    .. [3] Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., and Muller, K.-R.
       (2008). Optimizing spatial filters for robust EEG single-trial analysis.
       IEEE Signal Processing Magazine, 25(1), 41-56.
       https://doi.org/10.1109/MSP.2008.4408441
    .. [4] Grosse-Wentrup, M., and Buss, M. (2008).
       Multiclass common spatial patterns and information theoretic feature
       extraction. IEEE Transactions on Biomedical Engineering, 55(8), 1991-2000.
    .. [5] Cortes, C., and Vapnik, V. (1995).
       Support-vector networks. Machine Learning, 20, 273-297.
    .. [6] Blankertz, B., Lemm, S., Treder, M., Haufe, S., and Muller, K.-R.
       (2011). Single-trial analysis and classification of ERP components -
       a tutorial. NeuroImage, 56(2), 814-825.
       https://doi.org/10.1016/j.neuroimage.2010.06.048

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.feature_based import CSPClassifier
    >>> X, y = make_example_3d_numpy(n_cases=12, n_channels=4, random_state=0)
    >>> clf = CSPClassifier(classifier="lda", n_components=2)
    >>> clf.fit(X, y)  # doctest: +SKIP
    >>> clf.predict(X)  # doctest: +SKIP
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
        classifier: str = "svm",
        n_components: int = 4,
        reg: float | str | None = None,
        log: bool | None = True,
        cov_est: str = "concat",
        norm_trace: bool = False,
        cov_method_params: dict | None = None,
        component_order: str = "mutual_info",
        classifier_params: dict | None = None,
        random_state: int | None = None,
    ):
        self.classifier = classifier
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.classifier_params = classifier_params
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the CSP plus classifier pipeline."""
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
        self.classifier_ = self._build_classifier()
        self.classifier_.fit(self._csp_features(X, y), y)
        return self

    def _build_classifier(self):
        """Return an unfitted scikit-learn classifier for ``self.classifier``."""
        params = {} if self.classifier_params is None else dict(self.classifier_params)
        if self.classifier == "svm":
            from sklearn.svm import SVC

            params = {
                "kernel": "linear",
                "probability": True,
                "random_state": self.random_state,
                **params,
            }
            return SVC(**params)
        if self.classifier == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            return LinearDiscriminantAnalysis(**params)
        raise ValueError(
            f"classifier must be one of 'svm' or 'lda', got {self.classifier!r}."
        )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return np.asarray(self.classifier_.predict(self._csp_features(X)))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        probs = np.asarray(
            self.classifier_.predict_proba(self._csp_features(X)), dtype=float
        )

        clf_classes = np.asarray(self.classifier_.classes_)
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
        return [
            {"classifier": "svm", "n_components": 2, "random_state": 0},
            {"classifier": "lda", "n_components": 2},
        ]
