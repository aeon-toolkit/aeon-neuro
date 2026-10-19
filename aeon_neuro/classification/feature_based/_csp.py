"""CSP + scikit-learn classifier for aeon."""

from __future__ import annotations

__all__ = ["CSPClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from sklearn.base import BaseEstimator, is_classifier

from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)


class CSPClassifier(BaseClassifier):
    """Common Spatial Patterns plus a scikit-learn classifier.

    This estimator applies the ``CommonSpatialPatterns`` transformer, which wraps
    ``mne.decoding.CSP``, to multivariate equal-length time series, then
    classifies the resulting average power features with a scikit-learn
    classifier, by default ``LinearDiscriminantAnalysis`` (the classic CSP+LDA
    pipeline).

    The input shape is aeon's standard collection format:
    (n_cases, n_channels, n_timepoints).

    Parameters
    ----------
    classifier : sklearn classifier or None, default=None
        Unfitted scikit-learn classifier fitted on the CSP features, for example
        ``SVC(kernel="linear", probability=True)``. It is cloned before fitting,
        so the object passed in is not modified. If None,
        ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis()`` is used.
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
    random_state : int or None, default=None
        Seed used to set the cloned classifier's ``random_state`` parameters, if it
        has any. If None, the classifier's own setting is kept. Has no effect
        on deterministic classifiers such as ``LinearDiscriminantAnalysis``.

    Attributes
    ----------
    csp_ : CommonSpatialPatterns
        The fitted CSP transformer.
    classifier_ : sklearn classifier
        The fitted clone of ``classifier``.

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
    >>> from sklearn.svm import SVC
    >>> from aeon_neuro.classification.feature_based import CSPClassifier
    >>> X, y = make_example_3d_numpy(n_cases=12, n_channels=4, random_state=0)
    >>> clf = CSPClassifier(n_components=2)  # LinearDiscriminantAnalysis
    >>> clf = CSPClassifier(SVC(kernel="linear", probability=True), n_components=2)
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
        classifier=None,
        n_components: int = 4,
        reg: float | str | None = None,
        log: bool | None = True,
        cov_est: str = "concat",
        norm_trace: bool = False,
        cov_method_params: dict | None = None,
        component_order: str = "mutual_info",
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
        """Return an unfitted clone of ``self.classifier`` or the default LDA."""
        if self.classifier is None:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            classifier = LinearDiscriminantAnalysis()
        elif isinstance(self.classifier, BaseEstimator) and is_classifier(
            self.classifier
        ):
            classifier = self.classifier
        else:
            raise TypeError(
                "classifier must be a scikit-learn classifier, e.g. "
                f"LinearDiscriminantAnalysis(), or None, got {self.classifier!r}."
            )
        return _clone_estimator(classifier, self.random_state)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return np.asarray(self.classifier_.predict(self._csp_features(X)))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        Xt = self._csp_features(X)
        if not hasattr(self.classifier_, "predict_proba"):
            # One-hot encode predictions for classifiers without probabilities.
            probs = np.zeros((X.shape[0], self.n_classes_), dtype=float)
            for i, label in enumerate(self.classifier_.predict(Xt)):
                probs[i, self._class_dictionary[label]] = 1.0
            return probs

        probs = np.asarray(self.classifier_.predict_proba(Xt), dtype=float)

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
        from sklearn.svm import SVC

        return [
            {"n_components": 2},
            {
                "classifier": SVC(kernel="linear", probability=True),
                "n_components": 2,
                "random_state": 0,
            },
        ]
