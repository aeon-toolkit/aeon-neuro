"""xDAWN + scikit-learn classifier for aeon."""

from __future__ import annotations

__all__ = ["XdawnClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from sklearn.base import BaseEstimator, is_classifier

from aeon_neuro.transformations.collection.channel_creation import Xdawn


class XdawnClassifier(BaseClassifier):
    """xDAWN spatial filtering plus a scikit-learn classifier.

    This estimator applies the ``Xdawn`` transformer, which wraps
    ``mne.decoding.XdawnTransformer``, to multivariate equal-length time series,
    flattens the filtered series of each case into a feature vector, then
    classifies the features with a scikit-learn classifier, by default
    ``LinearDiscriminantAnalysis`` (the classic xDAWN+LDA pipeline).

    xDAWN enhances event-related potentials (ERPs), so this classifier is
    intended for ERP data such as P300 or oddball paradigms.

    The input shape is aeon's standard collection format:
    (n_cases, n_channels, n_timepoints).

    Parameters
    ----------
    classifier : sklearn classifier or None, default=None
        Unfitted scikit-learn classifier fitted on the xDAWN features, for
        example ``LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")``
        or ``SVC(kernel="linear", probability=True)``. It is cloned before
        fitting, so the object passed in is not modified. If None,
        ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis()`` is used.
    n_components : int, default=2
        Number of xDAWN spatial filters kept per class.
    reg : float, str or None, default=None
        Regularisation for covariance estimation, passed to
        ``mne.decoding.XdawnTransformer``.
    signal_cov : np.ndarray or None, default=None
        Covariance matrix of shape (n_channels, n_channels) that the class ERPs
        are compared against. If None, it is estimated from the training data.
    cov_method_params : dict or None, default=None
        Extra keyword arguments passed to ``mne.compute_covariance``.
    random_state : int or None, default=None
        Seed used to set the cloned classifier's ``random_state`` parameters, if it
        has any. If None, the classifier's own setting is kept. Has no effect
        on deterministic classifiers such as ``LinearDiscriminantAnalysis``.

    Attributes
    ----------
    xdawn_ : Xdawn
        The fitted xDAWN transformer.
    classifier_ : sklearn classifier
        The fitted clone of ``classifier``.

    Notes
    -----
    Each case is represented by its xDAWN-filtered series flattened to length
    ``n_classes * min(n_channels, n_components) * n_timepoints``. This is often
    large relative to the number of cases, so a shrinkage LDA,
    ``LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")``, is
    recommended for small ERP datasets [3]_.

    References
    ----------
    .. [1] Rivet, B., Souloumiac, A., Attina, V., and Gibert, G. (2009).
       xDAWN algorithm to enhance evoked potentials: application to
       brain-computer interface. IEEE Transactions on Biomedical Engineering,
       56(8), 2035-2043. https://doi.org/10.1109/TBME.2009.2012869
    .. [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., and Mattout, J.
       (2011). Theoretical analysis of xDAWN algorithm: application to an
       efficient sensor selection in a P300 BCI. 19th European Signal
       Processing Conference (EUSIPCO), 1382-1386.
    .. [3] Blankertz, B., Lemm, S., Treder, M., Haufe, S., and Muller, K.-R.
       (2011). Single-trial analysis and classification of ERP components -
       a tutorial. NeuroImage, 56(2), 814-825.
       https://doi.org/10.1016/j.neuroimage.2010.06.048

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.feature_based import XdawnClassifier
    >>> X, y = make_example_3d_numpy(n_cases=12, n_channels=4, random_state=0)
    >>> clf = XdawnClassifier(n_components=2)
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
        n_components: int = 2,
        reg: float | str | None = None,
        signal_cov: np.ndarray | None = None,
        cov_method_params: dict | None = None,
        random_state: int | None = None,
    ):
        self.classifier = classifier
        self.n_components = n_components
        self.reg = reg
        self.signal_cov = signal_cov
        self.cov_method_params = cov_method_params
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the xDAWN plus classifier pipeline."""
        self.xdawn_ = Xdawn(
            n_components=self.n_components,
            reg=self.reg,
            signal_cov=self.signal_cov,
            cov_method_params=self.cov_method_params,
        )
        self.classifier_ = self._build_classifier()
        self.classifier_.fit(self._xdawn_features(X, y), y)
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
        return np.asarray(self.classifier_.predict(self._xdawn_features(X)))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        Xt = self._xdawn_features(X)
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

    def _xdawn_features(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return flattened xDAWN features of shape (n_cases, n_features).

        Fits the xDAWN transformer when ``y`` is given, otherwise uses the fitted
        one.
        """
        if y is not None:
            Xt = self.xdawn_.fit_transform(X, y)
        else:
            Xt = self.xdawn_.transform(X)
        return Xt.reshape(Xt.shape[0], -1)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        from sklearn.svm import SVC

        return [
            {"n_components": 1},
            {
                "classifier": SVC(kernel="linear", probability=True),
                "n_components": 1,
                "random_state": 0,
            },
        ]
