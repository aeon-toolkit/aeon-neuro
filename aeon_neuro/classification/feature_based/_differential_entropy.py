"""Differential entropy features plus a scikit-learn classifier."""

from __future__ import annotations

__all__ = ["DifferentialEntropyClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier

from aeon_neuro.transformations.collection import DifferentialEntropy


class DifferentialEntropyClassifier(BaseClassifier):
    """Differential entropy band features with a scikit-learn classifier.

    This estimator composes two steps:

    ``DifferentialEntropy`` -> classifier

    ``DifferentialEntropy`` band-pass filters each channel in each frequency
    band and returns ``0.5 * log(2 * pi * e * variance)`` per channel and band.
    A scikit-learn classifier, by default ``StandardScaler`` followed by
    ``LogisticRegression``, is fitted on the ``n_channels * n_bands`` features.
    This is the standard baseline for EEG emotion and cognitive state
    recognition [1]_, [2]_.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input in Hz.
    bands : sequence of (low, high) or None, default=None
        Frequency bands in Hz. If None, delta 1-4, theta 4-8, alpha 8-14,
        beta 14-31 and gamma 31-50 Hz. Bands above the Nyquist frequency are
        dropped or capped; see ``DifferentialEntropy``.
    classifier : sklearn classifier or None, default=None
        Unfitted scikit-learn classifier fitted on the features. It is cloned
        before fitting. If None, ``make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=1000))``.
    order : int, default=4
        Order of the Butterworth band-pass filters.
    random_state : int or None, default=None
        Seed used to set the cloned classifier's ``random_state`` parameters, if
        it has any.

    Attributes
    ----------
    transformer_ : DifferentialEntropy
        The fitted feature extractor.
    classifier_ : sklearn classifier
        The fitted clone of ``classifier``.

    References
    ----------
    .. [1] Duan, R.-N., Zhu, J.-Y., and Lu, B.-L. (2013). Differential entropy
       feature for EEG-based emotion classification. 6th International
       IEEE/EMBS Conference on Neural Engineering, 81-84.
       https://doi.org/10.1109/NER.2013.6695876
    .. [2] Zheng, W.-L., and Lu, B.-L. (2015). Investigating critical frequency
       bands and channels for EEG-based emotion recognition with deep neural
       networks. IEEE Transactions on Autonomous Mental Development, 7(3),
       162-175. https://doi.org/10.1109/TAMD.2015.2431497

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.feature_based import (
    ...     DifferentialEntropyClassifier,
    ... )
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=10, n_channels=3, n_timepoints=200, random_state=0
    ... )
    >>> clf = DifferentialEntropyClassifier(sfreq=128).fit(X, y)
    >>> clf.predict(X).shape
    (10,)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "feature",
    }

    def __init__(
        self, sfreq, bands=None, classifier=None, order: int = 4, random_state=None
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.classifier = classifier
        self.order = order
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.transformer_ = DifferentialEntropy(
            sfreq=self.sfreq, bands=self.bands, order=self.order
        )
        features = self.transformer_.fit_transform(X)
        if self.classifier is None:
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000)
            )
        else:
            classifier = self.classifier
        self.classifier_ = _clone_estimator(classifier, self.random_state)
        self.classifier_.fit(features, y)
        return self

    def _predict(self, X):
        return self.classifier_.predict(self.transformer_.transform(X))

    def _predict_proba(self, X):
        features = self.transformer_.transform(X)
        if not hasattr(self.classifier_, "predict_proba"):
            return super()._predict_proba(X)
        proba = self.classifier_.predict_proba(features)
        out = np.zeros((len(X), self.n_classes_))
        for j, c in enumerate(self.classifier_.classes_):
            out[:, self._class_dictionary[c]] = proba[:, j]
        return out

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return {"sfreq": 100.0}
