"""Template-matching classifiers for evoked and steady-state EEG.

``TRCAClassifier`` and ``TemplateCCAClassifier`` learn one template per class
from the training data, so unlike CCA against sinusoidal references they do not
need each class's stimulation frequency.
"""

from __future__ import annotations

__all__ = ["TRCAClassifier", "TemplateCCAClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from scipy.linalg import eigh


def _softmax(scores, temperature):
    """Softmax of class scores (correlations) scaled by ``temperature``."""
    z = temperature * (scores - scores.max(axis=1, keepdims=True))
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _centre(X):
    """Remove each channel's mean over time."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=-1, keepdims=True)


class TRCAClassifier(BaseClassifier):
    """Task-related component analysis classifier.

    For each class, TRCA finds spatial filters ``w`` that maximise the
    covariance between that class's training trials relative to their total
    variance, by solving the generalised eigenproblem ``S w = lambda Q w`` [1]_.
    ``S = sum_{i != j} x_i x_j^T`` is the inter-trial covariance and
    ``Q = sum_i x_i x_i^T`` the covariance of the concatenated trials, with
    each channel centred in each trial. The class template is the mean training
    trial. A test case is scored against each class by the Pearson correlation
    between the spatially filtered case and the filtered template.

    In ensemble TRCA (``ensemble=True``, the default) the filters of every
    class are stacked into one matrix and used for every comparison, which is
    usually more accurate [1]_.

    Parameters
    ----------
    n_components : int, default=1
        Number of spatial filters per class (leading eigenvectors).
    ensemble : bool, default=True
        If True, use the stacked filters of all classes for every class.
        If False, score each class with its own filters only.
    reg : float, default=1e-6
        Ridge added to ``Q``, as a fraction of its mean diagonal, so that the
        eigenproblem is well posed when channels are nearly collinear.
    temperature : float, default=10.0
        Scale of the correlations in the softmax that gives
        ``predict_proba``. The probabilities rank classes; they are not
        calibrated.

    Attributes
    ----------
    templates_ : np.ndarray of shape (n_classes, n_channels, n_timepoints)
        Mean (channel-centred) training trial of each class.
    filters_ : list of np.ndarray of shape (n_channels, n_components)
        Spatial filters of each class.

    Notes
    -----
    Stimulus-locked trials of equal length are assumed, as in SSVEP and c-VEP
    problems. Band-pass the data first (for example 6-40 Hz): slow drifts
    otherwise dominate the correlations. On one 12-class SSVEP problem the
    balanced accuracy rose from chance to above 0.99 with a 6-40 Hz band-pass.
    A filter bank variant [1]_ is not implemented.

    References
    ----------
    .. [1] Nakanishi, M., Wang, Y., Chen, X., Wang, Y.-T., Gao, X., and Jung,
       T.-P. (2018). Enhancing detection of SSVEPs for a high-speed brain
       speller using task-related component analysis. IEEE Transactions on
       Biomedical Engineering, 65(1), 104-112.
       https://doi.org/10.1109/TBME.2017.2694818

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.distance_based import TRCAClassifier
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=10, n_channels=3, n_timepoints=50, random_state=0
    ... )
    >>> TRCAClassifier().fit(X, y).predict(X).shape
    (10,)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        n_components: int = 1,
        ensemble: bool = True,
        reg: float = 1e-6,
        temperature: float = 10.0,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.reg = reg
        self.temperature = temperature
        super().__init__()

    def _class_filters(self, Xc):
        n_channels = Xc.shape[1]
        total = Xc.sum(axis=0)
        Q = np.einsum("ict,ijt->cj", Xc, Xc)
        S = total @ total.T - Q
        Q = Q + self.reg * max(np.trace(Q), 1e-12) / n_channels * np.eye(n_channels)
        _, vecs = eigh(S, Q)
        k = min(self.n_components, n_channels)
        return vecs[:, ::-1][:, :k]

    def _fit(self, X, y):
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")
        X = _centre(X)
        self.templates_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        self.filters_ = [self._class_filters(X[y == c]) for c in self.classes_]
        return self

    def _scores(self, X):
        X = _centre(X)
        W_all = np.concatenate(self.filters_, axis=1)
        scores = np.zeros((len(X), self.n_classes_))
        for k, template in enumerate(self.templates_):
            W = W_all if self.ensemble else self.filters_[k]
            a = np.einsum("ck,nct->nkt", W, X).reshape(len(X), -1)
            b = (W.T @ template).ravel()
            a = a - a.mean(axis=1, keepdims=True)
            b = b - b.mean()
            den = np.sqrt((a * a).sum(axis=1) * (b @ b))
            with np.errstate(invalid="ignore", divide="ignore"):
                scores[:, k] = np.where(den > 0, a @ b / den, 0.0)
        return scores

    def _predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    def _predict_proba(self, X):
        return _softmax(self._scores(X), self.temperature)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return [{}, {"n_components": 2, "ensemble": False}]


class TemplateCCAClassifier(BaseClassifier):
    """Canonical correlation with class templates.

    Each class template is the mean training case. A test case is scored
    against each class by the first canonical correlation between the case and
    the template, both treated as ``n_timepoints`` observations of
    ``n_channels`` variables, and assigned to the class with the highest
    correlation. This is CCA with data-driven references, as used in
    individual-template CCA for SSVEP [1]_, so the stimulation frequencies are
    not needed.

    Parameters
    ----------
    reg : float, default=1e-6
        Ridge added to each channel covariance, as a fraction of its mean
        diagonal, before whitening.
    temperature : float, default=10.0
        Scale of the correlations in the softmax that gives
        ``predict_proba``. The probabilities rank classes; they are not
        calibrated.

    Attributes
    ----------
    templates_ : list of np.ndarray of shape (n_timepoints, n_channels)
        Whitened mean training case of each class.

    Notes
    -----
    With fewer time points than channels the canonical correlation is close to
    1 for every class, so the classifier needs ``n_timepoints`` well above
    ``n_channels``. As for ``TRCAClassifier``, band-pass the data first
    (for example 6-40 Hz), because slow drifts otherwise dominate.

    References
    ----------
    .. [1] Bin, G., Gao, X., Wang, Y., Li, Y., Hong, B., and Gao, S. (2011).
       A high-speed BCI based on code modulation VEP. Journal of Neural
       Engineering, 8(2), 025015.
       https://doi.org/10.1088/1741-2560/8/2/025015

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.distance_based import TemplateCCAClassifier
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=10, n_channels=3, n_timepoints=50, random_state=0
    ... )
    >>> TemplateCCAClassifier().fit(X, y).predict(X).shape
    (10,)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "distance",
    }

    def __init__(self, reg: float = 1e-6, temperature: float = 10.0):
        self.reg = reg
        self.temperature = temperature
        super().__init__()

    def _whiten(self, A):
        """Whiten the columns of A (n_timepoints, n_channels): A_w^T A_w = I."""
        A = A - A.mean(axis=0)
        C = A.T @ A
        n = len(C)
        C = C + self.reg * max(np.trace(C), 1e-12) / n * np.eye(n)
        vals, vecs = np.linalg.eigh(C)
        return A @ (vecs / np.sqrt(np.maximum(vals, np.finfo(np.float64).tiny)))

    def _fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.templates_ = [
            self._whiten(X[y == c].mean(axis=0).T) for c in self.classes_
        ]
        return self

    def _scores(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros((len(X), self.n_classes_))
        for i, x in enumerate(X):
            wx = self._whiten(x.T)
            for k, wt in enumerate(self.templates_):
                scores[i, k] = np.linalg.svd(wx.T @ wt, compute_uv=False)[0]
        return scores

    def _predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    def _predict_proba(self, X):
        return _softmax(self._scores(X), self.temperature)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return {}
