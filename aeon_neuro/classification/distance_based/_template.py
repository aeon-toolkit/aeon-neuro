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
from scipy.signal import butter, sosfiltfilt


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

    With ``filter_bank=n``, the data is split into ``n`` sub-bands as in [1]_:
    sub-band ``m`` (``m = 1, ..., n``) passes ``8m - 2`` Hz up to 90 Hz, capped
    at 90% of the Nyquist frequency. TRCA is fitted in each sub-band, and the
    class score is the weighted mean of the sub-band correlations, with
    weights ``m ** -1.25 + 0.25``.

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
    filter_bank : int or None, default=None
        Number of filter bank sub-bands. If None, TRCA is applied to the data
        as given. Sub-bands whose lower edge is not below the upper edge are
        dropped.
    sfreq : float or None, default=None
        Sampling frequency of the input in Hz. Required when ``filter_bank``
        is set.

    Attributes
    ----------
    templates_ : np.ndarray of shape (n_classes, n_channels, n_timepoints)
        Mean (channel-centred) training trial of each class. With a filter
        bank, the templates of the first sub-band.
    filters_ : list of np.ndarray of shape (n_channels, n_components)
        Spatial filters of each class. With a filter bank, those of the first
        sub-band.
    bands_ : list of (low, high)
        Sub-bands in Hz if ``filter_bank`` is set, otherwise empty.

    Notes
    -----
    Stimulus-locked trials of equal length are assumed, as in SSVEP and c-VEP
    problems. Without a filter bank, band-pass the data first (for example
    6-40 Hz): slow drifts otherwise dominate the correlations. On one 12-class
    SSVEP problem the balanced accuracy rose from chance to above 0.99 with a
    6-40 Hz band-pass. The filter bank removes drift itself, since its lowest
    edge is 6 Hz, so give it unfiltered data. Its sub-bands are zero-phase
    order 4 Butterworth filters, where [1]_ uses Chebyshev type I filters.

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
        filter_bank: int | None = None,
        sfreq: float | None = None,
    ):
        self.n_components = n_components
        self.ensemble = ensemble
        self.reg = reg
        self.temperature = temperature
        self.filter_bank = filter_bank
        self.sfreq = sfreq
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
        self.bands_ = self._filter_bank_bands()
        self.band_weights_ = np.array(
            [m**-1.25 + 0.25 for m in range(1, len(self.bands_) + 1)]
        )
        self.band_models_ = []
        for Xb in self._sub_bands(X):
            Xb = _centre(Xb)
            templates = np.stack([Xb[y == c].mean(axis=0) for c in self.classes_])
            filters = [self._class_filters(Xb[y == c]) for c in self.classes_]
            self.band_models_.append((templates, filters))
        self.templates_, self.filters_ = self.band_models_[0]
        return self

    def _filter_bank_bands(self):
        if self.filter_bank is None:
            return []
        if self.filter_bank < 1:
            raise ValueError("filter_bank must be None or at least 1.")
        if self.sfreq is None or not self.sfreq > 0:
            raise ValueError("sfreq must be a positive number to use filter_bank.")
        high = min(90.0, 0.45 * float(self.sfreq))
        bands = [(8.0 * m - 2.0, high) for m in range(1, self.filter_bank + 1)]
        bands = [(lo, hi) for lo, hi in bands if lo < hi]
        if not bands:
            raise ValueError(
                f"No filter bank sub-band fits below Nyquist for sfreq {self.sfreq}."
            )
        return bands

    def _sub_bands(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not self.bands_:
            yield X
            return
        for lo, hi in self.bands_:
            sos = butter(4, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos")
            # the default pad length of sosfiltfilt can exceed short series
            padlen = min(3 * (2 * len(sos) + 1), X.shape[-1] - 1)
            yield sosfiltfilt(sos, X, axis=-1, padlen=padlen)

    def _scores(self, X):
        scores = sum(
            w * self._band_scores(Xb, *model)
            for w, Xb, model in zip(
                self._weights(), self._sub_bands(X), self.band_models_
            )
        )
        return scores / self._weights().sum()

    def _weights(self):
        return self.band_weights_ if self.bands_ else np.ones(1)

    def _band_scores(self, X, templates, filters):
        X = _centre(X)
        W_all = np.concatenate(filters, axis=1)
        scores = np.zeros((len(X), self.n_classes_))
        for k, template in enumerate(templates):
            W = W_all if self.ensemble else filters[k]
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
        return [
            {},
            {"n_components": 2, "ensemble": False},
            {"filter_bank": 2, "sfreq": 100.0},
        ]


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
