"""SSVEP classifier using canonical correlation analysis (CCA)."""

import numbers

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.cross_decomposition import CCA

__all__ = ["SSVEPCCAClassifier"]


def _qr_basis(A):
    """Orthonormal basis for the centred, non-flat columns of A.

    Each column is centred to zero mean. Columns with (near) zero norm after
    centring are flat and are dropped. A thin QR decomposition of what
    remains gives an orthonormal basis for its column space.

    Parameters
    ----------
    A : np.ndarray of shape (n_timepoints, n_columns)
        Signals as columns, e.g. a case's channels or a reference's sine and
        cosine rows, transposed.

    Returns
    -------
    np.ndarray of shape (n_timepoints, n_kept) or None
        Orthonormal basis ``Q``, or ``None`` if every column is flat.
    """
    A = A - A.mean(axis=0)
    norms = np.linalg.norm(A, axis=0)
    # relative tolerance: a flat column may not centre to exact zeros
    A = A[:, norms > 1e-12 * max(norms.max(), 1.0)]
    if A.shape[1] == 0:
        return None
    Q, _ = np.linalg.qr(A)
    return Q


class SSVEPCCAClassifier(BaseClassifier):
    """Steady-state visual evoked potential (SSVEP) classifier using CCA.

    Each class is associated with a stimulation frequency. During ``fit``, a
    reference signal is built for every class from sines and cosines at that
    frequency and its harmonics. During ``predict``, canonical correlation
    analysis (CCA) measures how strongly each case correlates with every
    reference, and the class whose reference correlates most is returned [1]_.

    No training data is used to build the references beyond its length and the
    set of class labels, so the classifier is effectively training-free.

    The input shape is aeon's standard collection format:
    ``(n_cases, n_channels, n_timepoints)``.

    Parameters
    ----------
    sfreq : int or float
        Sampling frequency of the input in Hz. Must be finite and positive.
    class_frequencies : dict
        Mapping from class label to its stimulation frequency in Hz, for
        example ``{"left": 8, "right": 10, "up": 12, "down": 15}``. Every label
        present in ``y`` must be a key, the frequencies must be distinct, and
        the highest harmonic of each must be below the Nyquist frequency
        (``sfreq / 2``).
    n_harmonics : int, default=2
        Number of harmonics (including the fundamental) used to build each
        reference signal. Must be a positive integer.

    References
    ----------
    .. [1] Lin, Z., Zhang, C., Wu, W., and Gao, X. (2006). Frequency
       recognition based on canonical correlation analysis for SSVEP-based
       BCIs. IEEE Transactions on Biomedical Engineering, 53(12), 2610-2614.
       https://doi.org/10.1109/TBME.2006.886577

    Examples
    --------
    >>> import numpy as np
    >>> from aeon_neuro.classification.feature_based import SSVEPCCAClassifier
    >>> sfreq, n_timepoints = 250, 500
    >>> t = np.arange(n_timepoints) / sfreq
    >>> rng = np.random.default_rng(0)
    >>> freqs = {0: 8, 1: 13}
    >>> y = rng.integers(0, 2, size=20)
    >>> X = rng.standard_normal((20, 3, n_timepoints))
    >>> for i, label in enumerate(y):
    ...     X[i] += np.sin(2 * np.pi * freqs[label] * t)
    >>> clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs)
    >>> _ = clf.fit(X, y)
    >>> preds = clf.predict(X)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "feature",
    }

    def __init__(self, sfreq, class_frequencies, n_harmonics=2, solver="nipals"):
        self.sfreq = sfreq
        self.class_frequencies = class_frequencies
        self.n_harmonics = n_harmonics
        self.solver = solver
        super().__init__()

    def _fit(self, X, y=None):
        """Validate parameters and build a reference signal for each class.

        The classifier is training-free: references are built from
        ``class_frequencies`` and the series length only, so ``y`` is not used
        here (the class set and ordering come from ``self.classes_``). It is
        accepted for compatibility with the classifier API.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training cases.
        y : np.ndarray of shape (n_cases,), optional
            Class labels. Unused; present only for API compatibility.

        Returns
        -------
        self : SSVEPCCAClassifier
            Fitted classifier.
        """
        if (
            not isinstance(self.sfreq, numbers.Real)
            or isinstance(self.sfreq, bool)
            or not np.isfinite(self.sfreq)
            or self.sfreq <= 0
        ):
            raise ValueError(f"sfreq must be finite and positive, got {self.sfreq!r}.")

        if (
            not isinstance(self.n_harmonics, numbers.Integral)
            or isinstance(self.n_harmonics, bool)
            or self.n_harmonics < 1
        ):
            raise ValueError(
                f"n_harmonics must be a positive integer, got {self.n_harmonics!r}."
            )

        if not self.class_frequencies:
            raise ValueError("class_frequencies must contain at least one class.")

        nyquist = self.sfreq / 2
        for label, freq in self.class_frequencies.items():
            if (
                not isinstance(freq, numbers.Real)
                or isinstance(freq, bool)
                or not np.isfinite(freq)
                or freq <= 0
            ):
                raise ValueError(
                    f"Frequency of class {label!r} must be finite and positive, "
                    f"got {freq!r}."
                )
            highest_harmonic = self.n_harmonics * freq
            if highest_harmonic >= nyquist:
                raise ValueError(
                    f"Harmonic {self.n_harmonics} of class {label!r} is "
                    f"{highest_harmonic} Hz, which must be below the Nyquist "
                    f"frequency (sfreq / 2 = {nyquist})."
                )

        frequencies = list(self.class_frequencies.values())
        if len(set(frequencies)) != len(frequencies):
            raise ValueError(
                f"class_frequencies has duplicate frequencies: "
                f"{self.class_frequencies}."
            )

        missing = [c for c in self.classes_.tolist() if c not in self.class_frequencies]
        if missing:
            raise ValueError(f"Labels {missing} in y are not in class_frequencies.")

        return self

    def _build_references(self, n_timepoints):
        """Build a sine and cosine reference signal for each class.

        References are in ``self.classes_`` order so argmax maps back to the
        right label.

        Parameters
        ----------
        n_timepoints : int
            Length of the cases the references are compared with.

        Returns
        -------
        list of np.ndarray of shape (2 * n_harmonics, n_timepoints)
            One reference signal per class.
        """
        t = np.arange(n_timepoints) / self.sfreq
        references = []
        for label in self.classes_:
            freq = self.class_frequencies[label]
            reference = []
            for h in range(1, self.n_harmonics + 1):
                reference.append(np.sin(2 * np.pi * h * freq * t))
                reference.append(np.cos(2 * np.pi * h * freq * t))
            references.append(np.array(reference))
        return references

    def _cca_scores(self, X, references):
        """Canonical correlation between each case and every class reference.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Cases to score.
        references : list of np.ndarray of shape (2 * n_harmonics, n_timepoints)
            One reference signal per class, in ``self.classes_`` order.

        Returns
        -------
        np.ndarray of shape (n_cases, n_classes)
            ``scores[i, j]`` is the canonical correlation between case ``i`` and
            the reference of ``self.classes_[j]``.
        """
        scores = np.zeros((len(X), len(self.classes_)))
        if self.solver == "nipals":
            cca = CCA(n_components=1)
            for i, case in enumerate(X):
                for j, reference in enumerate(references):
                    S_x, S_y = cca.fit_transform(case.T, reference.T)
                    scores[i, j] = np.corrcoef(S_x.ravel(), S_y.ravel())[0, 1]
            scores = np.nan_to_num(scores, nan=0.0)
        elif self.solver == "cf":
            q_ys = [_qr_basis(ref.T) for ref in references]
            for i, case in enumerate(X):
                q_x = _qr_basis(case.T)
                if q_x is None:
                    continue  # flat trial, score stays 0
                for j, q_y in enumerate(q_ys):
                    s = np.linalg.svd(q_x.T @ q_y, compute_uv=False)
                    scores[i, j] = min(s[0], 1.0)
        return scores

    def _predict(self, X):
        """Predict the class of each case by maximum canonical correlation.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Cases to classify.

        Returns
        -------
        np.ndarray of shape (n_cases,)
            Predicted class label for each case.
        """
        references = self._build_references(X.shape[2])
        return self.classes_[np.argmax(self._cca_scores(X, references), axis=1)]

    def _predict_proba(self, X):
        """Return class probabilities from the canonical correlations.

        The non-negative correlations of each case are normalised to sum to one.
        A case whose correlations are all non-positive is given a uniform
        distribution. These are relative correlation strengths, not calibrated
        probabilities, but their ``argmax`` agrees with ``predict``.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Cases to classify.

        Returns
        -------
        np.ndarray of shape (n_cases, n_classes)
            Probability of each class for each case, summing to one per case.
        """
        references = self._build_references(X.shape[2])
        scores = np.clip(self._cca_scores(X, references), 0, None)
        row_sums = scores.sum(axis=1, keepdims=True)
        safe = np.where(row_sums > 0, row_sums, 1)
        proba = scores / safe
        proba[row_sums.ravel() == 0] = 1 / len(self.classes_)
        return proba

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
        return {"sfreq": 100, "class_frequencies": {0: 8, 1: 13, 2: 17}}
