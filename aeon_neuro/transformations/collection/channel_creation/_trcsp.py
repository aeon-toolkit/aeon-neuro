"""Tikhonov-Regularised Common Spatial Patterns spatial filtering."""

from __future__ import annotations

__all__ = ["TRCommonSpatialPatterns"]

import numbers

import numpy as np
from scipy import linalg

from aeon_neuro.transformations.collection.channel_creation._csp import (
    CommonSpatialPatterns,
)


class TRCommonSpatialPatterns(CommonSpatialPatterns):
    """Tikhonov-Regularised Common Spatial Patterns (TRCSP) spatial filter.

    TRCSP extends Common Spatial Patterns with Tikhonov regularisation of the
    spatial filters, which improves filter estimation when training data is
    limited [1]_. Like ``CommonSpatialPatterns``, it projects each case onto
    the learned filters and returns the filtered signals, of shape
    (n_cases, n_components, n_timepoints).

    Only binary classification is supported. Half of the filters maximise the
    variance of each class, relative to the regularised variance of the other.

    Parameters
    ----------
    n_components : int, default=4
        Number of spatial filters to keep. Must be even, as half come from each
        class, and at most twice the number of channels.
    alpha : float, default=1.0
        Strength of the Tikhonov regularisation. Must be non-negative. With
        ``alpha=0`` the filters are those of unregularised CSP; larger values
        favour filters with smaller weights.

    Attributes
    ----------
    filters_ : np.ndarray of shape (n_components, n_channels)
        The learned spatial filters, one per row. The first
        ``n_components // 2`` maximise the variance of ``classes_[1]``, the rest
        that of ``classes_[0]``, each ordered most discriminative first.
    classes_ : np.ndarray
        The class labels seen in ``fit``.

    Notes
    -----
    For each class, the covariance of every case is normalised by its trace and
    the results are averaged, giving C_a and C_b. The filters that maximise the
    variance of class b solve::

        max_w  (w^T C_b w) / (w^T C_a w + alpha * ||w||^2)

    which is the generalised eigenproblem C_b w = lambda (C_a + alpha I) w. The
    eigenvectors with the largest eigenvalues are kept, and the same is done
    with the classes swapped. The trace normalisation makes ``alpha``
    independent of the scale of the signals.

    This follows the ``TRCSP`` of MOABB, with these differences:

    - MOABB takes covariance matrices as input; this takes EEG collections and
      computes trace-normalised covariances itself.
    - MOABB returns log-variance features; this returns the filtered signals.
      Apply ``LogVariance`` afterwards to get the same features.
    - MOABB silently returns one filter fewer for an odd ``nfilter``; this
      raises a ``ValueError``.

    References
    ----------
    .. [1] Lotte, F., and Guan, C. (2011). Regularizing common spatial patterns
       to improve BCI designs: unified theory and new algorithms. IEEE
       Transactions on Biomedical Engineering, 58(2), 355-362.
       https://doi.org/10.1109/TBME.2010.2082539

    Examples
    --------
    >>> from aeon.classification.compose import ClassifierPipeline
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from aeon_neuro.transformations.collection import LogVariance
    >>> from aeon_neuro.transformations.collection.channel_creation import (
    ...     TRCommonSpatialPatterns,
    ... )
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=12, n_channels=4, n_timepoints=64, random_state=0
    ... )
    >>> trcsp = TRCommonSpatialPatterns(n_components=2, alpha=0.1)
    >>> trcsp.fit_transform(X, y).shape
    (12, 2, 64)
    >>> pipeline = ClassifierPipeline(
    ...     [TRCommonSpatialPatterns(n_components=2, alpha=0.1), LogVariance()],
    ...     LinearDiscriminantAnalysis(),
    ... )
    >>> pipeline.fit(X, y).predict(X).shape
    (12,)
    """

    def __init__(self, n_components: int = 4, alpha: float = 1.0):
        self.alpha = alpha
        super().__init__(n_components=n_components)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Learn the TRCSP spatial filters from a labelled 3D collection.

        For each class, the trace-normalised covariances of its cases are
        averaged. Then two regularised generalised eigenproblems are solved,
        one per class, and the top ``n_components // 2`` filters of each are
        kept.
        """
        self._validate_params()  # inherited: n_components is a positive int
        if self.n_components % 2:
            raise ValueError(
                f"n_components must be even, got {self.n_components}: half the "
                "filters come from each class."
            )
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, numbers.Real)
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError(
                f"alpha must be a non-negative number, got {self.alpha!r}."
            )

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "TRCommonSpatialPatterns only supports binary classification, "
                f"but y contains {len(self.classes_)} classes."
            )

        n_channels = X.shape[1]
        n_half = self.n_components // 2
        if n_half > n_channels:
            raise ValueError(
                f"n_components={self.n_components} needs {n_half} filters per "
                f"class, but X has only {n_channels} channels."
            )

        X_a = X[y == self.classes_[0]]
        X_b = X[y == self.classes_[1]]

        # mean of the trace-normalised covariances of each class
        Cov_a = np.mean([(c := np.cov(trial)) / np.trace(c) for trial in X_a], axis=0)
        Cov_b = np.mean([(c := np.cov(trial)) / np.trace(c) for trial in X_b], axis=0)

        eye = np.eye(n_channels)

        reg_term_a = Cov_a + eye * self.alpha
        reg_term_b = Cov_b + eye * self.alpha

        eigvals_a, eigvecs_a = linalg.eigh(Cov_a, reg_term_b)
        eigvals_b, eigvecs_b = linalg.eigh(Cov_b, reg_term_a)

        # sort eigenvectors in descending order of eigenvalues
        W_a = eigvecs_a[:, np.argsort(eigvals_a)[::-1]]
        W_b = eigvecs_b[:, np.argsort(eigvals_b)[::-1]]

        # top m filters from each class as rows, class b first as in MOABB
        m = self.n_components // 2
        self.filters_ = np.vstack([W_b[:, :m].T, W_a[:, :m].T])
        return self

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        return [
            {"n_components": 2, "alpha": 1.0},
            {"n_components": 4, "alpha": 0.1},
        ]
