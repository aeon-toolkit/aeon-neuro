"""Common Spatial Patterns spatial filtering, native NumPy/SciPy implementation."""

from __future__ import annotations

__all__ = ["CommonSpatialPatterns"]

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from scipy import linalg


class CommonSpatialPatterns(BaseCollectionTransformer):
    """Common Spatial Patterns (CSP) spatial filter.

    CSP learns spatial filters (linear combinations of channels) whose output
    variance best discriminates between classes [1]_ [2]_. This transformer is a
    pure spatial filter: it projects each case onto the first ``n_components``
    filters and returns the filtered signals, of shape
    (n_cases, n_components, n_timepoints). Feature extraction, such as the
    classic log-variance of each component, is left to downstream transformers,
    for example ``LogVariance``.

    The algorithm follows ``mne.decoding.CSP`` but is implemented with NumPy and
    SciPy only:

    - Class covariances are estimated (optionally regularised) for each class.
    - Two classes: the generalised eigenvalue problem
      ``C_1 w = lambda (C_1 + C_2) w`` is solved with ``scipy.linalg.eigh``.
    - More than two classes: the class covariances are jointly diagonalised with
      Pham's approximate joint diagonalisation (AJD) [3]_ [4]_.

    If the data is rank deficient (for example after average referencing),
    covariances are first restricted to the principal subspace of the mean class
    covariance, so at most ``rank`` filters are learned.

    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components (spatial filters) to keep. Must be at most the
        number of channels (or the rank of the data, if lower).
    reg : float, str or None, default=None
        Covariance regularisation.

        - None or "empirical": empirical covariance.
        - float in [0, 1]: shrinkage towards a scaled identity with this
          shrinkage coefficient (``sklearn.covariance.ShrunkCovariance``).
        - "ledoit_wolf": Ledoit-Wolf shrinkage (``sklearn.covariance.LedoitWolf``).
        - "oas": Oracle Approximating Shrinkage (``sklearn.covariance.OAS``).

        As in MNE, covariances are computed without removing the mean
        (``assume_centered=True``), so the input should be band-pass filtered,
        and are scaled by ``n / (n - 1)``.
    cov_est : {"concat", "epoch"}, default="concat"
        "concat" concatenates all cases of a class before estimating its
        covariance; "epoch" averages the covariances of the individual cases.
    norm_trace : bool, default=False
        If True, normalise each class covariance by its trace.
    cov_method_params : dict or None, default=None
        Extra keyword arguments passed to the scikit-learn covariance estimator
        selected by ``reg``, for example ``{"block_size": 500}`` for
        "ledoit_wolf".
    component_order : {"mutual_info", "alternate"}, default="mutual_info"
        How filters are ordered before the first ``n_components`` are kept.

        - "mutual_info": for two classes, by ``|eigenvalue - 0.5|`` in decreasing
          order (most discriminative first); for more classes, by decreasing
          approximate mutual information with the class label [4]_.
        - "alternate": two classes only; alternates between the filters with the
          largest and smallest eigenvalues.
    random_state : int or None, default=None
        Included for API consistency. Not used; the algorithm is deterministic.

    Attributes
    ----------
    filters_ : np.ndarray of shape (n_filters, n_channels)
        All learned spatial filters, one per row, in ``component_order``.
        ``n_filters`` is the number of channels, or the rank of the data if it is
        rank deficient. ``transform`` uses the first ``n_components`` rows.
    patterns_ : np.ndarray of shape (n_filters, n_channels)
        The spatial patterns (pseudo-inverse of the filters), for visualising the
        sources each filter extracts.
    evals_ : np.ndarray of shape (n_filters,) or None
        The generalised eigenvalues, in the same order as ``filters_``, for two
        classes. None for more than two classes.
    classes_ : np.ndarray
        The class labels seen in ``fit``.

    Notes
    -----
    - CSP is supervised, so ``fit`` requires ``y``.
    - This is intended for equal-length collections with shape
      (n_cases, n_channels, n_timepoints).

    References
    ----------
    .. [1] Koles, Z. J., Lazar, M. S., and Zhou, S. Z. (1990).
       Spatial patterns underlying population differences in the background EEG.
       Brain Topography, 2(4), 275-284.
    .. [2] Ramoser, H., Muller-Gerking, J., and Pfurtscheller, G. (2000).
       Optimal spatial filtering of single trial EEG during imagined hand
       movement. IEEE Transactions on Rehabilitation Engineering, 8(4),
       441-446. https://doi.org/10.1109/86.895946
    .. [3] Pham, D. T. (2001). Joint approximate diagonalization of positive
       definite Hermitian matrices. SIAM Journal on Matrix Analysis and
       Applications, 22(4), 1136-1152.
    .. [4] Grosse-Wentrup, M., and Buss, M. (2008).
       Multiclass common spatial patterns and information theoretic feature
       extraction. IEEE Transactions on Biomedical Engineering, 55(8), 1991-2000.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection import LogVariance
    >>> from aeon_neuro.transformations.collection.channel_creation import (
    ...     CommonSpatialPatterns,
    ... )
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=12, n_channels=4, n_timepoints=64, random_state=0
    ... )
    >>> csp = CommonSpatialPatterns(n_components=2)
    >>> Xt = csp.fit_transform(X, y)
    >>> Xt.shape
    (12, 2, 64)
    >>> csp.filters_.shape
    (4, 4)
    >>> LogVariance().fit_transform(Xt).shape
    (12, 2)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,
        "capability:multivariate": True,
        "requires_y": True,
    }

    def __init__(
        self,
        n_components: int = 4,
        reg: float | str | None = None,
        cov_est: str = "concat",
        norm_trace: bool = False,
        cov_method_params: dict | None = None,
        component_order: str = "mutual_info",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.reg = reg
        self.cov_est = cov_est
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Learn the CSP spatial filters from a labelled 3D collection."""
        if y is None:
            raise ValueError("CommonSpatialPatterns requires y in fit.")
        y = np.asarray(y)
        X = np.asarray(X, dtype=np.float64)
        self._validate_params()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                f"CommonSpatialPatterns requires at least two classes, got "
                f"{n_classes}."
            )
        if n_classes > 2 and self.component_order == "alternate":
            raise ValueError(
                "component_order='alternate' requires two classes, but y has "
                f"{n_classes}; use component_order='mutual_info' instead."
            )

        covs, sample_weights = self._class_covariances(X, y)
        restr = _restriction_matrix(X, covs.mean(axis=0))
        covs_r = restr @ covs @ restr.T

        if n_classes == 2:
            # GED of C_1 against C_1 + C_2, eigenvalues in [0, 1]
            evals, evecs_r = linalg.eigh(covs_r[0], covs_r[0] + covs_r[1])
        else:
            evals = None
            diagonaliser = _ajd_pham(covs_r)
            evecs_r = _normalise_eigenvectors(diagonaliser.T, covs_r, sample_weights)
        evecs = restr.T @ evecs_r

        order = self._component_order(evals, evecs, covs, sample_weights)
        evecs = evecs[:, order]
        self.evals_ = None if evals is None else evals[order]
        self.filters_ = evecs.T
        self.patterns_ = linalg.pinv(evecs)

        if self.n_components > self.filters_.shape[0]:
            raise ValueError(
                f"n_components={self.n_components} exceeds the number of CSP "
                f"filters ({self.filters_.shape[0]}), which is the number of "
                "channels or the rank of the data if lower."
            )
        return self

    def _transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Project X onto the first ``n_components`` CSP filters."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.filters_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} channels, but CommonSpatialPatterns was "
                f"fitted on {self.filters_.shape[1]} channels."
            )
        # (n_components, n_channels) @ (n_cases, n_channels, n_timepoints)
        return self.filters_[: self.n_components] @ X

    def _validate_params(self):
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, (int, np.integer))
            or self.n_components < 1
        ):
            raise ValueError(
                f"n_components must be a positive int, got {self.n_components!r}."
            )
        if self.cov_est not in ("concat", "epoch"):
            raise ValueError(
                f"cov_est must be 'concat' or 'epoch', got {self.cov_est!r}."
            )
        if self.component_order not in ("mutual_info", "alternate"):
            raise ValueError(
                "component_order must be 'mutual_info' or 'alternate', got "
                f"{self.component_order!r}."
            )
        _covariance_estimator(self.reg, self.cov_method_params)

    def _class_covariances(self, X, y):
        """Return class covariances and the class weights used to combine them."""
        estimator = _covariance_estimator(self.reg, self.cov_method_params)
        n_channels = X.shape[1]
        covs, weights = [], []
        for label in self.classes_:
            X_class = X[y == label]
            if self.cov_est == "concat":
                concat = X_class.transpose(1, 0, 2).reshape(n_channels, -1)
                cov = estimator(concat)
                # as in MNE, "concat" weights every class equally
                weights.append(n_channels)
            else:
                cov = np.mean([estimator(case) for case in X_class], axis=0)
                weights.append(len(X_class))
            if self.norm_trace:
                cov = cov / np.trace(cov)
            covs.append(cov)
        return np.stack(covs), np.asarray(weights, dtype=np.float64)

    def _component_order(self, evals, evecs, covs, sample_weights):
        """Return indices ordering the filters by ``component_order``."""
        if self.component_order == "alternate":
            ix = np.argsort(evals)
            order = np.empty_like(ix)
            order[1::2] = ix[: len(ix) // 2]
            order[0::2] = ix[len(ix) // 2 :][::-1]
            return order
        if evals is not None:
            return np.argsort(np.abs(evals - 0.5))[::-1]
        mutual_info = _mutual_information(covs, sample_weights, evecs)
        return np.argsort(mutual_info)[::-1]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        return [
            {"n_components": 2},
            {"n_components": 2, "reg": "ledoit_wolf", "cov_est": "epoch"},
        ]


def _covariance_estimator(reg, cov_method_params):
    """Return a function mapping (n_channels, n_samples) data to a covariance."""
    from sklearn.covariance import (
        OAS,
        EmpiricalCovariance,
        LedoitWolf,
        ShrunkCovariance,
    )

    params = {} if cov_method_params is None else dict(cov_method_params)
    if reg is None or (isinstance(reg, str) and reg == "empirical"):
        estimator = EmpiricalCovariance
    elif isinstance(reg, str):
        if reg == "ledoit_wolf":
            estimator = LedoitWolf
        elif reg == "oas":
            estimator = OAS
        else:
            raise ValueError(
                "reg must be None, a float in [0, 1], 'empirical', 'ledoit_wolf' "
                f"or 'oas', got {reg!r}."
            )
    elif isinstance(reg, (int, float, np.integer, np.floating)) and not isinstance(
        reg, bool
    ):
        if not 0 <= reg <= 1:
            raise ValueError(f"A float reg must be in [0, 1], got {reg}.")
        estimator = ShrunkCovariance
        params["shrinkage"] = float(reg)
    else:
        raise ValueError(
            "reg must be None, a float in [0, 1], 'empirical', 'ledoit_wolf' "
            f"or 'oas', got {reg!r}."
        )
    params.update(store_precision=False, assume_centered=True)

    def _estimate(data):
        # unbiased n / (n - 1) scaling, as in MNE
        n_samples = data.shape[1]
        cov = estimator(**params).fit(data.T).covariance_
        return cov * (n_samples / max(n_samples - 1, 1))

    return _estimate


def _restriction_matrix(X, C_ref):
    """Return a (rank, n_channels) matrix onto the principal subspace of C_ref.

    The rank is estimated from the singular values of the concatenated data, with
    the same tolerance as MNE. For full-rank data this is the identity.
    """
    n_channels = X.shape[1]
    if _is_clearly_full_rank(X):
        return np.eye(n_channels)
    rank = _data_rank(X)
    if rank == n_channels:
        return np.eye(n_channels)
    _, evecs = linalg.eigh(C_ref)
    return evecs[:, ::-1][:, :rank].T


def _is_clearly_full_rank(X):
    """Cheaply decide whether the concatenated data is certainly full rank.

    Uses the eigenvalues of the (n_channels, n_channels) Gram matrix
    ``sum_i X_i X_i^T``, which is one matrix product. Its rounding error is at
    most about ``n_samples * n_channels * eps * lambda_max``, so an eigenvalue
    well above that bound is certainly positive. A smallest eigenvalue that clear
    of zero is also far above the MNE rank tolerance, which on the eigenvalue
    scale is ``lambda_max * (n_channels * eps) ** 2``, so ``_data_rank`` would
    report full rank as well. If this returns False the data may still be full
    rank, and ``_data_rank`` decides.
    """
    n_cases, n_channels, n_timepoints = X.shape
    gram = np.tensordot(X, X, axes=([0, 2], [0, 2]))
    evals = linalg.eigvalsh(gram, check_finite=False)
    if evals[-1] <= 0:
        return False
    bound = 10 * n_cases * n_timepoints * n_channels * np.finfo(np.float64).eps
    return bool(evals[0] > evals[-1] * bound)


def _data_rank(X):
    """Return the rank of the concatenated data using MNE's tolerance.

    The singular values are those of the triangular factor R of the
    (n_samples, n_channels) data (D = QR, so D and R share singular values),
    which is cheaper than an SVD of the full data and equally accurate.
    """
    n_channels = X.shape[1]
    data = X.transpose(0, 2, 1).reshape(-1, n_channels)  # (n_samples, n_channels)
    (R,) = linalg.qr(data, mode="r", check_finite=False)
    s = linalg.svdvals(R, check_finite=False)
    return int(np.sum(s > s.max() * n_channels * np.finfo(np.float64).eps))


def _normalise_eigenvectors(evecs, covs, sample_weights):
    """Scale each eigenvector to unit variance under the weighted mean covariance."""
    mean_cov = np.average(covs, axis=0, weights=sample_weights)
    scale = np.sqrt(np.einsum("ck,cd,dk->k", evecs, mean_cov, evecs))
    return evecs / scale


def _mutual_information(covs, sample_weights, evecs):
    """Approximate mutual information of each filter with the class label.

    Information theoretic feature extraction of Grosse-Wentrup and Buss (2008).
    """
    class_probs = sample_weights / sample_weights.sum()
    # variance of each filter output under each class, shape (n_classes, n_filters)
    var = np.einsum("ck,ncd,dk->nk", evecs, covs, evecs)
    aa = np.sum(class_probs[:, None] * np.log(np.sqrt(var)), axis=0)
    bb = np.sum(class_probs[:, None] * (var**2 - 1), axis=0)
    return -(aa + (3.0 / 16) * bb**2)


def _ajd_pham(covs, eps=1e-6, max_iter=15):
    """Approximate joint diagonalisation of SPD matrices using Pham's algorithm.

    Port of ``mne.decoding.csp._ajd_pham`` (BSD-3-Clause), itself adapted from
    pyRiemann (BSD-3-Clause), using the same tolerance and iteration limit.

    Parameters
    ----------
    covs : np.ndarray of shape (n_matrices, n_channels, n_channels)
        Symmetric positive definite matrices to jointly diagonalise.
    eps : float, default=1e-6
        Tolerance for the stopping criterion.
    max_iter : int, default=15
        Maximum number of sweeps.

    Returns
    -------
    V : np.ndarray of shape (n_channels, n_channels)
        The joint diagonaliser, such that ``V @ covs[i] @ V.T`` is approximately
        diagonal for every ``i``.
    """
    n_matrices, n_channels, _ = covs.shape
    A = np.concatenate(covs, axis=0).T
    n_cols = A.shape[1]
    V = np.eye(n_channels)
    epsilon = n_channels * (n_channels - 1) * eps

    for _ in range(max_iter):
        decr = 0.0
        for ii in range(1, n_channels):
            for jj in range(ii):
                Ii = np.arange(ii, n_cols, n_channels)
                Ij = np.arange(jj, n_cols, n_channels)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_matrices * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.0j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp**2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = tau @ A[[ii, jj], :]
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_channels * n_matrices, 2), order="F")
                tmp = tmp @ tau.T
                tmp = np.reshape(tmp, (n_channels, n_matrices * 2), order="F")
                A[:, Ii] = tmp[:, :n_matrices]
                A[:, Ij] = tmp[:, n_matrices:]
                V[[ii, jj], :] = tau @ V[[ii, jj], :]
        if decr < epsilon:
            break
    return V
