"""Riemannian spatial filtering."""

from __future__ import annotations

__all__ = ["RiemannianSpatialFilter"]

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from scipy import linalg

_SOLVERS = ("eig", "trust-constr")


class RiemannianSpatialFilter(BaseCollectionTransformer):
    """Riemannian Spatial Filter (RSF) for two-class problems.

    Learns a set of spatial filters (linear combinations of channels) that
    maximise the Riemannian distance between the two class mean covariance
    matrices after filtering [1]_. The first ``n_components`` filters are used
    to project the series into a lower-dimensional channel space.

    By default the filters are found in closed form, as the generalised
    eigenvectors of the two Riemannian class means ranked by
    ``|log(eigenvalue)|``. They can instead be found by numerical optimisation
    of the same objective, as in [1]_, by setting ``solver``.

    Parameters
    ----------
    n_components : int, default=4
        Number of spatial filters to keep. Must be at most the number of
        channels.
    covariance_estimator : str, default="scm"
        Covariance estimator passed to ``pyriemann.estimation.Covariances``.
    solver : {"eig", "trust-constr"}, default="eig"
        How the filters are found.

        - ``"eig"``: closed form generalised eigendecomposition. Exact and
          deterministic; ``init``, ``max_iter`` and ``random_state`` are
          ignored.
        - ``"trust-constr"``: numerical optimisation with the trust-region
          method of ``scipy.optimize.minimize``, the default in [1]_.
    init : {"random", "csp"}, default="random"
        Starting filters for the numerical solver. ``"random"`` uses a random
        orthonormal matrix, ``"csp"`` uses the first ``n_components`` Common
        Spatial Pattern filters. If the optimiser ends further from the
        optimum than it started, the starting filters are kept.
    max_iter : int, default=1000
        Maximum number of iterations of the numerical solver.
    random_state : int, np.random.RandomState or None, default=None
        Seed or random number generator for the ``"random"`` initialisation.

    Attributes
    ----------
    filters_ : np.ndarray of shape (n_filters, n_channels)
        The learned spatial filters, one per row, ordered by decreasing
        ``|log(evals_)|``. ``transform`` uses the first ``n_components`` rows.
        ``n_filters`` is ``n_channels`` for the ``"eig"`` solver and
        ``n_components`` for the numerical solver.
    evals_ : np.ndarray of shape (n_filters,)
        The generalised eigenvalues of the class means along each filter, in
        the same order as ``filters_``.
    classes_ : np.ndarray
        The class labels seen in ``fit``.

    Notes
    -----
    Each case X_i, of shape (n_channels, n_timepoints), is summarised by its
    spatial covariance matrix C_i, of shape (n_channels, n_channels). For each
    class k in {a, b} the Riemannian (geometric) mean M_k is the matrix that
    minimises the sum of squared Riemannian distances to the covariances of
    that class::

        M_k = argmin_M  sum_{i : y_i = k}  d(M, C_i)^2

    The affine-invariant Riemannian distance between two symmetric positive
    definite matrices A and B is::

        d(A, B) = || log(A^(-1/2) B A^(-1/2)) ||_F
                = sqrt( sum_j log(lambda_j)^2 )

    where lambda_j are the generalised eigenvalues of (A, B).

    RSF seeks the filter matrix W, of shape (n_channels, n_components), that
    keeps the filtered class means as far apart as possible::

        W = argmax_W  d(W^T M_a W, W^T M_b W)^2

    The solution is given by the generalised eigenvalue problem::

        M_a w_j = lambda_j M_b w_j

        lambda_j = (w_j^T M_a w_j) / (w_j^T M_b w_j)

    so each lambda_j is the ratio of the class variances along filter w_j.
    Taking the n_components eigenvectors with the largest |log(lambda_j)| as
    the columns of W gives::

        d(W^T M_a W, W^T M_b W)^2 = sum_{j=1}^{n_components} log(lambda_j)^2

    which is the largest value attainable with n_components filters. A case
    is then transformed as Z_i = W^T X_i, of shape
    (n_components, n_timepoints).

    The numerical solver maximises the same squared distance directly over W.
    The distance is unchanged by any invertible mixing of the columns of W,
    so the optimised filters are then rotated onto the generalised
    eigenvectors of the filtered class means. This keeps the span found by
    the optimiser while making ``filters_`` and ``evals_`` ordered and
    comparable with those of the ``"eig"`` solver.

    References
    ----------
    .. [1] Pan, L., Wang, K., Huang, Y., Sun, X., Meng, J., Yi, W., Xu, M.,
       Jung, T.-P., and Ming, D. (2025). Enhancing motor imagery EEG
       classification with a Riemannian geometry-based spatial filtering
       (RSF) method. Neural Networks, 188, 107511.
       https://doi.org/10.1016/j.neunet.2025.107511

    Examples
    --------
    >>> from aeon.classification.compose import ClassifierPipeline
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification import CSPClassifier, RiemannianMDMClassifier
    >>> from aeon_neuro.transformations.collection.channel_creation import (
    ...     RiemannianSpatialFilter,
    ... )
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=12, n_channels=4, n_timepoints=64, random_state=0
    ... )
    >>> rsf = RiemannianSpatialFilter(n_components=2)
    >>> Xt = rsf.fit_transform(X, y)
    >>> Xt.shape
    (12, 2, 64)
    >>> rsf.filters_.shape
    (4, 4)

    RSF is used as a preprocessing step before a covariance based classifier,
    such as RSF followed by Minimum Distance to Mean (MDM). MDM on all channels
    and CSP+LDA are the usual baselines to compare it with.

    >>> rsf_mdm = ClassifierPipeline(
    ...     [RiemannianSpatialFilter(n_components=2)], RiemannianMDMClassifier()
    ... )
    >>> rsf_mdm.fit(X, y).predict(X).shape
    (12,)
    >>> RiemannianMDMClassifier().fit(X, y).predict(X).shape
    (12,)
    >>> CSPClassifier(n_components=2).fit(X, y).predict(X).shape
    (12,)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,
        "capability:multivariate": True,
        "requires_y": True,
        "fit_is_empty": False,
        "python_dependencies": ["pyriemann", "torch"],
    }

    def __init__(
        self,
        n_components=4,
        covariance_estimator="scm",
        solver="eig",
        init="random",
        max_iter=1000,
        random_state=None,
    ):
        self.n_components = n_components
        self.covariance_estimator = covariance_estimator
        self.solver = solver
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the Riemannian Spatial Filter.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series.
        y : np.ndarray of shape (n_cases,)
            Target variable for each case.

        Returns
        -------
        self
            Fitted transformer.
        """
        from pyriemann.utils.mean import mean_covariance

        from aeon_neuro.utils._riemannian import (
            build_covariance_transformer,
            compute_covariance_matrices,
        )

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "RiemannianSpatialFilter needs two classes to fit, but y contains "
                f"only one class ({self.classes_[0]})."
            )
        if n_classes > 2:
            raise ValueError(
                "RiemannianSpatialFilter only supports binary classification, "
                f"but y contains {n_classes} classes."
            )

        n_channels = X.shape[1]
        if isinstance(self.n_components, bool) or not isinstance(
            self.n_components, (int, np.integer)
        ):
            raise ValueError(
                f"n_components must be an integer, got {self.n_components!r}."
            )
        if not 1 <= self.n_components <= n_channels:
            raise ValueError(
                f"n_components must be between 1 and n_channels ({n_channels}), "
                f"got {self.n_components}."
            )
        if self.solver not in _SOLVERS:
            raise ValueError(
                f"solver must be one of {list(_SOLVERS)}, got {self.solver!r}."
            )
        if self.solver != "eig":
            if self.init not in ("random", "csp"):
                raise ValueError(
                    f"init must be one of ['csp', 'random'], got {self.init!r}."
                )
            if (
                isinstance(self.max_iter, bool)
                or not isinstance(self.max_iter, (int, np.integer))
                or self.max_iter < 1
            ):
                raise ValueError(
                    f"max_iter must be a positive integer, got {self.max_iter!r}."
                )

        # Plain spatial covariances only: ERP covariances are larger than
        # (n_channels, n_channels), so their filters could not be applied to X.
        self.transformer_ = build_covariance_transformer(
            "cov", self.covariance_estimator
        )
        X_cov = compute_covariance_matrices(X, self.transformer_, fit=True, y=y)

        # The Riemannian mean and distance are only defined for symmetric
        # positive definite matrices, e.g. not when n_timepoints < n_channels.
        cov_evals = np.linalg.eigvalsh(X_cov)
        tol = n_channels * np.finfo(X_cov.dtype).eps * cov_evals[:, -1]
        if not np.all(np.isfinite(cov_evals)) or np.any(cov_evals[:, 0] <= tol):
            raise ValueError(
                "RiemannianSpatialFilter requires positive definite covariance "
                "matrices, but at least one case has a singular covariance. Use "
                "more time points than channels, remove linearly dependent "
                "channels, or use a shrinkage covariance_estimator such as 'lwf'."
            )

        # Riemannian (geometric) mean of each class
        mean_a = mean_covariance(X_cov[y == self.classes_[0]], metric="riemann")
        mean_b = mean_covariance(X_cov[y == self.classes_[1]], metric="riemann")

        # The squared Riemannian distance between the filtered class means is
        # sum(log(evals) ** 2), so the most discriminative filters are those
        # with the largest |log(eval)|.
        if self.solver == "eig":
            evals, evecs = linalg.eigh(mean_a, mean_b)
        else:
            W = self._optimise(mean_a, mean_b, X_cov, y)
            # Rotate the optimised filters onto the generalised eigenvectors
            # of the filtered means. This leaves their span, and so the
            # distance, unchanged but gives each filter its own eigenvalue.
            evals, rotation = linalg.eigh(W.T @ mean_a @ W, W.T @ mean_b @ W)
            evecs = W @ rotation
        order = np.argsort(np.abs(np.log(evals)))[::-1]

        self.evals_ = evals[order]
        self.filters_ = evecs[:, order].T

        return self

    def _optimise(self, mean_a, mean_b, X_cov, y):
        """Numerically maximise the distance between the filtered class means.

        Parameters
        ----------
        mean_a, mean_b : np.ndarray of shape (n_channels, n_channels)
            Riemannian mean covariance of each class.
        X_cov : np.ndarray of shape (n_cases, n_channels, n_channels)
            Covariance matrix of each case, used by the CSP initialisation.
        y : np.ndarray of shape (n_cases,)
            Target variable for each case, used by the CSP initialisation.

        Returns
        -------
        W : np.ndarray of shape (n_channels, n_components)
            The filters, one per column.
        """
        from scipy.optimize import minimize
        from sklearn.utils import check_random_state

        n_channels = mean_a.shape[0]
        shape = (n_channels, self.n_components)

        if self.init == "csp":
            from pyriemann.spatialfilters import CSP

            csp = CSP(nfilter=self.n_components, metric="euclid")
            W0 = csp.fit(X_cov, y).filters_.T
        else:
            rng = check_random_state(self.random_state)
            W0 = np.linalg.qr(rng.standard_normal(shape))[0]

        def negative_squared_distance(w_flat):
            W = w_flat.reshape(shape)
            try:
                evals = linalg.eigh(
                    W.T @ mean_a @ W, W.T @ mean_b @ W, eigvals_only=True
                )
            except linalg.LinAlgError:
                # W has lost rank, so the filtered means are not comparable
                return 0.0
            evals = np.clip(evals, 1e-10, None)
            return -np.sum(np.log(evals) ** 2)

        result = minimize(
            negative_squared_distance,
            W0.ravel(),
            method=self.solver,
            options={"maxiter": self.max_iter},
        )

        # Never return filters worse than the ones the optimiser started from
        if negative_squared_distance(W0.ravel()) < result.fun:
            return W0
        return result.x.reshape(shape)

    def _transform(self, X, y=None):
        """Transform the data using the fitted Riemannian Spatial Filter.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series.
        y : None
            Ignored, present for interface compatibility.

        Returns
        -------
        np.ndarray of shape (n_cases, n_components, n_timepoints)
            Transformed time series.
        """
        return self.filters_[: self.n_components] @ X

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
        return [
            {"n_components": 1},
            {"n_components": 2, "covariance_estimator": "lwf"},
            {
                "n_components": 1,
                "solver": "trust-constr",
                "init": "csp",
                "max_iter": 5,
                "random_state": 0,
            },
        ]
