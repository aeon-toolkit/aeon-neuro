"""CSP channel selection using PyReimann implementation."""

from __future__ import annotations

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer


class CommonSpacialPatterns(BaseCollectionTransformer):
    """Common Spatial Patterns wrapper for aeon collections.

    This wraps ``mne.decoding.CSP`` for use as an aeon collection transformer.

    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components to keep.
    reg : float | str | None, default=None
        Regularisation passed to ``mne.decoding.CSP``.
    log : bool | None, default=True
        Whether to apply log transform to average power features.
    cov_est : {"concat", "epoch"}, default="concat"
        Covariance estimation mode passed to MNE CSP.
    transform_into : {"average_power", "csp_space"}, default="average_power"
        Output mode of MNE CSP.

        - "average_power" gives shape (n_cases, n_components)
        - "csp_space" gives shape (n_cases, n_components, n_timepoints)

        This wrapper always returns aeon collection format:
        - average_power -> (n_cases, n_components, 1)
        - csp_space -> (n_cases, n_components, n_timepoints)
    norm_trace : bool, default=False
        Passed to ``mne.decoding.CSP``.
    cov_method_params : dict | None, default=None
        Passed to ``mne.decoding.CSP``.
    component_order : {"mutual_info", "alternate"}, default="mutual_info"
        Passed to ``mne.decoding.CSP``.
    random_state : int | None, default=None
        Included for API consistency. Not currently used by MNE CSP directly.

    Notes
    -----
    - CSP is supervised, so ``fit`` requires ``y``.
    - This is intended for equal-length collections with shape
      (n_cases, n_channels, n_timepoints).
    - The wrapper depends on ``mne``.
    """

    _tags = {
        "capability:univariate": False,
        "capability:multivariate": True,
        "requires_y": True,
    }

    def __init__(
        self,
        n_components: int = 4,
        reg: float | str | None = None,
        log: bool | None = True,
        cov_est: str = "concat",
        transform_into: str = "average_power",
        norm_trace: bool = False,
        cov_method_params: dict | None = None,
        component_order: str = "mutual_info",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.transform_into = transform_into
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Fit CSP on a 3D collection."""
        if y is None:
            raise ValueError("CSPTransformer requires y in fit.")

        self._check_X(X)
        y = np.asarray(y)

        try:
            from mne.decoding import CSP
        except ImportError as exc:
            raise ImportError(
                "CSPTransformer requires the optional dependency 'mne'. "
                "Install it with: pip install mne"
            ) from exc

        self.csp_ = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            cov_est=self.cov_est,
            transform_into=self.transform_into,
            norm_trace=self.norm_trace,
            cov_method_params=self.cov_method_params,
            component_order=self.component_order,
        )

        # MNE CSP expects shape (n_epochs, n_channels, n_times)
        self.csp_.fit(X, y)

        self.n_channels_in_ = X.shape[1]
        self.n_timepoints_in_ = X.shape[2]

        return self

    def _transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Transform a 3D collection using the fitted CSP."""
        self._check_X(X)

        Xt = self.csp_.transform(X)

        # MNE returns:
        # - average_power -> (n_cases, n_components)
        # - csp_space -> (n_cases, n_components, n_timepoints)
        if Xt.ndim == 2:
            Xt = Xt[:, :, np.newaxis]
        elif Xt.ndim != 3:
            raise RuntimeError(
                f"Unexpected CSP output shape {Xt.shape}. " "Expected 2D or 3D output."
            )

        return Xt.astype(np.float64, copy=False)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        return [
            {"n_components": 2},
            {"n_components": 4, "transform_into": "csp_space"},
        ]
