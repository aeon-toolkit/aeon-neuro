"""Log-variance feature extraction for collections of time series."""

from __future__ import annotations

__all__ = ["LogVariance"]

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer


class LogVariance(BaseCollectionTransformer):
    """Log-variance of each channel over time.

    Computes the variance of each channel over the time dimension, optionally
    followed by the natural logarithm, giving one feature per channel. Applied
    to CSP-filtered EEG this is the classic CSP band power feature [1]_.

    Parameters
    ----------
    log : bool, default=True
        If True, return ``log(variance)``. If False, return the raw variance.

    Notes
    -----
    Zero variance (for example a flat channel) would give ``log(0) = -inf``.
    To keep features finite, the variance is floored at the smallest positive
    normal float64 (``np.finfo(np.float64).tiny``) before the logarithm is
    applied. Raw variance (``log=False``) is returned unchanged.

    References
    ----------
    .. [1] Ramoser, H., Muller-Gerking, J., and Pfurtscheller, G. (2000).
       Optimal spatial filtering of single trial EEG during imagined hand
       movement. IEEE Transactions on Rehabilitation Engineering, 8(4),
       441-446. https://doi.org/10.1109/86.895946

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection import LogVariance
    >>> X, _ = make_example_3d_numpy(
    ...     n_cases=5, n_channels=3, n_timepoints=20, random_state=0
    ... )
    >>> LogVariance().fit_transform(X).shape
    (5, 3)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "fit_is_empty": True,
    }

    def __init__(self, log: bool = True):
        self.log = log
        super().__init__()

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Return the (log) variance of each channel, shape (n_cases, n_channels)."""
        var = np.var(X, axis=2, dtype=np.float64)
        if self.log:
            return np.log(np.maximum(var, np.finfo(np.float64).tiny))
        return var

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return [{"log": True}, {"log": False}]
