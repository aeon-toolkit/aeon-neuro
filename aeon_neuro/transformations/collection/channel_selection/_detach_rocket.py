"""Detach-ROCKET channel selector."""

from __future__ import annotations

from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

__all__ = ["DetachRocket"]


class DetachRocket(BaseChannelSelector):
    """Channel selector based on the Detach-ROCKET approach.

    Placeholder implementation for a channel selector that will use
    ROCKET-based transformed features and linear model coefficients to
    estimate channel importance and select a subset of channels.

    Parameters
    ----------
    proportion : float, default=0.25
        Proportion of channels to retain.
    n_kernels : int, default=10000
        Number of ROCKET kernels to use.
    estimator : str, default="ridge"
        Linear estimator used to score transformed features.
    alpha : float, default=1.0
        Regularisation strength for the linear estimator when applicable.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Notes
    -----
    This class is currently a placeholder and is not implemented.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "requires_y": True,
    }

    def __init__(
        self,
        proportion: float = 0.25,
        n_kernels: int = 10000,
        estimator: str = "ridge",
        alpha: float = 1.0,
        random_state: int | None = None,
    ):
        self.proportion = proportion
        self.n_kernels = n_kernels
        self.estimator = estimator
        self.alpha = alpha
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the channel selector.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Input time series collection.
        y : np.ndarray of shape (n_cases,), default=None
            Class labels.

        Returns
        -------
        self : DetachRocketChannelSelector
            Reference to self.

        Raises
        ------
        NotImplementedError
            Always, since this is a placeholder.
        """
        raise NotImplementedError("DetachRocketChannelSelector is not implemented yet.")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {},
            {
                "proportion": 0.5,
                "n_kernels": 1000,
                "alpha": 0.5,
                "random_state": 0,
            },
        ]
