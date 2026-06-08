"""Riemannian channel selection using pyRiemann."""

from __future__ import annotations

import numpy as np
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

__all__ = ["Riemannian"]


class Riemannian(BaseChannelSelector):
    """Select channels using Riemannian electrode selection.

    Wrapper around ``pyriemann.channelselection.ElectrodeSelection`` for aeon
    collection data. The selector first estimates a covariance matrix for each
    case, then applies the Riemannian electrode selection procedure of
    Barachant and Bonnet [1] to identify a subset of informative channels.

    The selected channels are then used to subset the original collection of
    time series.

    Parameters
    ----------
    proportion : float, default=0.25
        Proportion of channels to retain. Must be in the interval ``(0, 1]``.
        The number of selected channels is computed as
        ``ceil(proportion * n_channels)``.
    metric : str, default="riemann"
        Metric passed to ``pyriemann.channelselection.ElectrodeSelection``.
    estimator : str, default="scm"
        Covariance estimator passed to ``pyriemann.estimation.Covariances``.
    n_jobs : int, default=1
        Number of jobs passed to ``pyriemann.channelselection.ElectrodeSelection``.

    Attributes
    ----------
    channels_selected_ : list of int
        Indices of the selected channels.
    n_channels_in_ : int
        Number of channels seen in ``fit``.
    selector_ : object
        Fitted ``pyriemann.channelselection.ElectrodeSelection`` instance.
    covariances_ : np.ndarray of shape (n_cases, n_channels, n_channels)
        Covariance matrices estimated from the training data.

    Raises
    ------
    ValueError
        If `proportion` is not in the interval ``(0, 1]``.
    ValueError
        If the computed number of retained channels is less than 1.

    Notes
    -----
    This implements a filter-style Riemannian channel selector. The selection
    is performed on covariance matrices estimated from the training collection,
    while the final transform subsets the original time series channels.

    References
    ----------
    .. [1] Barachant, A., and Bonnet, S. (2011). Channel selection procedure
           using Riemannian distance for BCI applications. 5th International
           IEEE/EMBS Conference on Neural Engineering, 348-351.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "requires_y": True,
    }

    def __init__(
        self,
        proportion: float = 0.25,
        metric: str = "riemann",
        estimator: str = "scm",
        n_jobs: int = 1,
    ):
        self.proportion = proportion
        self.metric = metric
        self.estimator = estimator
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the channel selector.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of equal-length time series.
        y : np.ndarray of shape (n_cases,)
            Class labels.

        Returns
        -------
        self : RiemannianChannelSelector
            Reference to self.
        """
        self._validate_params()

        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D numpy array of shape "
                "(n_cases, n_channels, n_timepoints)."
            )

        n_cases, n_channels, _ = X.shape
        self.n_channels_in_ = n_channels

        nelec = int(np.ceil(self.proportion * n_channels))
        if nelec < 1:
            raise ValueError(
                "The computed number of selected channels must be at least 1."
            )

        try:
            from pyriemann.channelselection import ElectrodeSelection
            from pyriemann.estimation import Covariances
        except ImportError as exc:
            raise ImportError(
                "RiemannianChannelSelector requires the optional dependency "
                "'pyriemann'. Install it with: pip install pyriemann"
            ) from exc

        cov = Covariances(estimator=self.estimator)
        covariances = cov.transform(X)

        selector = ElectrodeSelection(
            nelec=nelec,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        selector.fit(covariances, y)

        self.covariances_ = covariances
        self.selector_ = selector

        # pyRiemann exposes selected indices as `subelec_`
        self.channels_selected_ = list(selector.subelec_)

        return self

    def _validate_params(self):
        """Validate estimator parameters."""
        if not (0 < self.proportion <= 1):
            raise ValueError("proportion must be in the interval (0, 1].")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
            {"proportion": 0.25},
            {"proportion": 0.5, "metric": "riemann", "estimator": "oas"},
        ]
