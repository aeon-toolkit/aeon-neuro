"""Time-delay embedding of each channel, for augmented covariance matrices."""

from __future__ import annotations

__all__ = ["TimeDelayEmbedding"]

from numbers import Integral

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer


class TimeDelayEmbedding(BaseCollectionTransformer):
    """Time-delay embedding, the first step of augmented covariance matrices.

    Each case is stacked with ``order - 1`` copies of itself, each shifted a
    further ``lag`` samples later in time. The covariance of the output is the
    augmented covariance matrix (ACM) of [1]_, which builds on Takens' delay
    embedding theorem. A plain covariance only shows how the channels relate to
    each other. The ACM also shows how each channel relates to its own recent
    past and to the other channels' recent past: the autoregressive structure of
    the signal.

    Applying ``RiemannianMDMClassifier`` to the output gives ACM+MDM. A tangent
    space followed by an SVM gives ACM+TS+SVM, the best motor imagery pipeline
    in the MOABB benchmark [2]_.

    The output channels are ordered by delay: first all channels without delay,
    then all channels delayed by ``lag`` samples, and so on.

    Parameters
    ----------
    order : int, default=2
        Number of delayed copies of each channel, including the original. With
        ``order=1`` the data is unchanged.
    lag : int, default=1
        Delay in samples between consecutive copies.

    Notes
    -----
    The output has shape
    (n_cases, n_channels * order, n_timepoints - (order - 1) * lag).

    The best ``order`` and ``lag`` depend on the dataset and sampling rate (``lag``
    is in samples, so the same value spans less time at a higher rate). The
    authors select them by cross-validation, searching both from 1 to 10 [1]_.
    The defaults are a safe starting point; to tune them, search over them with
    ``GridSearchCV`` as in the example below. A larger ``order`` gives larger
    matrices, (n_channels * order) squared, which are harder to estimate. Use a
    regularised covariance estimator such as
    ``RiemannianMDMClassifier(covariance_estimator="lwf")`` [1]_; it is needed
    when the output has fewer timepoints than channels, or the plain estimate
    is not positive definite.

    MOABB's ``AugmentedDataset`` drops a further ``lag`` samples from the end of
    each series; otherwise the output is the same.

    References
    ----------
    .. [1] Carrara, I. and Papadopoulo, T. (2024). Classification of BCI-EEG
       Based on the Augmented Covariance Matrix. IEEE Transactions on Biomedical
       Engineering. arXiv:2302.04508.
    .. [2] Chevallier, S., Carrara, I., Aristimunha, B., et al. (2024). The
       largest EEG-based BCI reproducibility study for open science: the MOABB
       benchmark. arXiv:2404.15319.

    Examples
    --------
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.distance_based import RiemannianMDMClassifier
    >>> from aeon_neuro.transformations.collection.channel_creation import (
    ...     TimeDelayEmbedding,
    ... )
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=4, n_timepoints=100)
    >>> TimeDelayEmbedding(order=3, lag=2).fit_transform(X).shape
    (20, 12, 96)
    >>> acm_mdm = make_pipeline(
    ...     TimeDelayEmbedding(order=3, lag=2), RiemannianMDMClassifier()
    ... )
    >>> acm_mdm.fit(X, y).predict(X).shape
    (20,)

    Choose ``order`` and ``lag`` by cross-validation on the training data:

    >>> from sklearn.model_selection import GridSearchCV
    >>> grid = {
    ...     "TimeDelayEmbedding__order": [1, 2, 3],
    ...     "TimeDelayEmbedding__lag": [1, 2],
    ... }
    >>> search = GridSearchCV(acm_mdm, grid, cv=3).fit(X, y)
    >>> sorted(search.best_params_)
    ['TimeDelayEmbedding__lag', 'TimeDelayEmbedding__order']
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, order=2, lag=1):
        self.order = order
        self.lag = lag
        super().__init__()

    def _transform(self, X, y=None):
        """Stack each case with delayed copies of itself.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_cases, n_channels * order, n_timepoints_out)
            The embedded series, with
            ``n_timepoints_out = n_timepoints - (order - 1) * lag``.
        """
        for name in ("order", "lag"):
            value = getattr(self, name)
            is_int = isinstance(value, Integral) and not isinstance(value, bool)
            if not is_int or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")

        n_out = X.shape[2] - (self.order - 1) * self.lag
        if n_out < 2:
            raise ValueError(
                f"order={self.order} and lag={self.lag} need series longer than "
                f"{(self.order - 1) * self.lag + 1} timepoints, got {X.shape[2]}."
            )
        return np.concatenate(
            [X[:, :, p * self.lag : p * self.lag + n_out] for p in range(self.order)],
            axis=1,
        )

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
        return [{}, {"order": 3, "lag": 2}]
