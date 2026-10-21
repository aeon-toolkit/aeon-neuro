"""xDAWN channel creation using the MNE implementation."""

from __future__ import annotations

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer


class Xdawn(BaseCollectionTransformer):
    """xDAWN spatial filter wrapper for aeon collections.

    This wraps ``mne.decoding.XdawnTransformer`` for use as an aeon collection
    transformer.

    xDAWN learns spatial filters that maximise the signal to signal-plus-noise
    ratio (SSNR) of the event-related potential (ERP). For each class, the ERP is
    the average of that class's trials, and the filters maximise the power of the
    filtered ERP relative to the power of the filtered single trials. This is
    solved as a generalised eigenvalue problem, and the ``n_components`` filters
    with the largest eigenvalues are kept for each class.

    Each output channel is a weighted combination of all input channels, so the
    original electrodes are not preserved.

    Parameters
    ----------
    n_components : int, default=2
        Number of spatial filters kept per class. The output has
        ``n_classes * min(n_channels, n_components)`` channels.
    reg : float | str | None, default=None
        Regularisation for covariance estimation, passed to
        ``mne.decoding.XdawnTransformer``. If float, shrinkage is used
        (0 <= reg <= 1). If str, it is passed as the method to
        ``mne.compute_covariance``. If None, the empirical covariance is used.
    signal_cov : np.ndarray | None, default=None
        Covariance matrix of shape (n_channels, n_channels) that the class ERPs
        are compared against. If None, it is estimated from all trials and
        timepoints.
    cov_method_params : dict | None, default=None
        Extra keyword arguments passed to ``mne.compute_covariance``.

    Attributes
    ----------
    xdawn_ : mne.decoding.XdawnTransformer
        The fitted MNE xDAWN transformer. Its ``filters_`` attribute holds the
        learned spatial filters.
    n_channels_in_ : int
        Number of channels seen in ``fit``.
    n_timepoints_in_ : int
        Number of timepoints seen in ``fit``.

    Notes
    -----
    - xDAWN is supervised, so ``fit`` requires ``y``.
    - xDAWN relies on responses that are time-locked to a stimulus, so it is
      intended for ERP data such as P300 or oddball paradigms. It is not expected
      to perform well on resting-state or ongoing EEG.
    - This is intended for equal-length collections with shape
      (n_cases, n_channels, n_timepoints). The output has the same number of
      timepoints, with shape
      (n_cases, n_classes * min(n_channels, n_components), n_timepoints).
    - The wrapper depends on ``mne``.

    References
    ----------
    .. [1] Rivet, B., Souloumiac, A., Attina, V., and Gibert, G. (2009).
       xDAWN algorithm to enhance evoked potentials: application to
       brain-computer interface. IEEE Transactions on Biomedical Engineering,
       56(8), 2035-2043. https://doi.org/10.1109/TBME.2009.2012869
    .. [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., and Mattout, J.
       (2011). Theoretical analysis of xDAWN algorithm: application to an
       efficient sensor selection in a P300 BCI. 19th European Signal
       Processing Conference (EUSIPCO), 1382-1386.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection.channel_creation import Xdawn
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=8, random_state=0)
    >>> xdawn = Xdawn(n_components=2)
    >>> Xt = xdawn.fit_transform(X, y)  # doctest: +SKIP
    >>> Xt.shape  # 2 classes * 2 components  # doctest: +SKIP
    (20, 4, 12)
    """

    _tags = {
        "capability:univariate": False,
        "capability:multivariate": True,
        "requires_y": True,
    }

    def __init__(
        self,
        n_components: int = 2,
        reg: float | str | None = None,
        signal_cov: np.ndarray | None = None,
        cov_method_params: dict | None = None,
    ):
        self.n_components = n_components
        self.reg = reg
        self.signal_cov = signal_cov
        self.cov_method_params = cov_method_params

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Fit xDAWN spatial filters on a 3D collection.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training collection.
        y : np.ndarray of shape (n_cases,)
            Class labels. Required, since xDAWN averages the trials of each
            class to estimate its ERP.

        Returns
        -------
        self : Xdawn
            Reference to self.
        """
        if y is None:
            raise ValueError("Xdawn requires y in fit.")

        try:
            from mne.decoding import XdawnTransformer
        except ImportError as exc:
            raise ImportError(
                "Xdawn requires the optional dependency 'mne'. "
                "Install it with: pip install mne"
            ) from exc

        self.xdawn_ = XdawnTransformer(
            n_components=self.n_components,
            reg=self.reg,
            signal_cov=self.signal_cov,
            cov_method_params=self.cov_method_params,
        )

        # MNE expects shape (n_epochs, n_channels, n_times), which matches aeon's
        # (n_cases, n_channels, n_timepoints).
        self.xdawn_.fit(X, y)

        self.n_channels_in_ = X.shape[1]
        self.n_timepoints_in_ = X.shape[2]

        return self

    def _transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Project a 3D collection onto the fitted xDAWN filters.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The collection to transform. Must have the same number of channels
            as the data seen in ``fit``.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        Xt : np.ndarray of shape (n_cases, n_out_channels, n_timepoints)
            The filtered collection, where
            ``n_out_channels = n_classes * min(n_channels, n_components)``. Each
            output channel is one xDAWN component.
        """
        # MNE already returns aeon's 3D collection format, so no reshape is
        # needed, unlike CommonSpatialPatterns with "average_power".
        Xt = self.xdawn_.transform(X)
        return Xt.astype(np.float64, copy=False)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {"n_components": 1},
            {"n_components": 2, "reg": 0.1},
        ]
