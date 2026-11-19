"""xDAWN covariance estimation, wrapping pyRiemann's XdawnCovariances."""

from __future__ import annotations

__all__ = ["XDAWNCovariances"]

from numbers import Integral

from aeon.transformations.collection import BaseCollectionTransformer

from aeon_neuro.transformations.collection._erpcovariance import _check_classes


class XDAWNCovariances(BaseCollectionTransformer):
    """Event-related potential (ERP) covariance transformer with xDAWN filtering.

    As ``ERPCovariances``, but the signals are first reduced with xDAWN spatial
    filters [1]_, which keep the channel combinations where each class's ERP is
    strongest relative to the noise. This gives smaller, supervised covariance
    matrices [2]_. During `fit`, the xDAWN filters and an ERP for each class are
    estimated. During `transform`, the filtered ERPs are concatenated with the
    filtered channels of each case and the covariance of the combined series is
    computed. Uses `pyriemann.estimation.XdawnCovariances`.

    ``RiemannianMDMClassifier`` and ``RiemannianKNNClassifier`` run this step
    themselves with ``covariance_type="xdawn"``.

    Parameters
    ----------
    nfilter : int, default=4
        Number of xDAWN filters per class.
    applyfilters : bool, default=True
        If True, the filters are applied to both the ERPs and the cases. If
        False, they are applied to the ERPs only, so the cases keep all their
        channels; this can generalise better across subjects and sessions, at the
        cost of larger matrices.
    classes : list or None, default=None
        Classes to estimate ERPs for. If None, all classes in `y` are used.
    estimator : str, default="scm"
        Covariance matrix estimator of the output, see
        `pyriemann.geometry.covariance.covariances` for the available options.
    xdawn_estimator : str, default="scm"
        Covariance matrix estimator used to fit the xDAWN filters. A regularised
        estimator such as "lwf" or "oas" is more stable with few cases.
    estimator_params : dict or None, default=None
        Additional parameters passed to the output covariance estimator.

    References
    ----------
    .. [1] Rivet, B., Souloumiac, A., Attina, V., and Gibert, G. (2009). xDAWN
       Algorithm to Enhance Evoked Potentials: Application to Brain-Computer
       Interface. IEEE Transactions on Biomedical Engineering, 56(8), 2035-2043.
    .. [2] Barachant, A. and Congedo, M. (2014). A Plug&Play P300 BCI Using
       Information Geometry. arXiv:1409.0107.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection import XDAWNCovariances
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=8, n_timepoints=100)
    >>> XDAWNCovariances(nfilter=2).fit_transform(X, y).shape
    (20, 8, 8)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": False,
        "requires_y": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        nfilter=4,
        applyfilters=True,
        classes=None,
        estimator="scm",
        xdawn_estimator="scm",
        estimator_params=None,
    ):
        self.nfilter = nfilter
        self.applyfilters = applyfilters
        self.classes = classes
        self.estimator = estimator
        self.xdawn_estimator = xdawn_estimator
        self.estimator_params = estimator_params
        super().__init__()

    def _fit(self, X, y=None):
        from pyriemann.estimation import XdawnCovariances as PyRiemannXdawnCovariances

        _check_classes(self.classes, y)
        if (
            not isinstance(self.nfilter, Integral)
            or isinstance(self.nfilter, bool)
            or self.nfilter < 1
        ):
            raise ValueError("nfilter must be an integer of at least 1.")

        self.xdawn_covariances_ = PyRiemannXdawnCovariances(
            nfilter=self.nfilter,
            applyfilters=self.applyfilters,
            classes=self.classes,
            estimator=self.estimator,
            xdawn_estimator=self.xdawn_estimator,
            **(self.estimator_params or {}),
        )
        self.xdawn_covariances_.fit(X, y)
        return self

    def _transform(self, X, y=None):
        """Estimate xDAWN covariance matrices.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        np.ndarray of shape (n_cases, n_components, n_components)
            xDAWN covariance matrices. With ``n_filters = n_classes * nfilter``
            (at most ``n_classes * n_channels``), ``n_components`` is
            ``2 * n_filters`` if ``applyfilters`` is True, otherwise
            ``n_filters + n_channels``.
        """
        return self.xdawn_covariances_.transform(X)

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
            {"nfilter": 1},
            {"nfilter": 2, "estimator": "lwf", "applyfilters": False},
        ]
