"""Filter Bank Common Spatial Patterns (FBCSP) feature extraction."""

from __future__ import annotations

__all__ = ["FBCommonSpatialPatterns"]

from functools import partial
from numbers import Real

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer

from aeon_neuro.transformations.collection._log_variance import LogVariance
from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.series import BandPassSeriesTransformer


class FBCommonSpatialPatterns(BaseCollectionTransformer):
    """Filter Bank Common Spatial Patterns (FBCSP) feature extractor.

    Filter Bank Common Spatial Patterns is a variation of the CSP transformer
    in which X is first band-pass filtered into several frequency bands (the
    filter bank). A separate ``CommonSpatialPatterns`` is fitted in each band,
    so each band gets its own spatial filters, chosen to best separate the
    classes in that band. Each trial is then projected through every band's
    filters, and the log-variance (log power) of each resulting component is
    used as a feature.

    The features from all bands are concatenated, giving
    ``n_bands * n_components`` features per trial. Optionally, only the ``k``
    most informative features are kept, scored against the class labels with
    mutual information. Because the bands compete in this selection, the
    transformer can adapt to the frequency range that discriminates best for a
    given subject. Time is averaged out, so the output has shape
    (n_cases, n_features).

    Parameters
    ----------
    sfreq : float, default=250.0
        Sampling frequency of X in Hz, used to design the band-pass filters.
        Every band's upper edge must be below ``sfreq / 2``.
    bands : list of tuple of float or None, default=None
        Frequency bands as (l_freq, h_freq) pairs in Hz, with
        ``0 < l_freq < h_freq``. If None, uses the bands of Ang et al. (2012):
        nine 4 Hz wide bands from 4 to 40 Hz.
    n_components : int, default=4
        Number of CSP components kept in each band, passed to
        ``CommonSpatialPatterns``.
    k : int or "all", default="all"
        Number of features kept by ``SelectKBest`` with
        ``mutual_info_classif``. Must be at most ``n_bands * n_components``.
        "all" keeps every feature.
    reg : float, str or None, default=None
        Covariance regularisation, passed to ``CommonSpatialPatterns``.
    method : {"iir", "fir"}, default="iir"
        Band-pass filter type, passed to ``mne.filter.filter_data``.
    n_jobs : int, default=1
        Number of jobs used for filtering.
    random_state : int or None, default=None
        Seed for the small noise ``mutual_info_classif`` adds to the features to
        break ties. Set it for reproducible feature selection.

    Attributes
    ----------
    bands_ : list of tuple of float
        The (l_freq, h_freq) bands used, with None resolved to the defaults.
    csps_ : list of CommonSpatialPatterns
        One fitted CSP per band, in the same order as ``bands_``.
    selector_ : sklearn.feature_selection.SelectKBest
        The fitted feature selector. ``selector_.scores_`` holds each
        feature's mutual information and ``selector_.get_support()`` the
        selected features.

    See Also
    --------
    CommonSpatialPatterns : CSP spatial filter for a single band.
    LogVariance : Log-variance features.

    References
    ----------
    .. [1] Ang, K. K., Chin, Z. Y., Wang, C., Guan, C., and Zhang, H. (2012).
       Filter bank common spatial pattern algorithm on BCI competition IV
       datasets 2a and 2b. Frontiers in Neuroscience, 6, 39.
       https://doi.org/10.3389/fnins.2012.00039

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection import FBCommonSpatialPatterns
    >>> X, y = make_example_3d_numpy(n_cases=20, n_channels=8, n_timepoints=250,
    ...                              random_state=0)
    >>> fbcsp = FBCommonSpatialPatterns(sfreq=250, k=10, random_state=0)
    >>> fbcsp.fit_transform(X, y).shape
    (20, 10)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,  # CSP needs at least two channels
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "requires_y": True,  # CSP and SelectKBest are supervised
        "fit_is_empty": False,
        "capability:multithreading": True,  # n_jobs is passed to the filters
        "python_dependencies": "mne",
    }

    def __init__(
        self,
        sfreq: float = 250.0,
        bands: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        k: int | str = "all",
        reg: float | str | None = None,
        method: str = "iir",
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.k = k
        self.reg = reg
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit one CSP per band and the feature selector.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials.
        y : np.ndarray of shape (n_cases,)
            Class labels.

        Returns
        -------
        self
            Fitted transformer.
        """
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        # sfreq, method and each band's limits against sfreq / 2 are checked by
        # BandPassSeriesTransformer in self._filter
        bands = _DEFAULT_BANDS if self.bands is None else self.bands
        if not isinstance(bands, (list, tuple)) or len(bands) == 0:
            raise ValueError(
                "bands must be None or a non-empty list of (l_freq, h_freq) pairs."
            )
        for band in bands:
            if (
                not isinstance(band, (list, tuple))
                or len(band) != 2
                or not all(
                    isinstance(f, Real) and not isinstance(f, bool) for f in band
                )
            ):
                raise ValueError(
                    f"Each band must be a pair of numbers (l_freq, h_freq), "
                    f"got {band!r}."
                )
            l_freq, h_freq = band
            if not 0 < l_freq < h_freq:
                raise ValueError(
                    f"Each band must satisfy 0 < l_freq < h_freq, got {band!r}."
                )
        self.bands_ = [tuple(band) for band in bands]

        if self.k != "all" and (
            isinstance(self.k, bool)
            or not isinstance(self.k, (int, np.integer))
            or self.k < 1
        ):
            raise ValueError(f"k must be 'all' or a positive int, got {self.k!r}.")

        self.csps_ = [
            CommonSpatialPatterns(n_components=self.n_components, reg=self.reg).fit(
                self._filter(X, l_freq, h_freq), y
            )
            for l_freq, h_freq in self.bands_
        ]

        features = self._band_features(X)  # (n_cases, n_bands * n_components)

        if self.k != "all" and self.k > features.shape[1]:
            raise ValueError(
                f"k={self.k} exceeds the number of features ({features.shape[1]}),"
                f" which is n_bands * n_components."
            )

        self.selector_ = SelectKBest(
            score_func=partial(mutual_info_classif, random_state=self.random_state),
            k=self.k,
        ).fit(features, y)
        return self

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Return the selected FBCSP features.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of EEG trials.
        y : None
            Ignored for interface compatibility.

        Returns
        -------
        np.ndarray of shape (n_cases, n_features)
            ``n_features`` is ``k``, or ``n_bands * n_components`` if
            ``k="all"``.
        """
        return self.selector_.transform(self._band_features(X))

    def _band_features(self, X: np.ndarray) -> np.ndarray:
        """Return log-variance CSP features of every band, concatenated.

        Returns
        -------
        np.ndarray of shape (n_cases, n_bands * n_components)
        """
        band_features = []
        for (l_freq, h_freq), csp in zip(self.bands_, self.csps_):
            X_band = self._filter(X, l_freq, h_freq)
            components = csp.transform(X_band)  # (n_cases, n_components, n_timepoints)
            band_features.append(LogVariance().fit_transform(components))
        return np.hstack(band_features)

    def _filter(self, X: np.ndarray, l_freq: float, h_freq: float) -> np.ndarray:
        """Band-pass filter every case of X.

        ``BandPassSeriesTransformer`` filters one series of shape
        (n_channels, n_timepoints), so it is applied to each case in turn. It
        also validates ``sfreq`` and the band limits.

        Returns
        -------
        np.ndarray of shape (n_cases, n_channels, n_timepoints)
        """
        bandpass = BandPassSeriesTransformer(
            sfreq=self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method=self.method,
            n_jobs=self.n_jobs,
        )
        return np.stack([bandpass.fit_transform(case) for case in X])

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
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
        # TODO: aeon's test data is short and its sfreq is arbitrary, so use few
        # wide bands and small n_components, e.g.
        # {"sfreq": 100, "bands": [(8, 20), (20, 40)], "n_components": 2}.
        # Check the filters work on aeon's series length (IIR is fine on short
        # data; FIR may warn or fail).
        return [
            {"sfreq": 100, "bands": [(8, 20), (20, 40)], "n_components": 2},
        ]


# Ang et al. (2012): nine 4 Hz wide bands covering 4 to 40 Hz
_DEFAULT_BANDS = [(low, low + 4) for low in range(4, 40, 4)]
