"""Bag of Visual Words transformer for collections of time series."""

from __future__ import annotations

__all__ = ["BoVWTransformer"]


import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans


class BoVWTransformer(BaseCollectionTransformer):
    """Represent each case as a histogram of codewords from a learned dictionary.

    Each case is split into local windows with a sliding window. Each window is
    z-normalised and, optionally, reduced to the approximation coefficients of a
    single-level discrete wavelet transform (DWT). A codebook is learned by
    k-means clustering of the windows seen in ``fit``, and each case is
    represented as a histogram of how often each codeword occurs [1]_.

    For multivariate input, the channels of a window are concatenated into a
    single descriptor, so each codeword is a multichannel pattern.

    For input data of shape ``(n_cases, n_channels, n_timepoints)``, the output
    has shape ``(n_cases, n_words)``.

    Parameters
    ----------
    n_words : int, default=1000
        Number of codewords in the dictionary (k-means clusters).
    window_size : int, default=128
        Length of each local window.
    stride : int, default=1
        Step between consecutive windows.
    descriptor : {"dwt", "raw"}, default="dwt"
        Descriptor computed from each z-normalised window. ``"dwt"`` keeps the
        approximation coefficients of a single-level DWT and requires
        PyWavelets. ``"raw"`` uses the z-normalised window values.
    wavelet : str, default="db3"
        Wavelet used when ``descriptor="dwt"``. Any wavelet name accepted by
        ``pywt.dwt``.
    normalise : bool, default=True
        If True, divide each histogram by its number of windows so it sums to
        one.
    random_state : int or None, default=None
        Seed for the k-means clustering.

    Attributes
    ----------
    kmeans_ : sklearn.cluster.KMeans
        Fitted k-means model whose cluster centres are the codewords.

    Raises
    ------
    ValueError
        If ``n_words``, ``window_size`` or ``stride`` is not a positive integer,
        if ``descriptor`` is not ``"dwt"`` or ``"raw"``, or if ``window_size``
        is longer than the series.
    ImportError
        If ``descriptor="dwt"`` and PyWavelets is not installed.

    References
    ----------
    .. [1] Wang, J., Liu, P., She, M. F. H., Nahavandi, S., and Kouzani, A.
       (2013). Bag-of-words representation for biomedical time series
       classification. Biomedical Signal Processing and Control, 8(6),
       634-644. https://doi.org/10.1016/j.bspc.2013.06.004

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection import BoVWTransformer
    >>> X, _ = make_example_3d_numpy(
    ...     n_cases=5, n_channels=1, n_timepoints=50, random_state=0
    ... )
    >>> bovw = BoVWTransformer(
    ...     n_words=4, window_size=8, descriptor="raw", random_state=0
    ... )
    >>> bovw.fit_transform(X).shape
    (5, 4)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "output_data_type": "Tabular",
    }

    def __init__(
        self,
        n_words=1000,
        window_size=128,
        stride=1,
        descriptor="dwt",
        wavelet="db3",
        normalise=True,
        random_state=None,
    ):
        self.n_words = n_words
        self.window_size = window_size
        self.stride = stride
        self.descriptor = descriptor
        self.wavelet = wavelet
        self.normalise = normalise
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Extract windows from X and learn the codebook."""
        self._validate_params()
        if self.window_size > X.shape[2]:
            raise ValueError(
                f"window_size ({self.window_size}) must not be longer than the "
                f"series length ({X.shape[2]})."
            )
        windows = self._extract_windows(X)
        self.kmeans_ = KMeans(
            n_clusters=self.n_words, n_init="auto", random_state=self.random_state
        ).fit(windows)
        return self

    def _transform(self, X, y=None):
        """Return the codeword histogram of each case, shape (n_cases, n_words)."""
        windows = self._extract_windows(X)
        codes = self._encode(windows)
        return self._histogram(codes, X.shape[0])

    def _extract_windows(self, X):
        """Slide a window over each case and return the stacked local descriptors."""
        windows = sliding_window_view(X, self.window_size, axis=2)[:, :, :: self.stride]

        mean = windows.mean(axis=-1, keepdims=True)
        std = windows.std(axis=-1, keepdims=True)
        z_norm = (windows - mean) / np.maximum(std, 1e-8)

        if self.descriptor == "dwt":
            try:
                import pywt
            except ImportError as exc:
                raise ImportError(
                    "descriptor='dwt' requires the optional dependency "
                    "'PyWavelets'. Install it with: pip install PyWavelets, "
                    "or use descriptor='raw'."
                ) from exc
            features, _ = pywt.dwt(z_norm, self.wavelet, axis=-1)
        else:
            features = z_norm

        n_cases, n_channels, n_windows, n_features = features.shape
        return features.transpose(0, 2, 1, 3).reshape(
            n_cases * n_windows, n_channels * n_features
        )

    def _encode(self, windows):
        """Assign each window to its nearest codeword."""
        return self.kmeans_.predict(windows)

    def _histogram(self, codes, n_cases):
        """Count codeword occurrences per case and optionally normalise."""
        codes = codes.reshape(n_cases, -1)
        offsets = np.arange(n_cases)[:, None] * self.n_words
        hist = (
            np.bincount((codes + offsets).ravel(), minlength=n_cases * self.n_words)
            .reshape(n_cases, self.n_words)
            .astype(np.float64)
        )
        if self.normalise:
            hist /= hist.sum(axis=1, keepdims=True)
        return hist

    def _validate_params(self):
        """Check that the parameters are valid."""
        for name in ("n_words", "window_size", "stride"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.descriptor not in ("dwt", "raw"):
            raise ValueError(
                f"descriptor must be 'dwt' or 'raw', got {self.descriptor!r}."
            )

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for estimator checks."""
        return [
            {"n_words": 3, "window_size": 4, "descriptor": "raw", "random_state": 0},
            {
                "n_words": 2,
                "window_size": 3,
                "stride": 2,
                "descriptor": "raw",
                "normalise": False,
                "random_state": 1,
            },
        ]
