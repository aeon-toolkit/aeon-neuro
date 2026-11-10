"""Differential entropy features of band-pass filtered EEG."""

from __future__ import annotations

__all__ = ["DifferentialEntropy", "DE_BANDS"]

from numbers import Real

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from scipy.signal import butter, sosfiltfilt

#: delta, theta, alpha, beta and gamma bands in Hz, as in [1]_ and [2]_.
DE_BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 14.0), (14.0, 31.0), (31.0, 50.0))


class DifferentialEntropy(BaseCollectionTransformer):
    """Differential entropy of each channel in each frequency band.

    Each channel is band-pass filtered in each band, and the differential
    entropy of the filtered signal is computed under a Gaussian assumption,

    ``DE = 0.5 * log(2 * pi * e * variance)``,

    which is the log band power up to a constant. This gives
    ``n_channels * n_bands`` features per case, ordered band by band (all
    channels for the first band, then all channels for the second, and so on).
    It is the standard feature set for EEG emotion and cognitive state
    recognition [1]_, [2]_.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input in Hz.
    bands : sequence of (low, high) or None, default=None
        Frequency bands in Hz. If None, ``DE_BANDS`` (delta 1-4, theta 4-8,
        alpha 8-14, beta 14-31 and gamma 31-50 Hz) is used. Bands whose low
        edge is at or above 95% of the Nyquist frequency are dropped, and
        high edges above it are capped there.
    order : int, default=4
        Order of the Butterworth band-pass filter, applied forwards and
        backwards (zero phase).

    Attributes
    ----------
    bands_ : list of (low, high)
        The bands actually used, after dropping and capping at Nyquist.

    Notes
    -----
    Variance is floored at ``np.finfo(np.float64).tiny`` before the logarithm,
    so a flat channel gives a large negative but finite feature.

    References
    ----------
    .. [1] Duan, R.-N., Zhu, J.-Y., and Lu, B.-L. (2013). Differential entropy
       feature for EEG-based emotion classification. 6th International
       IEEE/EMBS Conference on Neural Engineering, 81-84.
       https://doi.org/10.1109/NER.2013.6695876
    .. [2] Zheng, W.-L., and Lu, B.-L. (2015). Investigating critical frequency
       bands and channels for EEG-based emotion recognition with deep neural
       networks. IEEE Transactions on Autonomous Mental Development, 7(3),
       162-175. https://doi.org/10.1109/TAMD.2015.2431497

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.transformations.collection import DifferentialEntropy
    >>> X, _ = make_example_3d_numpy(
    ...     n_cases=5, n_channels=3, n_timepoints=200, random_state=0
    ... )
    >>> DifferentialEntropy(sfreq=128).fit_transform(X).shape
    (5, 15)
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "fit_is_empty": False,
    }

    def __init__(self, sfreq, bands=None, order: int = 4):
        self.sfreq = sfreq
        self.bands = bands
        self.order = order
        super().__init__()

    def _fit(self, X, y=None):
        if (
            not isinstance(self.sfreq, Real)
            or isinstance(self.sfreq, bool)
            or not np.isfinite(self.sfreq)
            or self.sfreq <= 0
        ):
            raise ValueError("sfreq must be a finite positive number.")
        bands = DE_BANDS if self.bands is None else self.bands
        top = 0.95 * float(self.sfreq) / 2
        self.bands_ = []
        for lo, hi in bands:
            if not 0 < lo < hi:
                raise ValueError(f"Invalid band ({lo}, {hi}): need 0 < low < high.")
            if lo < top:
                self.bands_.append((float(lo), float(min(hi, top))))
        if not self.bands_:
            raise ValueError(
                f"No frequency band lies below the Nyquist frequency for sfreq "
                f"{self.sfreq}."
            )
        return self

    def _transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_timepoints = X.shape[-1]
        feats = []
        for lo, hi in self.bands_:
            sos = butter(
                self.order, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos"
            )
            # scipy's default pad length can exceed short series
            padlen = min(3 * (2 * len(sos) + 1), n_timepoints - 1)
            Xf = sosfiltfilt(sos, X, axis=-1, padlen=padlen)
            var = np.maximum(Xf.var(axis=-1), np.finfo(np.float64).tiny)
            feats.append(0.5 * np.log(2 * np.pi * np.e * var))
        return np.concatenate(feats, axis=1)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for estimator checks."""
        return [{"sfreq": 100.0}, {"sfreq": 100.0, "bands": [(2.0, 10.0)]}]
