"""Crop a collection to a time window."""

import math
from numbers import Real

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer

__all__ = ["TimeCrop"]


class TimeCrop(BaseCollectionTransformer):
    """Crop each series to a time window relative to an event.

    The start of the requested window is inclusive and the end is exclusive.
    Both bounds must lie on the sample grid.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    tmin : float or None, default=None
        Start of the crop window in seconds. If None, use the first sample.
    tmax : float or None, default=None
        End of the crop window in seconds, exclusive. If None, use the end of
        each series.
    start_time : float, default=0.0
        Time of the first sample in seconds relative to the event.

    Raises
    ------
    ValueError
        If ``sfreq`` is not finite and positive, a time is not finite or does
        not lie on the sample grid, ``tmin`` is not less than ``tmax``, or the
        requested window falls outside a series.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon_neuro.transformations.collection import TimeCrop
    >>> X = np.zeros((5, 2, 2048))
    >>> TimeCrop(sfreq=512, tmin=0.5, tmax=2.5, start_time=-1.5).fit_transform(
    ...     X
    ... ).shape
    (5, 2, 1024)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(self, sfreq, tmin=None, tmax=None, start_time=0.0):
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.start_time = start_time
        super().__init__()

    @staticmethod
    def _validate_finite_time(value, name):
        if (
            not isinstance(value, Real)
            or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite.")

    def _time_to_index(self, value, name):
        sample = (value - self.start_time) * self.sfreq
        nearest = round(sample)
        if not math.isclose(sample, nearest, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{name} must lie on the sample grid.")
        return int(nearest)

    def _transform(self, X, y=None):
        """Crop each series in X to the configured time window."""
        if (
            not isinstance(self.sfreq, Real)
            or isinstance(self.sfreq, (bool, np.bool_))
            or not math.isfinite(self.sfreq)
            or self.sfreq <= 0
        ):
            raise ValueError("sfreq must be finite and positive.")

        self._validate_finite_time(self.start_time, "start_time")
        for value, name in ((self.tmin, "tmin"), (self.tmax, "tmax")):
            if value is not None:
                self._validate_finite_time(value, name)

        if self.tmin is not None and self.tmax is not None:
            if self.tmin >= self.tmax:
                raise ValueError("tmin must be less than tmax.")

        start = 0 if self.tmin is None else self._time_to_index(self.tmin, "tmin")
        stop = None if self.tmax is None else self._time_to_index(self.tmax, "tmax")

        cropped = []
        for x in X:
            case_stop = x.shape[-1] if stop is None else stop
            if start < 0 or case_stop > x.shape[-1] or start >= case_stop:
                raise ValueError("The requested time window is outside the series.")
            cropped.append(x[:, start:case_stop].copy())

        if isinstance(X, np.ndarray):
            return np.asarray(cropped)
        return cropped

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for estimator checks."""
        return {"sfreq": 10}
