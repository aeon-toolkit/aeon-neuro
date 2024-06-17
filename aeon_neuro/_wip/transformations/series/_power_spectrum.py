"""Power spectrum matrix transform, may call it something else."""

from aeon.transformations.series.base import BaseSeriesTransformer



class PowerSpectrumMatrix(BaseSeriesTransformer):
    """Implement here."""
    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }
    def __init__(
        self,
        n_lags=None,
    ):
        self.n_lags = n_lags
        super().__init__(axis=1)
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        return X
