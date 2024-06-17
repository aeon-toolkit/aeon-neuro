"""Implement as a distance function here."""
def reimannian_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    r"""Compute the Reimannian distance between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    """
    return x+y
