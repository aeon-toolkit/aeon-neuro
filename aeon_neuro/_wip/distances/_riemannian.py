"""Implement as a distance function here."""

import math

import numpy as np
from scipy.linalg import eig, logm


def riemannian_distance_a(
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
    x_cov = np.cov(x)
    y_cov = np.cov(y)
    x_trace = np.trace(x_cov)
    y_trace = np.trace(y_cov)
    x_root = logm(x_cov)
    c_temp = logm(np.matmul(x_root, np.matmul(y_cov, x_root)))
    c = 2 * np.trace(c_temp)
    return math.sqrt(x_trace + y_trace - c)


def riemannian_distance_b(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    r"""Compute the Reimannian distance between two time series.

    https://hal.science/hal-00820475/document
    https://hal.science/hal-00602700/document

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    """
    x_cov = np.cov(x)
    y_cov = np.cov(y)
    eigenvalues = eig(x_cov, y_cov)
    distance = 0
    for i in eigenvalues[0]:
        distance += math.log(np.real(i)) ** 2
    return math.sqrt(distance)


def riemannian_distance_c(
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
    x_cov = np.cov(x)
    y_cov = np.cov(y)
    x_inv_root = logm(np.linalg.inv(x_cov))
    distance = logm(np.matmul(x_inv_root, np.matmul(y_cov, x_inv_root)))
    return np.linalg.norm(distance)
