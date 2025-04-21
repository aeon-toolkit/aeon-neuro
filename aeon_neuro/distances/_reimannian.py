import numpy as np
from scipy.linalg import logm, sqrtm


def affine_invariant_distance(P, Q):
    """
    Compute the affine-invariant Riemannian distance between two SPD matrices.

    This distance is defined as the Frobenius norm of the matrix logarithm
    of the congruence transformation of Q by the inverse square root of P:

        d(P, Q) = || logm(P^{-1/2} @ Q @ P^{-1/2}) ||_F

    It is invariant under affine transformations and commonly used for comparing
    SPD matrices such as EEG covariance matrices
    analysis.

    Parameters
    ----------
    P : ndarray of shape (n_channels, n_channels)
        A symmetric positive definite (SPD) covariance matrix.

    Q : ndarray of shape (n_channels, n_channels)
        A symmetric positive definite (SPD) covariance matrix to compare with `P`.

    Returns
    -------
    distance : float
        The affine-invariant Riemannian distance between `P` and `Q`.

    Notes
    -----
    Both `P` and `Q` must be symmetric positive definite. If matrices are close to
    singular, regularisation (e.g., adding epsilon * I) may be needed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import sqrtm
    >>> P = np.cov(np.random.randn(32, 100))
    >>> Q = np.cov(np.random.randn(32, 100))
    >>> affine_invariant_distance(P, Q)
    1.54  # example output (will vary)
    """
    P_inv_sqrt = np.linalg.inv(sqrtm(P))
    middle = P_inv_sqrt @ Q @ P_inv_sqrt
    log_middle = logm(middle)
    distance = np.linalg.norm(log_middle, "fro")
    return np.real(distance)


def log_euclidean_distance(P, Q):
    """
    Compute the Log-Euclidean distance between two covariance matrices.

    This distance is defined as the Frobenius norm of the difference of
    the matrix logarithms of `P` and `Q`:

        d(P, Q) = ||logm(P) - logm(Q)||_F

    The Log-Euclidean metric is a computationally efficient alternative
    to the affine-invariant Riemannian distance and is suitable for
    comparing symmetric positive definite (SPD) matrices such as covariance matrices.

    Parameters
    ----------
    P : ndarray of shape (n_channels, n_channels)
        A symmetric positive definite (SPD) covariance matrix.

    Q : ndarray of shape (n_channels, n_channels)
        A symmetric positive definite (SPD) covariance matrix to compare with `P`.

    Returns
    -------
    distance : float
        The Log-Euclidean distance between `P` and `Q`.

    Notes
    -----
    The input matrices must be symmetric and positive definite.
    The matrix logarithm is computed using `scipy.linalg.logm`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import logm
    >>> P = np.cov(np.random.randn(32, 100))
    >>> Q = np.cov(np.random.randn(32, 100))
    >>> log_euclidean_distance(P, Q)
    1.82  # example output (actual value will vary)
    """
    log_P = logm(P)
    log_Q = logm(Q)
    distance = np.linalg.norm(log_P - log_Q, "fro")
    return np.real(distance)
