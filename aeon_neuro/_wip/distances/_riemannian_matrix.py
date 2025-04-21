"""Riemannian distances between SPD/HPD matrices."""

import numpy as np
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import sqrtm


def riemannian_distance_1(
    A: np.ndarray,
    B: np.ndarray,
) -> float:
    r"""Compute the Reimannian distance between two SPD/HPD matrices.

    SPD: Symmetric Positive Definite
    HPD: Hermitian Positive Definite

    The first type of Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d_{R_1}(\mathbf{A}, \mathbf{B}) =
        \sqrt{\text{tr} \mathbf{A} +
        \text{tr} \mathbf{B} -
        2 \text{tr} \left[(\mathbf{A} \mathbf{B})^{1/2}\right]}

    Parameters
    ----------
    A : np.ndarray
        First SPD/HPD matrices, 2D ndarray.
    y : np.ndarray
        Second SPD/HPD matrices, same dimensions as A.

    Returns
    -------
    d : float
        Riemannian distance between A and B.

    References
    ----------
    .. [1] `Riemannian Distances for Signal Classification by Power Spectral Density
        <https://ieeexplore.ieee.org/document/6509394>`_
        Li, Y., & Wong, K.M. IEEE Journal of Selected Topics in Signal Processing,
        2013, 7, pp. 655-669
    """
    same_data = B is A
    if same_data:
        return 0

    d = np.sqrt(np.trace(A) + np.trace(B) - 2 * np.trace(sqrtm(np.dot(A, B))))

    return d


def riemannian_distance_2(
    A: np.ndarray,
    B: np.ndarray,
) -> float:
    r"""Compute the Reimannian distance between two SPD/HPD matrices.

    SPD: Symmetric Positive Definite
    HPD: Hermitian Positive Definite

    The second type of Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d_{R_2}(\mathbf{A}, \mathbf{B}) =
            \sqrt{\text{tr} \mathbf{A} +
            \text{tr} \mathbf{B} -
            2 \text{tr} \left[\mathbf{A}^{1/2} \mathbf{B}^{1/2}\right]}
            
    Parameters
    ----------
    A : np.ndarray
        First SPD/HPD matrices, 2D ndarray.
    B : np.ndarray
        Second SPD/HPD matrices, same dimensions as A.

    Returns
    -------
    d : float
        Riemannian distance between A and B.

    References
    ----------
    .. [1] `Riemannian Distances for Signal Classification by Power Spectral Density
        <https://ieeexplore.ieee.org/document/6509394>`_
        Li, Y., & Wong, K.M. IEEE Journal of Selected Topics in Signal Processing,
        2013, 7, pp. 655-669
    """
    same_data = B is A
    if same_data:
        return 0

    d = np.sqrt(np.trace(A) + np.trace(B) - 2 * np.trace(np.dot(sqrtm(A), sqrtm(B))))

    return d


def riemannian_distance_3(
    A: np.ndarray,
    B: np.ndarray,
    squared: bool = False,
) -> float:
    r"""Affine-invariant Riemannian distance between SPD/HPD matrices.

    A direct call to method ``pyriemann.utils.distance.distance_riemann`` [1]_.

    The affine-invariant Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [2]_:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}

    where :math:`\lambda_i` are the joint eigenvalues of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First SPD/HPD matrices, 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second SPD/HPD matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

    Returns
    -------
    d : float or ndarray, shape (...,)
        Affine-invariant Riemannian distance between A and B.

    References
    ----------
    # noqa: E501
    .. [1] https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.utils.distance.distance_riemann.html
    .. [2] `A differential geometric approach to the geometric mean of
        symmetric positive-definite matrices
        <https://epubs.siam.org/doi/10.1137/S0895479803436937>`_
        M. Moakher. SIAM J Matrix Anal Appl, 2005, 26 (3), pp. 735-747
    """
    return distance_riemann(A, B, squared)


def weighted_riemannian_distance_1(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
) -> float:
    r"""Compute the weighted Reimannian distance between two SPD/HPD matrices.

    SPD: Symmetric Positive Definite
    HPD: Hermitian Positive Definite

    The first type of weighted Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d_{R_1w}(\mathbf{A}, \mathbf{B}, \mathbf{W_1}) =
        \sqrt{\text{tr}\left[ \mathbf{W_1} \mathbf{A}\right] +
        \text{tr}\left[ \mathbf{W_1} \mathbf{B}\right] -
        2 \text{tr}\left[ (\mathbf{B}^{1/2} \mathbf{W_1} \mathbf{A} \mathbf{W_1} \mathbf{B}^{1/2})^{1/2}\right]}

    where :math:`\mathbf{W}` is the weight matrix of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : np.ndarray
        First SPD/HPD matrices, 2D ndarray.
    B : np.ndarray
        Second SPD/HPD matrices, same dimensions as A.
    W : np.ndarray
        Weight matrix.

    Returns
    -------
    d : float
        Weighted Riemannian distance between A and B.

    References
    ----------
    .. [1] `Riemannian Distances for Signal Classification by Power Spectral Density
        <https://ieeexplore.ieee.org/document/6509394>`_
        Li, Y., & Wong, K.M. IEEE Journal of Selected Topics in Signal Processing,
        2013, 7, pp. 655-669
    """
    same_data = B is A
    if same_data:
        return 0

    B_sqrt = sqrtm(B)
    term1 = np.trace(np.dot(W, A))
    term2 = np.trace(np.dot(W, B))
    term3 = np.dot(np.dot(B_sqrt, W), np.dot(A, np.dot(W, B_sqrt)))
    term3_ = np.trace(sqrtm(term3))
    d_W = np.sqrt(term1 + term2 - 2 * term3_)

    return d_W


def weighted_riemannian_distance_2(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
) -> float:
    r"""Compute the weighted Reimannian distance between two SPD/HPD matrices.

    SPD: Symmetric Positive Definite
    HPD: Hermitian Positive Definite

    The second type of weighted Riemannian distance between two SPD/HPD matrices
    :math:`\mathbf{A}` and :math:`\mathbf{B}` is [1]_:

    .. math::
        d_{R_2w}(\mathbf{A}, \mathbf{B}, \mathbf{W_2}) =
        \sqrt{\text{tr}\left[ \mathbf{W_2} \mathbf{A}\right] +
        \text{tr}\left[ \mathbf{W_2} \mathbf{B}\right] -
        \text{tr}\left[ \mathbf{W_2} \mathbf{A}^{1/2} \mathbf{B}^{1/2}\right] -
        \text{tr}\left[ \mathbf{W_2} \mathbf{B}^{1/2} \mathbf{A}^{1/2}\right]}

    where :math:`\mathbf{W}` is the weight matrix of :math:`\mathbf{A}` and
    :math:`\mathbf{B}`.

    Parameters
    ----------
    A : np.ndarray
        First SPD/HPD matrices, 2D ndarray.
    B : np.ndarray
        Second SPD/HPD matrices, same dimensions as A.
    W : np.ndarray
        Weight matrix.

    Returns
    -------
    d : float
        Weighted Riemannian distance between A and B.

    References
    ----------
    .. [1] `Riemannian Distances for Signal Classification by Power Spectral Density
        <https://ieeexplore.ieee.org/document/6509394>`_
        Li, Y., & Wong, K.M. IEEE Journal of Selected Topics in Signal Processing,
        2013, 7, pp. 655-669
    """
    same_data = B is A
    if same_data:
        return 0

    B_sqrt = sqrtm(B)
    A_sqrt = sqrtm(A)
    term1 = np.trace(np.dot(W, A))
    term2 = np.trace(np.dot(W, B))
    term3 = np.trace(np.dot(np.dot(W, A_sqrt), B_sqrt))
    term4 = np.trace(np.dot(np.dot(W, B_sqrt), A_sqrt))
    d_W = np.sqrt(term1 + term2 - term3 - term4)

    return d_W


def pairwise_riemannian_distance_1(
    X: np.ndarray,
    Y: np.ndarray = None,
) -> np.ndarray:
    r"""Compute the pairwise Reimannian distance between two sets of SPD/HPD matrices.
    
    The first type of pairwise Riemannian distance
    
    Parameters
    ----------
    X : np.ndarray
        First set of SPD/HPD matrices, 3D ndarray.
    Y : np.ndarray, default None
    
    Returns
    -------
    distances : np.ndarray
        Pairwise Riemannian distances between X and Y or X itself.
    """
    if Y is None:
        Y = X
        
    n_samples_X, n_samples_Y = X.shape[0], Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            d = 0
            for k in range(X.shape[1]):
                d += weighted_riemannian_distance_1(X[i, k], Y[j, k])
            distances[i, j] = d
            
    return distances


def pairwise_riemannian_distance_2(
    X: np.ndarray,
    Y: np.ndarray = None,
) -> np.ndarray:
    r"""Compute the pairwise Reimannian distance between two sets of SPD/HPD matrices.
    
    The second type of pairwise Riemannian distance
    
    Parameters
    ----------
    X : np.ndarray
        First set of SPD/HPD matrices, 3D ndarray.
    Y : np.ndarray, default None
    
    Returns
    -------
    distances : np.ndarray
        Pairwise Riemannian distances between X and Y or X itself.
    """
    if Y is None:
        Y = X
        
    n_samples_X, n_samples_Y = X.shape[0], Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            d = 0
            for k in range(X.shape[1]):
                d += weighted_riemannian_distance_2(X[i, k], Y[j, k])
            distances[i, j] = d
            
    return distances


def pairwise_weighted_riemannian_distance_1(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray = None,
) -> np.ndarray:
    r"""Compute the pairwise weighted Reimannian distance between two sets of SPD/HPD matrices.
    
    The first type of pairwise weighted Riemannian distance
    
    Parameters
    ----------
    X : np.ndarray
        First set of SPD/HPD matrices, 3D ndarray.
    W : np.ndarray
        Weight matrix.
    Y : np.ndarray, default None
    
    Returns
    -------
    distances : np.ndarray
        Pairwise weighted Riemannian distances between X and Y or X itself.
    """
    if Y is None:
        Y = X
        
    n_samples_X, n_samples_Y = X.shape[0], Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            d = 0
            for k in range(X.shape[1]):
                d += weighted_riemannian_distance_1(X[i, k], Y[j, k], W)
            distances[i, j] = d
            
    return distances


def pairwise_weighted_riemannian_distance_2(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray = None,
) -> np.ndarray:
    r"""Compute the pairwise weighted Reimannian distance between two sets of SPD/HPD matrices.
    
    The second type of pairwise weighted Riemannian distance
    
    Parameters
    ----------
    X : np.ndarray
        First set of SPD/HPD matrices, 3D ndarray.
    W : np.ndarray
        Weight matrix.
    Y : np.ndarray, default None
    
    Returns
    -------
    distances : np.ndarray
        Pairwise weighted Riemannian distances between X and Y or X itself.
    """
    if Y is None:
        Y = X
        
    n_samples_X, n_samples_Y = X.shape[0], Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            d = 0
            for k in range(X.shape[1]):
                d += weighted_riemannian_distance_2(X[i, k], Y[j, k], W)
            distances[i, j] = d
            
    return distances