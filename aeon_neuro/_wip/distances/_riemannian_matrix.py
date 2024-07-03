"""Riemannian distances between SPD/HPD matrices."""

import numpy as np
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import sqrtm


def _is_hpd(matrix):
    r"""Check if a matrix is Hermitian Positive Definite (HPD)."""
    if not np.allclose(matrix, matrix.conj().T):
        return False
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def _check_inputs(A, B, W=None):
    if W is None:
        if not A.shape == B.shape:
            raise ValueError("Inputs must have equal dimensions")
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Inputs must be 2D ndarrays")
        if not _is_hpd(A):
            raise ValueError("Matrix A must be Hermitian Positive Definite (HPD)")
        if not _is_hpd(B):
            raise ValueError("Matrix B must be Hermitian Positive Definite (HPD)")
    else:
        if not A.shape == B.shape == W.shape:
            raise ValueError("Inputs must have equal dimensions")
        if A.ndim != 2 or B.ndim != 2 or W.ndim != 2:
            raise ValueError("Inputs must be 2D ndarrays")
        if not _is_hpd(A):
            raise ValueError("Matrix A must be Hermitian Positive Definite (HPD)")
        if not _is_hpd(B):
            raise ValueError("Matrix B must be Hermitian Positive Definite (HPD)")
        if not _is_hpd(W):
            raise ValueError("Matrix W must be Hermitian Positive Definite (HPD)")


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
        d_{R1}(\mathbf{A}, \mathbf{B}) =
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
    _check_inputs(A, B)
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
        d_{R2}(\mathbf{A}, \mathbf{B}) =
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
    _check_inputs(A, B)
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
        d_{W1}(\mathbf{A}, \mathbf{B}, \mathbf{W}) =
        \sqrt{\text{tr} \mathbf{W} \mathbf{A} +
        \text{tr} \mathbf{W} \mathbf{B} -
        2 \text{tr} \mathbf{A}^{1/2} \mathbf{W} \mathbf{B} \mathbf{W} \mathbf{A}^{1/2}}

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
    _check_inputs(A, B, W)
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
        d_{W2}(\mathbf{A}, \mathbf{B}, \mathbf{W}) =
        \sqrt{\text{tr} \mathbf{W} \mathbf{A} +
        \text{tr} \mathbf{W} \mathbf{B} -
        \text{tr} \mathbf{W} \mathbf{A}^{1/2} \mathbf{B}^{1/2} -
        \text{tr} \mathbf{W} \mathbf{B}^{1/2} \mathbf{A}^{1/2}}

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
    _check_inputs(A, B, W)
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
