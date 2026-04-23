"""Burg-type Nuttall-Strand algorithm."""

import numpy as np
from scipy.linalg import solve
from aeon.transformations.series.base import BaseSeriesTransformer


class NuttallStrand(BaseSeriesTransformer):
    """Power spectral density matrix transformer.

    Estimate the power spectral density matrix between channels in the frequency
    domain using Burg-type Nuttall-Strand algorithm, a.k.a. multivariate Burg.
    
    The result is a set of Hermitian positive definite (HPD) 
    complex-valued matrices of shape (n_channels, n_channels).

    Parameters
    ----------
    model_order : int
        The order of the model used for spectral estimation.
        Better to be less than 15. 
        If the data is complex, the model_order should be smaller.
    n_freqs : int, optional
        Number of power spectral density matrices in the output for one sample.
        Number of frequency points in the curve representation of X after transformation,
        by default 10.
    fmax : float, optional
        Maximum frequency of interest in Hz, by default 30 * 2 * pi.
    
    References
    ----------
    .. [1] O. Strand, Multichannel complex maximum entropy (autoregressive) spectral analysis,
    in IEEE Transactions on Automatic Control, vol. 22, no. 4, pp. 634-640,
    August 1977, doi: 10.1109/TAC.1977.1101545.
    .. [2] Nuttall, Albert H.., Multivariate Linear Predictive Spectral Analysis Employing
    Weighted Forward and Backward Averaging: A Generalization of Burg's Algorithm.” (1976).
    
    Examples
    --------
    >>> from aeon_neuro._wip.transformations.series._nuttall_strand import (
    ...     NuttallStrand)
    >>> from aeon_neuro._wip.distances.tests.test_riemannian_matrix import _is_hpd
    >>> import numpy as np
    >>> n_channels, n_timepoints = 5, 360
    >>> n_freqs=5
    >>> X = np.random.standard_normal(size=(n_channels, n_timepoints))
    >>> transformer = NuttallStrand(model_order=3, n_freqs=n_freqs)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape == (n_freqs, n_channels, n_channels)
    True
    >>> for i in range(n_freqs): 
    ...     _is_hpd(X_transformed[i])
    True
    True
    True
    True
    True
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, model_order, n_freqs=10, fmax=30*2*np.pi):
        self.model_order = model_order
        self.n_freqs = n_freqs
        self.fmax = fmax
        super().__init__(axis=1)  # (n_channels, n_timepoints)

    
    def _get_inv(self, A, rcond_threshold=1e-10, cond_threshold=1e10):
        r"""Calculate the inverse of a matrix."""
        try:
            # Calculate the condition number of the matrix
            cond_A = np.linalg.cond(A)
            if cond_A > cond_threshold:
                # If the condition number is too large, use the pseudo-inverse
                # print(f"Condition number is {cond_A:.2e}, using pinv instead of inv")
                inv_A = np.linalg.pinv(A, rcond=rcond_threshold)
            else:
                # If the condition number is appropriate, calculate the inverse
                inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # If the matrix is singular, use the pseudo-inverse
            # print("Matrix is singular, using pinv instead of inv")
            inv_A = np.linalg.pinv(A, rcond=rcond_threshold)

        return inv_A

    def _solve_Cfw(self, E, G, B, P_fw, P_bw):
        r"""Solve a vector equation."""
        P_fw_inv = self._get_inv(P_fw)
        n = B.shape[0]
        # Construct Kronecker product and vectorization
        I = np.eye(n)
        K1 = np.kron(I, B)
        K2 = np.kron((P_fw_inv @ E).T, P_bw)
        # Construct the full coefficient matrix
        K = K1 + K2
        # Vectorize -2G
        vec_neg_2G = (-2 * G).flatten().reshape(-1, 1)
        # Solve for the vectorized C_N
        vec_C_N = solve(K, vec_neg_2G)

        # Reshape C_N back to matrix form
        C_N = vec_C_N.reshape(n, n).T
        return C_N

    def _update_filter_coefficients(self, F, B, C_fw, C_bw):
        r"""Update filter coefficients."""
        B_bw = B
        n_columns = F.shape[1]
        zero_rows = np.zeros((n_columns, n_columns))

        F_q = np.vstack((F, zero_rows)) + np.vstack((zero_rows, B_bw)) @ C_fw
        B_q = np.vstack((F, zero_rows)) @ C_bw + np.vstack((zero_rows, B_bw))

        return F_q, B_q

    def _compute_psd_matrix(self, F, P_fw, n_channels):
        r"""Calculate power spectral density matrix."""
        psd_matrix = np.zeros((self.n_freqs, n_channels, n_channels), dtype=complex)
        freqs = np.linspace(0, self.fmax, self.n_freqs)

        for i, omega in enumerate(freqs):
            A_w = np.eye(n_channels, dtype=complex)
            A_neg_w = np.eye(n_channels, dtype=complex)
            
            for q in range(1, self.model_order+1):
                assert len(F) == self.model_order + 1
                A_w += F[self.model_order][q*n_channels: (q+1)*n_channels] * np.exp(-1j * omega * q)
                A_neg_w += F[self.model_order][q*n_channels: (q+1)*n_channels] * np.exp(-1j * -omega * q)

            psd_matrix[i] = self._get_inv(A_neg_w) @ P_fw[self.model_order] @ self._get_inv(A_w).T

        return psd_matrix
    
    
    def _nuttall_strand_algorithm(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility

        Returns
        -------
        transformed version of X
        a set of power spectral density matrices of shape (n_freqs, n_channels, n_channels)
        """
        # time_series [n_channels, n_timepoints]
        n_channels, n_timepoints = X.shape
        T = n_timepoints
        # forward and backward power matrices
        P_fw = np.zeros((self.model_order+1, n_channels, n_channels))
        P_bw = np.zeros((self.model_order+1, n_channels, n_channels))
        covariance_matrix = (X @ X.T) / T
        P_fw[0] = covariance_matrix
        P_bw[0] = covariance_matrix

        e = np.zeros((self.model_order+1, n_channels, T))
        b = np.zeros((self.model_order+1, n_channels, T))

        C_fw = np.zeros((self.model_order+1, n_channels, n_channels))
        C_bw = np.zeros((self.model_order+1, n_channels, n_channels))

        F = []
        B_bw = []
        F.append(np.eye(n_channels))
        B_bw.append(np.eye(n_channels))

        for q in range(1, self.model_order + 1):
            for k in range(0, (T - q - 1)):
                if q == 1:
                    e[1, :, k] = X[:, k+1]
                    b[1, :, k] = X[:, k]
                else:
                    e[q, :, k] = e[q-1, :, k+1] + C_fw[q-1].T @ b[q-1, :, k+1]
                    b[q, :, k] = b[q-1, :, k] + C_bw[q-1].T @ e[q-1, :, k]

            E = (e[q] @ e[q].T) / (T-q)
            G = (b[q] @ e[q].T) / (T-q)
            Bi = (b[q] @ b[q].T) / (T-q)
            
            C_fw[q] = self._solve_Cfw(E, G, Bi, P_fw[q-1], P_bw[q-1])
            C_bw[q] = self._get_inv(P_fw[q-1]) @ C_fw[q].T @ P_bw[q-1]

            P_fw[q] = P_fw[q-1] - C_fw[q].T @ P_bw[q-1] @ C_fw[q]
            P_bw[q] = P_bw[q-1] - C_bw[q].T @ P_fw[q-1] @ C_bw[q]

            F_q, B_q = self._update_filter_coefficients(F[q-1], B_bw[q-1], C_fw[q], C_bw[q])
            F.append(F_q)
            B_bw.append(B_q)

        return self._compute_psd_matrix(F, P_fw, n_channels)
    
    
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility

        Returns
        -------
        transformed version of X
        a set of power spectral density matrices of shape (n_freqs, n_channels, n_channels)
        """
        
        return self._nuttall_strand_algorithm(X)