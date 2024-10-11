"""Calculate optimal weights for weighted Riemannian distance."""

import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import eig


def _get_inv(A, rcond_threshold=1e-10, cond_threshold=1e10):
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


def _svd_w1(P_ik, P_jk):
    P_half_ik = sqrtm(P_ik)
    P_half_jk = sqrtm(P_jk)

    # Singular Value Decomposition (SVD)
    product_matrix = P_half_jk @ P_half_ik
    # if product_matrix.dtype == np.complex256:
    #     product_matrix = product_matrix.astype(np.complex128)
    V_ij1, Sigma_ij, V_ij2_H = np.linalg.svd(product_matrix)
    
    # Construct unitary matrices
    U_ik = V_ij1
    U_jk = V_ij2_H.T.conj()
    
    P_tilde_ik = P_half_ik @ U_ik
    P_tilde_jk = P_half_jk @ U_jk

    return P_tilde_ik, P_tilde_jk


def _compute_summation_matrices_w1(X_train_psd, y_train):
    n_cases, k, M, M = X_train_psd.shape

    M_S1 = np.zeros((M, M), dtype=np.complex128)
    M_D1 = np.zeros((M, M), dtype=np.complex128)

    # Compute the summation matrices
    for i in range(n_cases):
        for j in range(i+1, n_cases):
            for m in range(k):
                P_ik = X_train_psd[i, m]
                P_jk = X_train_psd[j, m]

                P_tilde_ik, P_tilde_jk = _svd_w1(P_ik, P_jk)

                diff = P_tilde_ik - P_tilde_jk
                diff_H = diff.conj().T 

                if y_train[i] == y_train[j]:
                    M_S1 += diff @ diff_H
                else:
                    M_D1 += diff @ diff_H

    return M_S1, M_D1


def _compute_summation_matrices_w2(X_train_psd, y_train):
    n_cases, k, M, M = X_train_psd.shape

    M_S1 = np.zeros((M, M), dtype=np.complex128)
    M_D1 = np.zeros((M, M), dtype=np.complex128)
    
    # Compute the summation matrices
    for i in range(n_cases):
        for j in range(i+1, n_cases):
            for m in range(k):
                P_ik = X_train_psd[i, m]
                P_jk = X_train_psd[j, m]
                
                P_half_ik = sqrtm(P_ik)
                P_half_jk = sqrtm(P_jk)

                diff = P_half_ik - P_half_jk
                diff_H = diff.conj().T 
                
                if y_train[i] == y_train[j]:
                    M_S1 += diff @ diff_H
                else:
                    M_D1 += diff @ diff_H

    return M_S1, M_D1


def _eigenvector_decomposition(M_S1, M_D1, PCA_num):
    # Perform eigenvalue decomposition on the matrix M_S1^-1 * M_D1
    eigenvalues, eigenvectors = eig(_get_inv(M_S1) @ M_D1)

    # Sort the eigenvectors based on the eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, idx[:PCA_num]]

    return top_eigenvectors

def _compute_weighting_matrix(top_eigenvectors):
    Omega = top_eigenvectors
    W = Omega @ Omega.conj().T
    return W


def get_optimal_weight1(X_train_psd, y_train):
    """Calculate optimal weights for weighted Riemannian distance 1.
    
    Parameters
    ----------
    X_train_psd : np.ndarray
        Power spectral density matrices of shape (n_cases, n_freqs, n_channels, n_channels).
        Get from Burg-type Nuttall-Strand transformation.
    y_train : np.ndarray
        Labels of shape (n_cases,).
        
    Returns
    -------
    W1 : np.ndarray
        Optimal weighting matrix of shape (n_channels, n_channels).
    """
    PCA_num = X_train_psd.shape[-1]
    M_S1, M_D1 = _compute_summation_matrices_w1(X_train_psd, y_train)
    top_eigenvectors = _eigenvector_decomposition(M_S1, M_D1, PCA_num)
    W1 = _compute_weighting_matrix(top_eigenvectors)
    return W1


def get_optimal_weight2(X_train_psd, y_train):
    """Calculate optimal weights for weighted Riemannian distance 2.
    
    Parameters
    ----------
    X_train_psd : np.ndarray
        Power spectral density matrices of shape (n_cases, n_freqs, n_channels, n_channels).
        Get from Burg-type Nuttall-Strand transformation.
    y_train : np.ndarray
        Labels of shape (n_cases,).
        
    Returns
    -------
    W2 : np.ndarray
        Optimal weighting matrix of shape (n_channels, n_channels).
    """
    PCA_num = X_train_psd.shape[-1]
    M_S2, M_D2 = _compute_summation_matrices_w2(X_train_psd, y_train)
    top_eigenvectors = _eigenvector_decomposition(M_S2, M_D2, PCA_num)
    W2 = _compute_weighting_matrix(top_eigenvectors)
    return W2