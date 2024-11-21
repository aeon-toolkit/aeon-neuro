"""Tests for the Riemannian distance matrix."""

import numpy as np
import pytest

from aeon_neuro._wip.distances._riemannian_matrix import (
    riemannian_distance_1,
    riemannian_distance_2,
    riemannian_distance_3,
    weighted_riemannian_distance_1,
    weighted_riemannian_distance_2,
)

A = np.array([[2, -1j], [1j, 2]])

B = np.array([[3, 0], [0, 3]])

W = np.array([[1, 0], [0, 1]])

C = np.array([[1, 2], [3, 4]])


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


def test_is_hpd():
    assert _is_hpd(A) is True
    assert _is_hpd(B) is True
    assert _is_hpd(C) is False


def test_check_inputs():
    with pytest.raises(ValueError):
        _check_inputs(A, C)
    with pytest.raises(ValueError):
        _check_inputs(A, np.array([1, 2]))
    with pytest.raises(ValueError):
        _check_inputs(A, np.array([[1, 2]]))
    with pytest.raises(ValueError):
        _check_inputs(A, B, np.array([1, 2]))
    with pytest.raises(ValueError):
        _check_inputs(A, B, np.array([[1, 2]]))
    with pytest.raises(ValueError):
        _check_inputs(A, B, C)

    _check_inputs(A, B)
    _check_inputs(A, B, W)


def test_riemannian_distance_1():
    d = riemannian_distance_1(A, B)
    assert d >= 0
    assert np.isclose(d, 0.732, atol=0.001)


def test_riemannian_distance_2():
    d = riemannian_distance_2(A, B)
    assert d >= 0
    assert np.isclose(d, 0.732, atol=0.001)


def test_riemannian_distance_3():
    d = riemannian_distance_3(A, B)
    assert d >= 0
    assert np.isclose(d, 1.099, atol=0.001)


def test_weighted_riemannian_distance_1():
    d_W = weighted_riemannian_distance_1(A, B, W)
    assert d_W >= 0
    assert np.isclose(d_W, 0.732, atol=0.001)


def test_weighted_riemannian_distance_2():
    d_W = weighted_riemannian_distance_2(A, B, W)
    assert d_W >= 0
    assert np.isclose(d_W, 0.732, atol=0.001)
