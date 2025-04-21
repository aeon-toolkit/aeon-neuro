"""Test distance functions."""

import numpy as np
from pyriemann.utils.distance import distance_riemann

from aeon_neuro.distances import affine_invariant_distance


def test_affine():
    """Test affine distance is equivalent to the one in pyriemann package."""
    # Create a random 4x4 matrix
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 4)
    # Make it symmetric positive definite
    SPD1 = A @ A.T + 4 * np.eye(4)
    SPD2 = B @ B.T + 4 * np.eye(4)
    d1 = distance_riemann(SPD1, SPD2)
    d2 = affine_invariant_distance(SPD1, SPD2)
    assert round(d1, 4) == round(d2, 4)
