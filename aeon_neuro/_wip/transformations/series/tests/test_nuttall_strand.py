"""Tests for the Nuttall-Strand transformer."""

import numpy as np
from aeon_neuro._wip.transformations.series._nuttall_strand import NuttallStrand
from aeon_neuro._wip.distances.tests.test_riemannian_matrix import _is_hpd

n_channels, n_timepoints = 7, 12000
rng = np.random.default_rng(seed=0)
n_freqs=10


def test_transform():
    """Test Nuttall-Strand power spectral density matrices transformer."""
    transformer = NuttallStrand(model_order=3, n_freqs=n_freqs)
    X = rng.normal(scale=2, size=(n_channels, n_timepoints))
    X_transformed = transformer.fit_transform(X)

    # test expected shape
    assert X_transformed.shape == (n_freqs, n_channels, n_channels)

    # test positive semi-definite
    for i in range(n_freqs):
        assert _is_hpd(X_transformed[i])
    