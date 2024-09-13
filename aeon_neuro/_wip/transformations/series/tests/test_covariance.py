"""Tests for covariance matrix transform."""

import numpy as np

from aeon_neuro._wip.transformations.series._covariance import CovarianceMatrix

n_channels, n_timepoints = 7, 12000
rng = np.random.default_rng(seed=0)


def test_transform():
    """Test covariance matrix transformer."""
    transformer = CovarianceMatrix()
    X = rng.normal(scale=2, size=(n_channels, n_timepoints))
    X_transformed = transformer.fit_transform(X)

    # test expected shape
    assert X_transformed.shape == (n_channels, n_channels)

    # test positive semi-definite
    evals = np.linalg.eigvals(X_transformed)
    assert np.all(evals >= 0)

    # test correlation matrix = identity matrix
    transformer.set_params(correlation=True)
    X_transformed = transformer.fit_transform(X)
    np.testing.assert_allclose(X_transformed, np.eye(n_channels), atol=0.02)
