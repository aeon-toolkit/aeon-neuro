"""Tests for PSDWelchTransformer."""

__maintainer__ = [""]

from aeon.testing.data_generation import make_series

from aeon_neuro._wip.transformations.series._psd_welch import PSDWelchTransformer


def test_pca():
    """Test PSD Welch transformer."""
    X = make_series(n_columns=3, return_numpy=True).T
    transformer = PSDWelchTransformer()
    Xt = transformer.fit_transform(X)
    # test that the shape is correct
    assert Xt.shape == (X.shape[0], 1)
