"""Tests for the xDAWN transformer."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.transformations.collection.channel_creation import XDAWN


def _make_erp_data(n_cases=100, n_channels=8, n_timepoints=100, random_state=0):
    """Return noise where class 1 has a time-locked bump in a spatial pattern."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_cases, n_channels, n_timepoints)
    y = np.arange(n_cases) % 2
    pattern = rng.randn(n_channels)
    bump = np.exp(-0.5 * ((np.arange(n_timepoints) - n_timepoints / 2) / 5) ** 2)
    X[y == 1] += pattern[:, None] * bump[None, :]
    return X, y, bump


@pytest.mark.parametrize("n_components, n_channels", [(1, 4), (2, 8), (4, 2)])
def test_xdawn_output_shape(n_components, n_channels):
    """Test output has n_classes * min(n_channels, n_components) channels."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=n_channels, n_timepoints=30, random_state=0
    )

    Xt = XDAWN(n_components=n_components).fit_transform(X, y)

    assert Xt.shape == (20, 2 * min(n_channels, n_components), 30)
    assert Xt.dtype == np.float64


def test_xdawn_transform_matches_fit_transform():
    """Test transform after fit gives the same output as fit_transform."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=4, n_timepoints=30, random_state=0
    )

    xdawn = XDAWN(n_components=2)
    Xt1 = xdawn.fit_transform(X, y)
    Xt2 = xdawn.transform(X)

    np.testing.assert_allclose(Xt1, Xt2)


def test_xdawn_enhances_erp():
    """Test the first xDAWN component recovers the time-locked response."""
    X, y, bump = _make_erp_data()

    Xt = XDAWN(n_components=1).fit_transform(X, y)

    # The class 1 average of the first class 1 component should follow the bump
    # much more closely than any single raw channel does.
    erp = Xt[y == 1, 1].mean(axis=0)
    xdawn_corr = abs(np.corrcoef(erp, bump)[0, 1])
    raw_corr = max(
        abs(np.corrcoef(X[y == 1, c].mean(axis=0), bump)[0, 1])
        for c in range(X.shape[1])
    )

    assert xdawn_corr > 0.95
    assert xdawn_corr >= raw_corr


def test_xdawn_requires_y():
    """Test fit without y raises a ValueError."""
    X, _ = make_example_3d_numpy(
        n_cases=20, n_channels=4, n_timepoints=30, random_state=0
    )

    with pytest.raises(ValueError, match="requires_y"):
        XDAWN().fit(X)
