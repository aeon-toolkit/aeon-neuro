"""Tests for the UMAP embedding transformer."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection.embedding._umap import UMAP


def test_umap_rejects_different_series_length():
    """Transform should reject series of a different length to those in fit."""
    pytest.importorskip("umap")
    rng = np.random.RandomState(0)
    transformer = UMAP(n_neighbors=3, random_state=0).fit(
        rng.normal(size=(10, 1, 50))
    )

    with pytest.raises(ValueError, match="different length"):
        transformer.transform(rng.normal(size=(10, 1, 40)))
