"""Tests for UMAP channel creation."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection.channel_selection import UMAP


def test_umap_transforms_train_and_test_to_latent_channels():
    """A TRAIN-fitted UMAP model should transform both collection splits."""
    rng = np.random.RandomState(0)
    X_train = rng.normal(size=(6, 4, 5))
    X_test = rng.normal(size=(3, 4, 5))
    transformer = UMAP(n_neighbors=2, n_components=2, random_state=0)

    train_transformed = transformer.fit_transform(X_train)
    test_transformed = transformer.transform(X_test)

    assert train_transformed.shape == (6, 2, 5)
    assert test_transformed.shape == (3, 2, 5)


def test_umap_rejects_different_channel_count():
    """Transform should reject data with a different number of channels."""
    rng = np.random.RandomState(1)
    transformer = UMAP(n_neighbors=2, random_state=0).fit(rng.normal(size=(5, 3, 4)))

    with pytest.raises(ValueError, match="different number of channels"):
        transformer.transform(rng.normal(size=(2, 2, 4)))


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("n_neighbors", 1, "n_neighbors"),
        ("n_components", 0, "n_components"),
        ("min_dist", -0.1, "min_dist"),
        ("min_dist", 1.1, "min_dist"),
    ],
)
def test_umap_rejects_invalid_parameters(parameter, value, message):
    """Invalid UMAP parameters should fail with a clear error."""
    params = {"n_neighbors": 2, parameter: value}

    with pytest.raises(ValueError, match=message):
        UMAP(**params).fit(np.ones((4, 3, 2)))
