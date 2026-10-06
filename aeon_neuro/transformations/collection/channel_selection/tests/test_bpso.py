"""Tests for the binary particle swarm channel selector."""

import numpy as np
import pytest
from aeon.classification import DummyClassifier

from aeon_neuro.transformations.collection.channel_selection import BPSO


@pytest.fixture
def classification_data():
    """Create a small balanced multivariate classification problem."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 4, 8))
    y = np.repeat([0, 1], 10)
    X[y == 1, 0] += 3
    return X, y


def test_bpso_selects_requested_number_of_channel_indices(classification_data):
    """BPSO should expose channel indices and transform to the requested size."""
    X, y = classification_data
    selector = BPSO(
        proportion=0.5,
        n_particles=4,
        max_iter=2,
        estimator=DummyClassifier(),
        random_state=0,
    )

    transformed = selector.fit_transform(X, y)

    assert len(selector.channels_selected_) == 2
    assert len(set(selector.channels_selected_)) == 2
    assert all(0 <= channel < X.shape[1] for channel in selector.channels_selected_)
    assert transformed.shape == (20, 2, 8)
    assert 0 <= selector.best_score_ <= 1


def test_bpso_is_reproducible(classification_data):
    """The same random state should produce the same fitted result."""
    X, y = classification_data
    params = {
        "proportion": 0.5,
        "n_particles": 4,
        "max_iter": 3,
        "estimator": DummyClassifier(),
        "random_state": 42,
    }

    first = BPSO(**params).fit(X, y)
    second = BPSO(**params).fit(X, y)

    assert first.channels_selected_ == second.channels_selected_
    assert first.best_score_ == second.best_score_


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("proportion", 0, "proportion"),
        ("proportion", 1.1, "proportion"),
        ("n_particles", 0, "n_particles"),
        ("max_iter", 0, "max_iter"),
        ("inertia", -0.1, "inertia"),
        ("cognitive", -0.1, "cognitive"),
        ("social", -0.1, "social"),
    ],
)
def test_bpso_rejects_invalid_parameters(
    classification_data, parameter, value, message
):
    """Invalid optimisation parameters should fail with a clear error."""
    X, y = classification_data
    params = {
        "n_particles": 2,
        "max_iter": 1,
        "estimator": DummyClassifier(),
        "random_state": 0,
        parameter: value,
    }

    with pytest.raises(ValueError, match=message):
        BPSO(**params).fit(X, y)


def test_particle_sampling_preserves_subset_size():
    """Every sampled particle should retain exactly the requested channels."""
    rng = np.random.RandomState(7)
    probabilities = rng.uniform(size=(10, 6))

    particles = BPSO._sample_particles(probabilities, rng, n_selected=3)

    np.testing.assert_array_equal(particles.sum(axis=1), np.full(10, 3))
