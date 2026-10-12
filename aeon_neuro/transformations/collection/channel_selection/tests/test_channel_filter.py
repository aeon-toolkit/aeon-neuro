"""Tests for the ChannelFilter transformer."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection import channel_selection
from aeon_neuro.transformations.collection.channel_selection import ChannelFilter


def test_channel_filter_public_export():
    """ChannelFilter should be available through the public API."""
    assert "ChannelFilter" in channel_selection.__all__
    assert hasattr(channel_selection, "ChannelFilter")


@pytest.fixture
def channel_data():
    """Create data with one strongly class-dependent channel."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 0.1, size=(30, 4, 20))
    y = np.repeat([0, 1], 15)

    # Channel 0 contains a strong signal related to the class label.
    X[:, 0, :] += 6.0 * y[:, None]

    return X, y


@pytest.mark.parametrize(
    "score",
    ["variance", "class_mean", "mutual_information"],
)
def test_builtin_scores_select_signal_channel(channel_data, score):
    """Built-in scores should identify the informative channel."""
    X, y = channel_data

    score_params = {"random_state": 0} if score == "mutual_information" else None

    selector = ChannelFilter(
        score_channel=score,
        proportion=0.25,
        score_params=score_params,
    )

    transformed = selector.fit_transform(X, y)

    assert transformed.shape == (30, 1, 20)
    assert selector.scores_.shape == (4,)

    np.testing.assert_array_equal(selector.channels_selected_, [0])
    np.testing.assert_array_equal(transformed, X[:, [0], :])


def test_channel_filter_custom_scorer(channel_data):
    """A custom scorer should receive its keyword arguments."""
    X, y = channel_data

    def custom_score(X, y=None, offset=0.0):
        return float(np.mean(X) + offset)

    selector = ChannelFilter(
        score_channel=custom_score,
        proportion=0.25,
        score_params={"offset": 1.0},
    )

    transformed = selector.fit_transform(X, y)

    assert transformed.shape == (30, 1, 20)
    np.testing.assert_array_equal(selector.channels_selected_, [0])
    np.testing.assert_array_equal(transformed, X[:, [0], :])


@pytest.mark.parametrize(
    "params, message",
    [
        ({"proportion": 0}, "proportion"),
        ({"proportion": 1.1}, "proportion"),
        ({"score_channel": "invalid"}, "score_channel"),
        ({"redundancy_method": "invalid"}, "redundancy_method"),
        (
            {
                "redundancy_method": "correlation",
                "redundancy_threshold": -0.1,
            },
            "redundancy_threshold",
        ),
    ],
)
def test_channel_filter_invalid_parameters(channel_data, params, message):
    """Invalid configuration should raise a descriptive ValueError."""
    X, y = channel_data
    config = {"proportion": 0.75, **params}

    with pytest.raises(ValueError, match=message):
        ChannelFilter(**config).fit(X, y)


@pytest.mark.parametrize(
    "score",
    ["class_mean", "mutual_information"],
)
def test_supervised_scores_require_labels(channel_data, score):
    """Supervised scorers should reject missing class labels."""
    X, _ = channel_data

    selector = ChannelFilter(
        score_channel=score,
        proportion=0.5,
    )

    with pytest.raises(ValueError, match="y is required"):
        selector.fit(X)
