"""Tests for the TRCommonSpatialPatterns spatial filter."""

import numpy as np
import pytest

from aeon_neuro.transformations.collection.channel_creation import (
    TRCommonSpatialPatterns,
)


def _data(n_channels=6, n_per_class=20):
    """Return binary data where class c has extra power on channel c."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], n_per_class)
    X = rng.normal(size=(len(y), n_channels, 100))
    for c in range(2):
        X[y == c, c] *= 3
    return X, y


# MOABB TRCSP(nfilter=4, alpha=alpha).filters_ on the trace-normalised
# covariances np.cov(case) / trace of each case of _data(), generated offline
# by Claude (Anthropic) with moabb 1.8.0 (commit 8901391). Rows 0-1 favour
# class 1, rows 2-3 class 0.
MOABB_FILTERS = {
    0.01: np.array(
        [
            [0.0118852042, -3.5074281555, -0.1388119780, -0.0612374097,
             0.0364403423, 0.1336466340],
            [0.0092656647, -0.0297764712, 1.7612801950, 0.4164261107,
             -0.0856277912, -3.1790182840],
            [3.4904500304, -0.0055614882, 0.0482115010, 0.0266521923,
             -0.0326318996, -0.1772429994],
            [-0.0341287389, -0.0088951129, -0.5260741731, 1.4661823617,
             -3.2462191636, -0.0221277296],
        ]
    ),
    0.1: np.array(
        [
            [0.0071836456, -2.4154495338, -0.0612138367, -0.0184784705,
             0.0170281909, 0.0351839123],
            [-0.0206114684, 0.0022831141, 0.2358980754, -0.2477768499,
             -0.1677470466, -2.4154941007],
            [2.4095386014, -0.0022138456, 0.0079205986, 0.0084918560,
             -0.0228992626, -0.0458503651],
            [-0.0233153315, -0.0135434122, -0.1445456161, 0.8022456731,
             -2.3034043514, 0.0467176008],
        ]
    ),
    1.0: np.array(
        [
            [0.0002456222, -0.9659810864, -0.0150675592, -0.0005805920,
             0.0042260234, -0.0017462793],
            [-0.0623339097, 0.0098165595, -0.4602657109, -0.2960690900,
             -0.0351836606, -0.7911940549],
            [0.9655862333, 0.0019006720, -0.0041063361, 0.0006548576,
             -0.0092041331, 0.0048336701],
            [0.0090762633, 0.0215838924, -0.1015202677, -0.2604567986,
             0.9208593035, -0.0975826274],
        ]
    ),
}  # fmt: skip


def _align_signs(filters, reference):
    """Flip each row of filters so it points the same way as reference."""
    signs = np.sign(np.sum(filters * reference, axis=1))
    return filters * signs[:, None]


@pytest.mark.parametrize("alpha", sorted(MOABB_FILTERS))
def test_trcsp_matches_moabb(alpha):
    """Filters equal MOABB's TRCSP, up to the sign of each filter."""
    X, y = _data()
    trcsp = TRCommonSpatialPatterns(n_components=4, alpha=alpha).fit(X, y)
    expected = MOABB_FILTERS[alpha]
    np.testing.assert_allclose(
        _align_signs(trcsp.filters_, expected), expected, atol=1e-6
    )


@pytest.mark.parametrize("n_components", [2, 4, 6])
def test_trcsp_returns_spatially_filtered_collection(n_components):
    """Output is (n_cases, n_components, n_timepoints) and uses filters_."""
    X, y = _data()
    trcsp = TRCommonSpatialPatterns(n_components=n_components)
    Xt = trcsp.fit_transform(X, y)
    assert Xt.shape == (len(X), n_components, X.shape[2])
    assert trcsp.filters_.shape == (n_components, X.shape[1])
    np.testing.assert_allclose(Xt, trcsp.filters_ @ X)


def test_trcsp_finds_discriminative_channels():
    """Each class's first filter loads mostly on that class's strong channel."""
    X, y = _data()
    trcsp = TRCommonSpatialPatterns(n_components=4, alpha=0.1).fit(X, y)
    assert np.argmax(np.abs(trcsp.filters_[0])) == 1  # class 1 first
    assert np.argmax(np.abs(trcsp.filters_[2])) == 0


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_trcsp_maximises_regularised_ratio(alpha):
    """No random direction beats the first filter of each class."""
    X, y = _data()
    trcsp = TRCommonSpatialPatterns(n_components=2, alpha=alpha).fit(X, y)
    cov_a = np.mean([(c := np.cov(trial)) / np.trace(c) for trial in X[y == 0]], 0)
    cov_b = np.mean([(c := np.cov(trial)) / np.trace(c) for trial in X[y == 1]], 0)
    eye = alpha * np.eye(X.shape[1])

    def ratio(w, num, den):
        return (w @ num @ w) / (w @ (den + eye) @ w)

    W = np.random.default_rng(1).normal(size=(1000, X.shape[1]))
    for w, num, den in ((trcsp.filters_[0], cov_b, cov_a),
                        (trcsp.filters_[1], cov_a, cov_b)):  # fmt: skip
        best = ratio(w, num, den)
        assert all(ratio(v, num, den) <= best + 1e-12 for v in W)


def test_trcsp_alpha_is_scale_invariant():
    """Rescaling the signals does not change the filters."""
    X, y = _data()
    filters = TRCommonSpatialPatterns(alpha=0.1).fit(X, y).filters_
    scaled = TRCommonSpatialPatterns(alpha=0.1).fit(X * 1e-6, y).filters_
    np.testing.assert_allclose(_align_signs(scaled, filters), filters, atol=1e-8)


def test_trcsp_swapping_labels_swaps_halves():
    """Relabelling the classes swaps the two halves of filters_."""
    X, y = _data()
    filters = TRCommonSpatialPatterns(n_components=4).fit(X, y).filters_
    swapped = TRCommonSpatialPatterns(n_components=4).fit(X, 1 - y).filters_
    expected = np.vstack([filters[2:], filters[:2]])
    np.testing.assert_allclose(_align_signs(swapped, expected), expected, atol=1e-10)


def test_trcsp_string_labels():
    """String labels work and are stored in classes_."""
    X, y = _data()
    labels = np.array(["left", "right"])[y]
    trcsp = TRCommonSpatialPatterns(n_components=2).fit(X, labels)
    np.testing.assert_array_equal(trcsp.classes_, ["left", "right"])
    numeric = TRCommonSpatialPatterns(n_components=2).fit(X, y)
    np.testing.assert_allclose(trcsp.filters_, numeric.filters_)


def test_trcsp_requires_y():
    """Fitting without labels raises an error."""
    X, _ = _data()
    with pytest.raises(ValueError):
        TRCommonSpatialPatterns().fit(X)


@pytest.mark.parametrize(
    "labels, match",
    [
        (np.zeros(40, dtype=int), "contains 1 classes"),
        (np.arange(40) % 3, "contains 3 classes"),
    ],
)
def test_trcsp_binary_only(labels, match):
    """One class or more than two classes raise a ValueError."""
    X, _ = _data()
    with pytest.raises(ValueError, match=match):
        TRCommonSpatialPatterns().fit(X, labels)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"n_components": 3}, "n_components must be even"),
        ({"n_components": 0}, "n_components must be a positive int"),
        ({"n_components": 14}, "needs 7 filters per class"),
        ({"alpha": -0.1}, "alpha must be a non-negative number"),
        ({"alpha": np.inf}, "alpha must be a non-negative number"),
        ({"alpha": True}, "alpha must be a non-negative number"),
        ({"alpha": "1"}, "alpha must be a non-negative number"),
    ],
)
def test_trcsp_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X, y = _data()
    with pytest.raises(ValueError, match=match):
        TRCommonSpatialPatterns(**params).fit(X, y)
