"""Tests for the CommonSpatialPatterns spatial filter."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.collection.channel_creation._csp import _ajd_pham


def _data(n_classes=2, n_channels=6, n_per_class=20, average_reference=False):
    """Return data where each class has extra power on a different channel."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(n_classes), n_per_class)
    X = rng.normal(size=(len(y), n_channels, 100))
    for c in range(n_classes):
        X[y == c, c] *= 3
    if average_reference:
        X -= X.mean(axis=1, keepdims=True)
    return X, y


def _concat_cov(Z):
    Z = Z.transpose(1, 0, 2).reshape(Z.shape[1], -1)
    return Z @ Z.T / (Z.shape[1] - 1)


@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("n_components", [1, 2, 4])
def test_csp_returns_spatially_filtered_collection(n_classes, n_components):
    """Test output is (n_cases, n_components, n_timepoints) and uses filters_."""
    X, y = _data(n_classes)
    csp = CommonSpatialPatterns(n_components=n_components)
    Xt = csp.fit_transform(X, y)

    assert Xt.shape == (len(y), n_components, 100)
    assert Xt.dtype == np.float64
    assert csp.filters_.shape == (6, 6)
    assert csp.patterns_.shape == (6, 6)
    expected = np.einsum("kc,nct->nkt", csp.filters_[:n_components], X)
    np.testing.assert_allclose(Xt, expected)


def test_csp_binary_solves_generalised_eigenproblem():
    """Test binary filters satisfy C_1 w = lambda (C_1 + C_2) w, ordered by evals."""
    X, y = _data(2)
    csp = CommonSpatialPatterns().fit(X, y)

    c1, c2 = _concat_cov(X[y == 0]), _concat_cov(X[y == 1])
    for w, lam in zip(csp.filters_, csp.evals_):
        np.testing.assert_allclose(c1 @ w, lam * (c1 + c2) @ w, atol=1e-10)
    assert np.all(np.diff(np.abs(csp.evals_ - 0.5)) <= 1e-12)
    # the most discriminative filters pick out the channels with extra power
    assert set(np.abs(csp.filters_[:2]).argmax(axis=1)) == {0, 1}


def test_csp_alternate_order():
    """Test 'alternate' interleaves the largest and smallest eigenvalues."""
    X, y = _data(2)
    evals = CommonSpatialPatterns(component_order="alternate").fit(X, y).evals_
    s = np.sort(evals)
    np.testing.assert_allclose(evals, [s[-1], s[0], s[-2], s[1], s[-3], s[2]])


@pytest.mark.parametrize("n_classes", [3, 4])
def test_csp_multiclass_jointly_diagonalises(n_classes):
    """Test multiclass filters approximately diagonalise every class covariance."""
    X, y = _data(n_classes)
    csp = CommonSpatialPatterns().fit(X, y)
    assert csp.evals_ is None

    for c in range(n_classes):
        D = csp.filters_ @ _concat_cov(X[y == c]) @ csp.filters_.T
        off_diag = D - np.diag(np.diag(D))
        assert np.abs(off_diag).max() < 0.05 * np.abs(np.diag(D)).max()


def test_ajd_pham_exact_joint_diagonaliser():
    """Test AJD recovers a diagonaliser for exactly jointly diagonal matrices."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    covs = np.array([A @ np.diag(rng.uniform(1, 5, 4)) @ A.T for _ in range(3)])
    V = _ajd_pham(covs.copy(), max_iter=100)
    for cov in covs:
        D = V @ cov @ V.T
        off_diag = D - np.diag(np.diag(D))
        assert np.abs(off_diag).max() < 1e-6 * np.abs(np.diag(D)).max()


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"cov_est": "epoch"},
        {"reg": 0.1},
        {"reg": "ledoit_wolf"},
        {"reg": "oas", "norm_trace": True},
    ],
)
def test_csp_matches_mne(n_classes, params):
    """Test filters and outputs match mne.decoding.CSP up to filter sign."""
    mne = pytest.importorskip("mne")
    from mne.decoding import CSP

    X, y = _data(n_classes)
    with mne.utils.use_log_level("error"):
        ref = CSP(n_components=3, transform_into="csp_space", **params).fit(X, y)
    csp = CommonSpatialPatterns(n_components=3, **params).fit(X, y)

    sign = np.sign(np.sum(ref.filters_ * csp.filters_, axis=1))[:, None]
    np.testing.assert_allclose(csp.filters_ * sign, ref.filters_, rtol=1e-8)
    np.testing.assert_allclose(
        np.var(csp.transform(X), axis=2), np.var(ref.transform(X), axis=2)
    )


def test_csp_rank_deficient():
    """Test average-referenced (rank n_channels - 1) data gives rank filters."""
    X, y = _data(2, average_reference=True)
    csp = CommonSpatialPatterns(n_components=5).fit(X, y)

    assert csp.filters_.shape == (5, 6)
    assert np.all(np.isfinite(csp.transform(X)))
    with pytest.raises(ValueError, match="exceeds the number of CSP filters"):
        CommonSpatialPatterns(n_components=6).fit(X, y)


def test_csp_string_labels():
    """Test fitting with non-integer labels."""
    X, y = _data(3)
    y = np.array(["left", "right", "feet"])[y]
    csp = CommonSpatialPatterns(n_components=2).fit(X, y)
    np.testing.assert_array_equal(csp.classes_, ["feet", "left", "right"])
    assert csp.transform(X).shape == (len(y), 2, 100)


def test_csp_removed_parameters():
    """Test transform_into and log are no longer constructor parameters."""
    params = CommonSpatialPatterns().get_params()
    assert "transform_into" not in params
    assert "log" not in params
    with pytest.raises(TypeError):
        CommonSpatialPatterns(transform_into="average_power")
    with pytest.raises(TypeError):
        CommonSpatialPatterns(log=True)


@pytest.mark.parametrize(
    "params, y, match",
    [
        ({"n_components": 0}, None, "n_components"),
        ({"n_components": 7}, None, "exceeds"),
        ({"cov_est": "bad"}, None, "cov_est"),
        ({"component_order": "bad"}, None, "component_order"),
        ({"reg": "shrunk"}, None, "reg"),
        ({"reg": 1.5}, None, "reg"),
        ({"component_order": "alternate"}, np.repeat([0, 1, 2], 10), "two classes"),
        ({}, np.zeros(30), "at least two classes"),
    ],
)
def test_csp_invalid_parameters(params, y, match):
    """Test invalid parameters and labels raise informative errors."""
    X, y_default = _data(3, n_per_class=10)
    with pytest.raises(ValueError, match=match):
        CommonSpatialPatterns(**params).fit(X, y_default if y is None else y)


def test_csp_requires_y():
    """Test CSP raises if y is not given to fit."""
    X, _ = make_example_3d_numpy(n_cases=10, n_channels=4, random_state=0)
    with pytest.raises(ValueError):
        CommonSpatialPatterns(n_components=2).fit(X)


@pytest.mark.parametrize(
    "case", ["full", "average_reference", "duplicate_channel", "ill_conditioned"]
)
def test_csp_fast_full_rank_check_agrees_with_exact_rank(case):
    """Test the Gram fast path never claims full rank when the exact rank is lower."""
    from aeon_neuro.transformations.collection.channel_creation._csp import (
        _data_rank,
        _is_clearly_full_rank,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8, 200))
    if case == "average_reference":
        X -= X.mean(axis=1, keepdims=True)
    elif case == "duplicate_channel":
        X[:, 7] = X[:, 3]
    elif case == "ill_conditioned":
        X[:, 7] = X[:, 3] + 1e-9 * X[:, 7]

    rank = _data_rank(X)
    fast = _is_clearly_full_rank(X)
    if fast:
        assert rank == 8
    expected = {"full": 8, "average_reference": 7, "duplicate_channel": 7}
    if case in expected:
        assert rank == expected[case]
    if case == "full":
        assert fast
