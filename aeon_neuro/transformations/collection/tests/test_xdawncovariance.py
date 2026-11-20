"""Tests for the XDAWNCovariances transformer."""

import numpy as np
import pytest
from aeon.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances as PyriemannXdawnCovariances

from aeon_neuro.classification.distance_based import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)
from aeon_neuro.transformations.collection import ERPCovariances, XDAWNCovariances

n_cases, n_channels, n_timepoints = 60, 6, 100


def _make_erp_data(n_cases=n_cases, random_state=0):
    """Return noise where class 1 has a time-locked bump in a spatial pattern."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_cases, n_channels, n_timepoints))
    y = np.arange(n_cases) % 2
    pattern = rng.standard_normal(n_channels)
    t = np.arange(n_timepoints)
    bump = 3 * np.exp(-0.5 * ((t - n_timepoints / 2) / 5) ** 2)
    X[y == 1] += pattern[:, None] * bump[None, :]
    return X, y


@pytest.mark.parametrize(
    "params, n_components",
    [
        ({}, 2 * 2 * 4),
        ({"nfilter": 1}, 2 * 2 * 1),
        ({"nfilter": 10}, 2 * 2 * n_channels),
        ({"nfilter": 2, "applyfilters": False}, 2 * 2 + n_channels),
        ({"nfilter": 2, "classes": [1]}, 2 * 2),
    ],
)
def test_output_shape(params, n_components):
    """Output is one square matrix per case, sized by classes and nfilter."""
    X, y = _make_erp_data()
    Xt = XDAWNCovariances(**params).fit_transform(X, y)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_components, n_components)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"nfilter": 2, "estimator": "lwf", "xdawn_estimator": "oas"},
        {"applyfilters": False},
        {"classes": [1]},
    ],
)
def test_matches_pyriemann(params):
    """Output matches pyRiemann's XdawnCovariances with the same settings."""
    X, y = _make_erp_data()
    X_train, y_train, X_test = X[:40], y[:40], X[40:]
    expected = PyriemannXdawnCovariances(**params).fit(X_train, y_train)
    Xt = XDAWNCovariances(**params).fit(X_train, y_train).transform(X_test)
    np.testing.assert_allclose(Xt, expected.transform(X_test))


def test_numpy_integer_nfilter():
    """Nfilter accepts numpy integers, e.g. from a parameter grid."""
    X, y = _make_erp_data()
    Xt = XDAWNCovariances(nfilter=np.int64(2)).fit_transform(X, y)
    np.testing.assert_allclose(Xt, XDAWNCovariances(nfilter=2).fit_transform(X, y))


def test_univariate_raises():
    """XDAWN needs more than one channel."""
    X, y = _make_erp_data()
    with pytest.raises(ValueError, match="univariate"):
        XDAWNCovariances().fit(X[:, :1], y)


def test_requires_y():
    """Fitting without labels raises an error."""
    X, _ = _make_erp_data()
    with pytest.raises(ValueError, match="requires_y"):
        XDAWNCovariances().fit(X)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"classes": []}, "classes must contain at least one class."),
        ({"classes": [0, 5]}, r"classes \[5\] are not present in y."),
        ({"nfilter": 0}, "nfilter must be an integer of at least 1."),
        ({"nfilter": True}, "nfilter must be an integer of at least 1."),
        ({"nfilter": 1.5}, "nfilter must be an integer of at least 1."),
    ],
)
def test_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X, y = _make_erp_data()
    with pytest.raises(ValueError, match=match):
        XDAWNCovariances(**params).fit_transform(X, y)


@pytest.mark.parametrize("clf", [RiemannianMDMClassifier, RiemannianKNNClassifier])
@pytest.mark.parametrize("covariance_type", ["erp", "xdawn"])
def test_riemannian_classifiers(clf, covariance_type):
    """The Riemannian classifiers separate classes with a time-locked response."""
    X, y = _make_erp_data(n_cases=200)
    X_train, y_train, X_test, y_test = X[:140], y[:140], X[140:], y[140:]
    params = {"nfilter": 2} if covariance_type == "xdawn" else {}
    model = clf(
        covariance_type=covariance_type,
        covariance_estimator="lwf",
        covariance_params=params,
    ).fit(X_train, y_train)
    assert np.mean(model.predict(X_test) == y_test) > 0.9


def test_classifier_uses_xdawn_covariances():
    """covariance_type="xdawn" gives the same model as the transformer + MDM."""
    from pyriemann.classification import MDM

    X, y = _make_erp_data()
    X_train, y_train, X_test = X[:40], y[:40], X[40:]
    transformer = XDAWNCovariances(nfilter=2).fit(X_train, y_train)
    mdm = MDM().fit(transformer.transform(X_train), y_train)
    expected = mdm.predict_proba(transformer.transform(X_test))

    clf = RiemannianMDMClassifier(
        covariance_type="xdawn", covariance_params={"nfilter": 2}
    ).fit(X_train, y_train)
    np.testing.assert_allclose(clf.predict_proba(X_test), expected)


def test_classifier_invalid_covariance_type():
    """An unknown covariance_type raises a ValueError."""
    X, y = _make_erp_data()
    with pytest.raises(ValueError, match="covariance_type must be one of"):
        RiemannianMDMClassifier(covariance_type="nope").fit(X, y)


@pytest.mark.parametrize("clf", [RiemannianMDMClassifier, RiemannianKNNClassifier])
@pytest.mark.parametrize(
    "transformer, covariance_type, params",
    [
        (ERPCovariances, "erp", {"svd": 2}),
        (XDAWNCovariances, "xdawn", {"nfilter": 2}),
    ],
)
def test_precomputed_pipeline(clf, transformer, covariance_type, params):
    """A transformer then a precomputed classifier match the built-in step."""
    X, y = _make_erp_data()
    X_train, y_train, X_test = X[:40], y[:40], X[40:]

    pipeline = make_pipeline(
        transformer(estimator="lwf", **params),
        clf(covariance_type="precomputed"),
    ).fit(X_train, y_train)
    built_in = clf(
        covariance_type=covariance_type,
        covariance_estimator="lwf",
        covariance_params=params,
    ).fit(X_train, y_train)

    np.testing.assert_allclose(
        pipeline.predict_proba(X_test), built_in.predict_proba(X_test)
    )
