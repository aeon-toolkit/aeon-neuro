"""Tests for the ERPCovariances transformer."""

import numpy as np
import pytest
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances as PyriemannERPCovariances
from sklearn.pipeline import make_pipeline

from aeon_neuro.transformations.collection import ERPCovariances

n_cases, n_channels, n_timepoints = 60, 4, 100


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
        ({}, 3 * n_channels),
        ({"svd": 2}, n_channels + 2 * 2),
        ({"svd": 10}, 3 * n_channels),
        ({"classes": [1]}, 2 * n_channels),
        ({"classes": [1], "svd": 1}, n_channels + 1),
    ],
)
def test_output_shape(params, n_components):
    """Output is one square matrix per case, sized by classes and svd."""
    X, y = _make_erp_data()
    Xt = ERPCovariances(**params).fit_transform(X, y)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_components, n_components)


@pytest.mark.parametrize("estimator", ["scm", "lwf", "oas"])
def test_output_is_spd(estimator):
    """Each output matrix is symmetric positive definite."""
    X, y = _make_erp_data()
    Xt = ERPCovariances(estimator=estimator).fit_transform(X, y)
    np.testing.assert_allclose(Xt, np.transpose(Xt, (0, 2, 1)), atol=1e-10)
    assert np.all(np.linalg.eigvalsh(Xt) > 0)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"estimator": "lwf"},
        {"estimator": "oas", "svd": 2},
        {"classes": [1]},
        {"classes": [1, 0], "svd": 1},
    ],
)
def test_matches_pyriemann(params):
    """Output matches pyRiemann's ERPCovariances with the same settings."""
    X, y = _make_erp_data()
    expected = PyriemannERPCovariances(**params).fit(X, y).transform(X)
    Xt = ERPCovariances(**params).fit_transform(X, y)
    np.testing.assert_allclose(Xt, expected)


def test_prototypes_learned_from_training_data_only():
    """Prototypes come from fit data and are not changed by transform."""
    X, y = _make_erp_data()
    X_train, y_train, X_test = X[:40], y[:40], X[40:]

    transformer = ERPCovariances().fit(X_train, y_train)
    prototypes = transformer.erp_covariances_.P_.copy()
    expected_prototypes = np.concatenate(
        [X_train[y_train == c].mean(axis=0) for c in (0, 1)]
    )
    np.testing.assert_allclose(prototypes, expected_prototypes)

    Xt = transformer.transform(X_test)
    np.testing.assert_array_equal(transformer.erp_covariances_.P_, prototypes)

    expected = PyriemannERPCovariances().fit(X_train, y_train).transform(X_test)
    np.testing.assert_allclose(Xt, expected)


def test_string_labels():
    """String class labels work, including in classes."""
    X, y = _make_erp_data()
    y_str = np.where(y == 1, "target", "nontarget")
    Xt = ERPCovariances(classes=["target"]).fit_transform(X, y_str)
    expected = ERPCovariances(classes=[1]).fit_transform(X, y)
    np.testing.assert_allclose(Xt, expected)


def test_requires_y():
    """Fitting without labels raises an error."""
    X, _ = _make_erp_data()
    with pytest.raises(ValueError, match="requires_y"):
        ERPCovariances().fit(X)


def test_transform_different_length_raises():
    """Transforming series of a different length to fit raises an error."""
    X, y = _make_erp_data()
    transformer = ERPCovariances().fit(X, y)
    with pytest.raises(ValueError, match="different length"):
        transformer.transform(X[:, :, :50])


@pytest.mark.parametrize(
    "params, match",
    [
        ({"classes": []}, "classes must contain at least one class."),
        ({"classes": [1, 1]}, "classes must not contain duplicates."),
        ({"classes": [0, 5]}, r"classes \[5\] are not present in y."),
        ({"svd": 0}, "svd must be None or an integer of at least 1."),
        ({"svd": -1}, "svd must be None or an integer of at least 1."),
        ({"svd": True}, "svd must be None or an integer of at least 1."),
        ({"svd": 1.5}, "svd must be None or an integer of at least 1."),
        ({"svd": "2"}, "svd must be None or an integer of at least 1."),
        ({"estimator": "nope"}, "Unknown function name"),
    ],
)
def test_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X, y = _make_erp_data()
    with pytest.raises(ValueError, match=match):
        ERPCovariances(**params).fit_transform(X, y)


def test_mdm_pipeline():
    """ERP covariances with MDM separate classes with a time-locked response."""
    X, y = _make_erp_data(n_cases=200)
    X_train, y_train, X_test, y_test = X[:140], y[:140], X[140:], y[140:]
    pipeline = make_pipeline(ERPCovariances(estimator="lwf"), MDM())
    pipeline.fit(X_train, y_train)
    assert pipeline.score(X_test, y_test) > 0.9
