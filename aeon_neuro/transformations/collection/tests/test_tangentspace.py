"""Tests for the TangentSpace transformer."""

import numpy as np
import pytest
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace as PyriemannTangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from aeon_neuro.transformations.collection import TangentSpace

n_cases, n_channels, n_timepoints = 60, 4, 100
n_features = n_channels * (n_channels + 1) // 2


def _make_mi_data(n_cases=n_cases, random_state=0):
    """Return noise where each class has more power in a different channel.

    This mimics motor imagery, where imagining a left or right hand movement
    changes the power over opposite sides of the motor cortex.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_cases, n_channels, n_timepoints))
    y = np.arange(n_cases) % 2
    X[y == 0, 0] *= 3
    X[y == 1, 1] *= 3
    return X, y


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"estimator": "lwf"},
        {"estimator": "oas", "metric": "logeuclid"},
        {"metric": "euclid"},
    ],
)
def test_matches_pyriemann(params):
    """Output equals pyRiemann's Covariances followed by TangentSpace."""
    X, _ = _make_mi_data()
    Xt = TangentSpace(**params).fit_transform(X)

    covs = Covariances(estimator=params.get("estimator", "scm")).fit_transform(X)
    expected = PyriemannTangentSpace(
        metric=params.get("metric", "riemann")
    ).fit_transform(covs)

    assert Xt.shape == (n_cases, n_features)
    np.testing.assert_allclose(Xt, expected)


def test_reference_point_fitted_on_train_only():
    """Transform uses the reference point from fit, not from the new data."""
    X_train, _ = _make_mi_data(random_state=0)
    X_test, _ = _make_mi_data(n_cases=10, random_state=1)

    Xt = TangentSpace().fit(X_train).transform(X_test)

    ts = PyriemannTangentSpace().fit(Covariances().fit_transform(X_train))
    expected = ts.transform(Covariances().fit_transform(X_test))
    np.testing.assert_allclose(Xt, expected)


def test_tsupdate_recomputes_reference_point():
    """With tsupdate, the reference point is recomputed on the transform data."""
    X_train, _ = _make_mi_data(random_state=0)
    X_test, _ = _make_mi_data(n_cases=10, random_state=1)
    X_test = X_test * 2  # shift the test data away from the training mean

    fixed = TangentSpace().fit(X_train).transform(X_test)
    updated = TangentSpace(tsupdate=True).fit(X_train).transform(X_test)

    assert not np.allclose(fixed, updated)
    # recentred on the test data, so the test vectors average to about zero
    np.testing.assert_allclose(updated.mean(axis=0), 0, atol=1e-6)


def test_distance_to_reference_point_is_preserved():
    """The norm of each vector equals the Riemannian distance to the mean."""
    from pyriemann.utils.distance import distance_riemann

    X, _ = _make_mi_data()
    transformer = TangentSpace().fit(X)
    Xt = transformer.transform(X)

    covs = Covariances().fit_transform(X)
    reference = transformer.tangent_space_.reference_
    expected = [distance_riemann(c, reference) for c in covs]
    np.testing.assert_allclose(np.linalg.norm(Xt, axis=1), expected)


def test_univariate():
    """A single channel gives one feature per case."""
    X, _ = _make_mi_data()
    Xt = TangentSpace().fit_transform(X[:, :1])
    assert Xt.shape == (n_cases, 1)


@pytest.mark.parametrize(
    "classifier",
    [LogisticRegression(), SVC(kernel="linear")],
    ids=["logistic_regression", "svm"],
)
def test_pipeline_classifies_motor_imagery(classifier):
    """Tangent space features let a linear classifier separate the classes."""
    X_train, y_train = _make_mi_data(random_state=0)
    X_test, y_test = _make_mi_data(n_cases=40, random_state=1)

    pipeline = make_pipeline(TangentSpace(), classifier)
    pipeline.fit(X_train, y_train)

    assert pipeline.score(X_test, y_test) >= 0.9


def test_tsupdate_must_be_bool():
    """A non-bool tsupdate is rejected."""
    X, _ = _make_mi_data()
    with pytest.raises(ValueError, match="tsupdate must be a bool"):
        TangentSpace(tsupdate="yes").fit(X)


def test_scm_too_few_timepoints():
    """With scm, fewer timepoints than channels is rejected."""
    X, _ = _make_mi_data()
    with pytest.raises(ValueError, match="needs more timepoints than channels"):
        TangentSpace().fit(X[:, :, :n_channels])


def test_scm_flat_channel():
    """With scm, a flat channel gives a singular covariance and is rejected."""
    X, _ = _make_mi_data()
    X[3, 2] = 1.0
    with pytest.raises(ValueError, match=r"cases \[3\] are not positive definite"):
        TangentSpace().fit(X)


def test_scm_average_referenced():
    """With scm, average-referenced channels are linearly dependent."""
    X, _ = _make_mi_data()
    X = X - X.mean(axis=1, keepdims=True)
    with pytest.raises(ValueError, match="not positive definite"):
        TangentSpace().fit(X)


def test_transform_checks_new_data():
    """The same checks run on the data passed to transform."""
    X, _ = _make_mi_data()
    transformer = TangentSpace().fit(X)
    X_bad = X.copy()
    X_bad[0, 1] = 0.0
    with pytest.raises(ValueError, match="not positive definite"):
        transformer.transform(X_bad)


@pytest.mark.parametrize("estimator", ["lwf", "oas"])
def test_regularised_estimators_handle_rank_deficient_data(estimator):
    """Shrinkage estimators give finite features where scm is rejected."""
    X, _ = _make_mi_data()
    X_avg = X - X.mean(axis=1, keepdims=True)
    X_short = X[:, :, :n_channels]

    for data in (X_avg, X_short):
        Xt = TangentSpace(estimator=estimator).fit_transform(data)
        assert Xt.shape == (n_cases, n_features)
        assert np.all(np.isfinite(Xt))
