"""Test riemannian distance functions."""

import numpy as np
import pytest
import torch
from aeon.testing.data_generation import make_example_3d_numpy
from pyriemann.estimation import Covariances
from pyriemann.utils.covariance import covariances
from sklearn.base import clone

from aeon_neuro.classification.distance_based import (
    RiemannianKNNClassifier,
    RiemannianMDMClassifier,
)


def test_riemannian_mdm_classifier():
    """Basic test for RiemannianMDMClassifier."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=40,
        random_state=0,
    )

    clf = RiemannianMDMClassifier(metric="riemann")
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_riemannian_knn_classifier():
    """Basic test for RiemannianKNNClassifier."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=40,
        random_state=0,
    )

    clf = RiemannianKNNClassifier(n_neighbors=3, metric="riemann")
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


CLASSIFIERS = [
    RiemannianMDMClassifier(metric="riemann"),
    RiemannianKNNClassifier(n_neighbors=3, metric="riemann"),
]


def _make_data():
    return make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=40,
        random_state=0,
    )


@pytest.fixture
def fake_cuda(monkeypatch):
    """Send data down the GPU branch while keeping tensors on the CPU.

    Lets the tensor path (numpy -> torch -> pyRiemann -> numpy) be tested on
    machines without CUDA.
    """
    as_tensor = torch.as_tensor

    def as_tensor_cpu(data, dtype=None, device=None):
        return as_tensor(data, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "as_tensor", as_tensor_cpu)


@pytest.mark.parametrize("estimator", ["scm", "lwf"])
def test_compute_covariances_cpu(estimator, monkeypatch):
    """CPU path returns numpy covariances matching pyRiemann directly."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    X, _ = _make_data()
    clf = RiemannianMDMClassifier(covariance_estimator=estimator)

    X_cov = clf._compute_covariances(X, fit=True)

    assert isinstance(X_cov, np.ndarray)
    assert X_cov.shape == (12, 4, 4)
    np.testing.assert_allclose(X_cov, covariances(X, estimator=estimator))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("estimator", ["scm", "lwf"])
def test_compute_covariances_gpu(estimator, monkeypatch):
    """On a real GPU, covariances are computed on CUDA and match the CPU."""
    X, _ = _make_data()
    clf = RiemannianMDMClassifier(covariance_estimator=estimator)

    # record the device of the data pyRiemann actually receives
    seen_devices = []
    fit_transform = Covariances.fit_transform

    def recording_fit_transform(self, X, y=None):
        seen_devices.append(X.device.type)
        return fit_transform(self, X, y)

    monkeypatch.setattr(Covariances, "fit_transform", recording_fit_transform)

    X_cov = clf._compute_covariances(X, fit=True)

    assert seen_devices == ["cuda"]
    assert isinstance(X_cov, np.ndarray)
    assert X_cov.shape == (12, 4, 4)
    np.testing.assert_allclose(X_cov, covariances(X, estimator=estimator), atol=1e-10)


@pytest.mark.parametrize("estimator", ["scm", "lwf"])
def test_compute_covariances_tensor_path(estimator, fake_cuda):
    """Tensor path returns numpy covariances equal to the numpy path."""
    X, _ = _make_data()
    clf = RiemannianMDMClassifier(covariance_estimator=estimator)

    X_cov = clf._compute_covariances(X, fit=True)

    assert isinstance(X_cov, np.ndarray)
    np.testing.assert_allclose(X_cov, covariances(X, estimator=estimator))


def test_compute_covariances_fit_builds_transformer():
    """fit=True builds a transformer using the configured estimator."""
    X, _ = _make_data()
    clf = RiemannianMDMClassifier(
        covariance_estimator="lwf", covariance_params={"assume_centered": False}
    )
    clf._compute_covariances(X, fit=True)

    assert clf.covariance_transformer_.estimator == "lwf"
    assert clf.covariance_transformer_.kwds == {"assume_centered": False}


@pytest.mark.parametrize("clf", CLASSIFIERS)
def test_tensor_path_matches_numpy_path(clf, monkeypatch, fake_cuda):
    """Predictions through the tensor path match the numpy path."""
    X, y = _make_data()

    tensor_clf = clone(clf).fit(X, y)
    tensor_preds = tensor_clf.predict(X)
    tensor_probs = tensor_clf.predict_proba(X)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    numpy_clf = clone(clf).fit(X, y)

    np.testing.assert_array_equal(tensor_preds, numpy_clf.predict(X))
    np.testing.assert_allclose(tensor_probs, numpy_clf.predict_proba(X))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("clf", CLASSIFIERS)
def test_cuda_matches_cpu(clf, monkeypatch):
    """On a real GPU, results match the CPU."""
    X, y = _make_data()

    gpu_clf = clone(clf).fit(X, y)
    gpu_probs = gpu_clf.predict_proba(X)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cpu_clf = clone(clf).fit(X, y)

    np.testing.assert_allclose(gpu_probs, cpu_clf.predict_proba(X), atol=1e-8)
