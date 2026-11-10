"""Tests for DifferentialEntropyClassifier."""

import numpy as np
from aeon.testing.estimator_checking import check_estimator
from sklearn.linear_model import RidgeClassifier

from aeon_neuro.classification.feature_based import DifferentialEntropyClassifier


def _alpha_power_problem(n_cases, rng, sfreq=128.0, n_timepoints=256):
    """Class "high" has a stronger 10 Hz rhythm on channel 0 than "low"."""
    t = np.arange(n_timepoints) / sfreq
    y = np.repeat(["low", "high"], n_cases // 2)
    X = rng.standard_normal((n_cases, 3, n_timepoints))
    gain = np.where(y == "high", 3.0, 0.5)
    phase = rng.uniform(0, 2 * np.pi, n_cases)
    X[:, 0] += gain[:, None] * np.sin(2 * np.pi * 10 * t[None] + phase[:, None])
    return X, y


def test_differential_entropy_classifier_estimator_checks():
    """Run aeon's estimator checks."""
    check_estimator(DifferentialEntropyClassifier, raise_exceptions=True)


def test_differential_entropy_classifier_separates_band_power():
    """A difference in alpha power is learned and generalises."""
    rng = np.random.default_rng(0)
    X_train, y_train = _alpha_power_problem(40, rng)
    X_test, y_test = _alpha_power_problem(40, rng)
    clf = DifferentialEntropyClassifier(sfreq=128.0, random_state=0)
    clf.fit(X_train, y_train)
    assert np.mean(clf.predict(X_test) == y_test) >= 0.9
    proba = clf.predict_proba(X_test)
    assert proba.shape == (40, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_differential_entropy_classifier_without_predict_proba():
    """A classifier with no predict_proba still gives probabilities."""
    rng = np.random.default_rng(1)
    X, y = _alpha_power_problem(20, rng)
    clf = DifferentialEntropyClassifier(sfreq=128.0, classifier=RidgeClassifier())
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (20, 2)
    assert set(np.unique(proba)) <= {0.0, 1.0}
