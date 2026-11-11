"""Tests for TRCAClassifier and TemplateCCAClassifier."""

import numpy as np
import pytest
from aeon.testing.estimator_checking import check_estimator

from aeon_neuro.classification.distance_based import (
    TemplateCCAClassifier,
    TRCAClassifier,
)


def _ssvep_problem(n_per_class, rng, sfreq=250.0, n_timepoints=250, noise=3.0):
    """Four classes, each a phase-locked sinusoid mixed into 6 noisy channels."""
    freqs = [8.0, 10.0, 12.0, 15.0]
    t = np.arange(n_timepoints) / sfreq
    mixing = np.random.default_rng(42).standard_normal(6)
    X, y = [], []
    for k, f in enumerate(freqs):
        source = np.sin(2 * np.pi * f * t + k)
        for _ in range(n_per_class):
            noisy = np.outer(mixing, source) + noise * rng.standard_normal((6, len(t)))
            X.append(noisy)
            y.append(str(k))
    return np.array(X), np.array(y)


@pytest.mark.parametrize("cls", [TRCAClassifier, TemplateCCAClassifier])
def test_template_classifier_estimator_checks(cls):
    """Run aeon's estimator checks."""
    check_estimator(cls, raise_exceptions=True)


@pytest.mark.parametrize(
    "clf",
    [
        TRCAClassifier(),
        TRCAClassifier(ensemble=False),
        TRCAClassifier(n_components=2),
        TRCAClassifier(filter_bank=3, sfreq=250.0),
        TemplateCCAClassifier(),
    ],
)
def test_template_classifier_decodes_ssvep(clf):
    """Noisy phase-locked responses are decoded without stimulus frequencies."""
    rng = np.random.default_rng(0)
    X_train, y_train = _ssvep_problem(10, rng)
    X_test, y_test = _ssvep_problem(10, rng)
    clf.fit(X_train, y_train)
    assert np.mean(clf.predict(X_test) == y_test) >= 0.9
    proba = clf.predict_proba(X_test)
    assert proba.shape == (40, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        clf.classes_[proba.argmax(axis=1)], clf.predict(X_test)
    )


def test_trca_filter_maximises_the_rayleigh_quotient():
    """The TRCA filter beats random directions on (w S w) / (w Q w)."""
    rng = np.random.default_rng(0)
    X, y = _ssvep_problem(8, rng)
    clf = TRCAClassifier().fit(X, y)
    Xc = X[y == "0"] - X[y == "0"].mean(axis=-1, keepdims=True)
    total = Xc.sum(axis=0)
    Q = np.einsum("ict,ijt->cj", Xc, Xc)
    S = total @ total.T - Q

    def quotient(w):
        return (w @ S @ w) / (w @ Q @ w)

    best = quotient(clf.filters_[0][:, 0])
    for _ in range(200):
        assert quotient(rng.standard_normal(6)) <= best + 1e-6


def test_template_cca_scores_a_template_against_itself_as_one():
    """With one training case per class, a training case scores 1 on its class."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 4, 100))
    y = np.array(["a", "b", "c"])
    clf = TemplateCCAClassifier().fit(X, y)
    scores = clf._scores(X)
    np.testing.assert_allclose(np.diag(scores), 1.0, atol=1e-6)
    np.testing.assert_array_equal(clf.predict(X), y)


def test_trca_invalid_n_components():
    """n_components below 1 raises."""
    X, y = _ssvep_problem(3, np.random.default_rng(0))
    with pytest.raises(ValueError, match="n_components"):
        TRCAClassifier(n_components=0).fit(X, y)


def test_filter_bank_trca_bands_and_drift():
    """Sub-bands follow Nakanishi et al., and drift does not hurt the bank."""
    rng = np.random.default_rng(0)
    X_train, y_train = _ssvep_problem(10, rng)
    X_test, y_test = _ssvep_problem(10, rng)
    drift = 50 * np.linspace(-1, 1, X_train.shape[-1])
    X_train = X_train + rng.standard_normal((len(X_train), 6, 1)) * drift
    X_test = X_test + rng.standard_normal((len(X_test), 6, 1)) * drift

    clf = TRCAClassifier(filter_bank=5, sfreq=250.0).fit(X_train, y_train)
    assert clf.bands_ == [
        (6.0, 90.0),
        (14.0, 90.0),
        (22.0, 90.0),
        (30.0, 90.0),
        (38.0, 90.0),
    ]
    np.testing.assert_allclose(clf.band_weights_[0], 1.25)
    assert np.mean(clf.predict(X_test) == y_test) >= 0.9


def test_filter_bank_trca_caps_and_drops_bands():
    """At 64 Hz the upper edge is capped and bands above it are dropped."""
    X, y = _ssvep_problem(3, np.random.default_rng(0), sfreq=64.0, n_timepoints=128)
    clf = TRCAClassifier(filter_bank=5, sfreq=64.0).fit(X, y)
    assert clf.bands_ == [(6.0, 0.45 * 64), (14.0, 0.45 * 64), (22.0, 0.45 * 64)]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"filter_bank": 2}, "sfreq"),
        ({"filter_bank": 0, "sfreq": 250.0}, "filter_bank"),
        ({"filter_bank": 2, "sfreq": 10.0}, "Nyquist"),
    ],
)
def test_filter_bank_trca_invalid_parameters(kwargs, match):
    """A filter bank needs a positive size and a usable sampling rate."""
    X, y = _ssvep_problem(3, np.random.default_rng(0))
    with pytest.raises(ValueError, match=match):
        TRCAClassifier(**kwargs).fit(X, y)
