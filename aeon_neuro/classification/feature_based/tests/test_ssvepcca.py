"""Tests for the SSVEP CCA classifier."""

import numpy as np
import pytest

from aeon_neuro.classification.feature_based import SSVEPCCAClassifier

sfreq, n_channels, n_timepoints = 250, 4, 500
freqs = {"a": 8, "b": 11, "c": 15, "d": 23}
t = np.arange(n_timepoints) / sfreq


def _make_ssvep_data(n_cases=80, random_state=0):
    """Noisy multichannel data where each case oscillates at its class freq."""
    rng = np.random.default_rng(random_state)
    labels = rng.choice(list(freqs), size=n_cases)
    X = rng.standard_normal((n_cases, n_channels, n_timepoints))
    for i, label in enumerate(labels):
        phase = rng.uniform(0, 2 * np.pi)
        weights = rng.standard_normal(n_channels)
        X[i] += weights[:, None] * np.sin(2 * np.pi * freqs[label] * t + phase)
    return X, labels


def test_predict_shape_and_labels():
    """Predictions have the right shape and only contain known labels."""
    X, y = _make_ssvep_data()
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(freqs))


def test_references_shape():
    """One reference per class, each (2 * n_harmonics, n_timepoints)."""
    X, y = _make_ssvep_data()
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs, n_harmonics=3).fit(
        X, y
    )
    assert len(clf._references) == len(freqs)
    for reference in clf._references:
        assert reference.shape == (2 * 3, n_timepoints)


def test_classifies_clean_signal():
    """A strong tone at each class frequency is classified correctly."""
    y = np.array(list(freqs) * 5)
    X = np.zeros((len(y), n_channels, n_timepoints))
    for i, label in enumerate(y):
        X[i] += 5 * np.sin(2 * np.pi * freqs[label] * t)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


def test_predict_proba_sums_to_one():
    """Class probabilities are non-negative and sum to one per case."""
    X, y = _make_ssvep_data(n_cases=20)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(freqs))
    assert np.all(proba >= 0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_references_from_fit_length_only():
    """References depend on fit length and labels, not case values."""
    X, y = _make_ssvep_data()
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    expected = []
    for label in clf.classes_:
        rows = []
        for h in (1, 2):
            rows.append(np.sin(2 * np.pi * h * freqs[label] * t))
            rows.append(np.cos(2 * np.pi * h * freqs[label] * t))
        expected.append(np.array(rows))
    for ref, exp in zip(clf._references, expected):
        np.testing.assert_allclose(ref, exp)


def test_extra_keys_in_class_frequencies_allowed():
    """class_frequencies may list classes not present in y."""
    X, y = _make_ssvep_data()
    extra = dict(freqs, e=27)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=extra).fit(X, y)
    assert set(clf.predict(X)).issubset(set(freqs))


@pytest.mark.parametrize(
    "params, match",
    [
        ({"sfreq": 0}, "sfreq must be finite and positive"),
        ({"sfreq": -250}, "sfreq must be finite and positive"),
        ({"sfreq": np.inf}, "sfreq must be finite and positive"),
        ({"sfreq": "250"}, "sfreq must be finite and positive"),
        ({"n_harmonics": 0}, "n_harmonics must be a positive integer"),
        ({"n_harmonics": True}, "n_harmonics must be a positive integer"),
        ({"n_harmonics": 1.5}, "n_harmonics must be a positive integer"),
        ({"class_frequencies": {}}, "must contain at least one class"),
        (
            {"class_frequencies": {"a": 8, "b": 11, "c": 15, "d": -1}},
            "must be finite and positive",
        ),
        (
            {"class_frequencies": {"a": 8, "b": 11, "c": 15, "d": 200}},
            "below the Nyquist frequency",
        ),
        (
            {"class_frequencies": {"a": 8, "b": 8, "c": 15, "d": 23}},
            "duplicate frequencies",
        ),
        (
            {"class_frequencies": {"a": 8, "b": 11, "c": 15}},
            r"Labels \['d'\] in y are not in class_frequencies",
        ),
    ],
)
def test_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X, y = _make_ssvep_data(n_cases=20)
    kwargs = {"sfreq": sfreq, "class_frequencies": freqs, **params}
    with pytest.raises(ValueError, match=match):
        SSVEPCCAClassifier(**kwargs).fit(X, y)
