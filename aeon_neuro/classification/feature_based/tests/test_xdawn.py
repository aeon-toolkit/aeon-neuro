"""Test xDAWN from MNE combined with a scikit-learn classifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from aeon_neuro.classification.feature_based import XDAWNClassifier

CLASSIFIERS = [
    None,
    LinearDiscriminantAnalysis(),
    SVC(kernel="linear", probability=True),
]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_xdawn_classifier_multivariate_equal_length(classifier):
    """Test XDAWNClassifier on multivariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = XDAWNClassifier(classifier, n_components=2, random_state=0)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_xdawn_classifier_multiclass(classifier):
    """Test XDAWNClassifier returns one probability column per class."""
    X, y = make_example_3d_numpy(
        n_cases=30,
        n_channels=4,
        n_timepoints=64,
        n_labels=3,
        random_state=0,
    )

    clf = XDAWNClassifier(classifier, n_components=2, random_state=0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (30, 3)
    assert set(clf.predict(X)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_xdawn_classifier_reproducible(classifier):
    """Test XDAWNClassifier gives the same output for a fixed random state."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    probs1 = (
        XDAWNClassifier(classifier, n_components=2, random_state=0)
        .fit(X, y)
        .predict_proba(X)
    )
    probs2 = (
        XDAWNClassifier(classifier, n_components=2, random_state=0)
        .fit(X, y)
        .predict_proba(X)
    )

    np.testing.assert_allclose(probs1, probs2)


def test_xdawn_classifier_default_lda():
    """Test the default classifier is a shrinkage LinearDiscriminantAnalysis."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = XDAWNClassifier(n_components=2).fit(X, y)

    assert isinstance(clf.classifier_, LinearDiscriminantAnalysis)
    assert clf.classifier_.solver == "lsqr"
    assert clf.classifier_.shrinkage == "auto"


def test_xdawn_classifier_passed_classifier_is_cloned():
    """Test a passed classifier keeps its settings and is not fitted in place."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.5)

    clf = XDAWNClassifier(lda, n_components=2).fit(X, y)

    assert isinstance(clf.classifier_, LinearDiscriminantAnalysis)
    assert clf.classifier_ is not lda
    assert clf.classifier_.solver == "eigen"
    assert clf.classifier_.shrinkage == 0.5
    assert not hasattr(lda, "classes_")


def test_xdawn_classifier_without_predict_proba():
    """Test classifiers without predict_proba give one-hot probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = XDAWNClassifier(RidgeClassifier(), n_components=2).fit(X, y)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert set(np.unique(probs)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(clf.classes_[probs.argmax(axis=1)], preds)


def test_xdawn_classifier_feature_shape():
    """Test features are the flattened xDAWN series of each case."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = XDAWNClassifier(n_components=2).fit(X, y)

    assert clf._xdawn_features(X).shape == (12, 2 * 2 * 64)


def test_xdawn_classifier_decimate_feature_shape():
    """Test decimate reduces the number of timepoints per xDAWN component."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = XDAWNClassifier(n_components=2, decimate=4).fit(X, y)

    assert clf._xdawn_features(X).shape == (12, 2 * 2 * 16)


def test_xdawn_classifier_decimate_one_matches_default():
    """Test decimate=1 gives the same probabilities as the default."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    probs_default = XDAWNClassifier(n_components=2).fit(X, y).predict_proba(X)
    probs_one = XDAWNClassifier(n_components=2, decimate=1).fit(X, y).predict_proba(X)

    np.testing.assert_allclose(probs_default, probs_one)


@pytest.mark.parametrize("decimate", [0, -1, 2.5, True])
def test_xdawn_classifier_invalid_decimate(decimate):
    """Test decimate must be an integer >= 1."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    with pytest.raises(ValueError, match="decimate must be an integer"):
        XDAWNClassifier(n_components=2, decimate=decimate).fit(X, y)


def test_xdawn_classifier_detects_erp():
    """Test XDAWNClassifier separates trials with and without an ERP."""
    rng = np.random.RandomState(0)
    n_cases, n_channels, n_timepoints = 200, 8, 100
    X = rng.randn(n_cases, n_channels, n_timepoints)
    y = np.arange(n_cases) % 2
    pattern = rng.randn(n_channels)
    bump = np.exp(-0.5 * ((np.arange(n_timepoints) - 50) / 5) ** 2)
    X[y == 1] += 0.5 * pattern[:, None] * bump[None, :]

    clf = XDAWNClassifier(
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), n_components=2
    )
    clf.fit(X[:100], y[:100])

    assert np.mean(clf.predict(X[100:]) == y[100:]) > 0.8


@pytest.mark.parametrize("classifier", ["lda", LinearDiscriminantAnalysis])
def test_xdawn_classifier_invalid_classifier(classifier):
    """Test a string or an uninstantiated class raises a TypeError."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    with pytest.raises(TypeError, match="must be a scikit-learn classifier"):
        XDAWNClassifier(classifier, n_components=2).fit(X, y)
