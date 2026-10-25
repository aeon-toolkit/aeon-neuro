"""Test CSP from MNE combined with a scikit-learn classifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from aeon_neuro.classification.feature_based import CSPClassifier

CLASSIFIERS = [
    None,
    SVC(kernel="linear", probability=True),
    SVC(kernel="rbf", probability=True),
]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_csp_classifier_multivariate_equal_length(classifier):
    """Test CSPClassifier on multivariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPClassifier(classifier, n_components=2, random_state=0)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_csp_classifier_multiclass(classifier):
    """Test CSPClassifier returns one probability column per class."""
    X, y = make_example_3d_numpy(
        n_cases=30,
        n_channels=4,
        n_timepoints=64,
        n_labels=3,
        random_state=0,
    )

    clf = CSPClassifier(classifier, n_components=2, random_state=0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (30, 3)
    assert set(clf.predict(X)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_csp_classifier_reproducible(classifier):
    """Test CSPClassifier gives the same output for a fixed random state."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    probs1 = (
        CSPClassifier(classifier, n_components=2, random_state=0)
        .fit(X, y)
        .predict_proba(X)
    )
    probs2 = (
        CSPClassifier(classifier, n_components=2, random_state=0)
        .fit(X, y)
        .predict_proba(X)
    )

    np.testing.assert_allclose(probs1, probs2)


def test_csp_classifier_default_lda():
    """Test the default classifier is LinearDiscriminantAnalysis."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPClassifier(n_components=2).fit(X, y)

    assert isinstance(clf.classifier_, LinearDiscriminantAnalysis)
    assert clf.classifier_.solver == "svd"


def test_csp_classifier_passed_classifier_is_cloned():
    """Test a passed classifier keeps its settings and is not fitted in place."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    clf = CSPClassifier(lda, n_components=2).fit(X, y)

    assert isinstance(clf.classifier_, LinearDiscriminantAnalysis)
    assert clf.classifier_ is not lda
    assert clf.classifier_.solver == "lsqr"
    assert clf.classifier_.shrinkage == "auto"
    assert not hasattr(lda, "classes_")
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0, atol=1e-6)


def test_csp_classifier_random_state_set_on_classifier():
    """Test random_state reseeds the classifier, and None keeps its own seed."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )
    svc = SVC(probability=True, random_state=5)

    seeds = [
        CSPClassifier(svc, n_components=2, random_state=0)
        .fit(X, y)
        .classifier_.random_state
        for _ in range(2)
    ]
    assert seeds[0] == seeds[1] != 5

    clf = CSPClassifier(svc, n_components=2).fit(X, y)
    assert clf.classifier_.random_state == 5


def test_csp_classifier_without_predict_proba():
    """Test classifiers without predict_proba give one-hot probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPClassifier(RidgeClassifier(), n_components=2).fit(X, y)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert set(np.unique(probs)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(clf.classes_[probs.argmax(axis=1)], preds)


@pytest.mark.parametrize("classifier", ["lda", LinearDiscriminantAnalysis])
def test_csp_classifier_invalid_classifier(classifier):
    """Test a string or an uninstantiated class raises a TypeError."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    with pytest.raises(TypeError, match="must be a scikit-learn classifier"):
        CSPClassifier(classifier, n_components=2).fit(X, y)
