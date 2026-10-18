"""Test CSP from MNE combined with a scikit-learn SVM or LDA classifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.feature_based import CSPClassifier


@pytest.mark.parametrize("classifier", ["svm", "lda"])
def test_csp_classifier_multivariate_equal_length(classifier):
    """Test CSPClassifier on multivariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPClassifier(classifier=classifier, n_components=2, random_state=0)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("classifier", ["svm", "lda"])
def test_csp_classifier_multiclass(classifier):
    """Test CSPClassifier returns one probability column per class."""
    X, y = make_example_3d_numpy(
        n_cases=30,
        n_channels=4,
        n_timepoints=64,
        n_labels=3,
        random_state=0,
    )

    clf = CSPClassifier(classifier=classifier, n_components=2, random_state=0)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (30, 3)
    assert set(clf.predict(X)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("classifier", ["svm", "lda"])
def test_csp_classifier_reproducible(classifier):
    """Test CSPClassifier gives the same output for a fixed random state."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    probs1 = (
        CSPClassifier(classifier=classifier, n_components=2, random_state=0)
        .fit(X, y)
        .predict_proba(X)
    )
    probs2 = (
        CSPClassifier(classifier=classifier, n_components=2, random_state=0)
        .fit(X, y)
        .predict_proba(X)
    )

    np.testing.assert_allclose(probs1, probs2)


def test_csp_classifier_svm_params():
    """Test SVM defaults and classifier_params reach the fitted estimator."""
    from sklearn.svm import SVC

    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPClassifier(n_components=2, random_state=0).fit(X, y)
    assert isinstance(clf.classifier_, SVC)
    assert clf.classifier_.kernel == "linear"
    assert clf.classifier_.probability

    clf = CSPClassifier(
        n_components=2,
        classifier_params={"C": 0.5, "kernel": "rbf"},
        random_state=0,
    ).fit(X, y)
    assert clf.classifier_.C == 0.5
    assert clf.classifier_.kernel == "rbf"
    assert clf.classifier_.random_state == 0


def test_csp_classifier_lda_params():
    """Test LDA classifier_params reach the fitted estimator."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPClassifier(
        classifier="lda",
        n_components=2,
        classifier_params={"solver": "lsqr", "shrinkage": "auto"},
    )
    clf.fit(X, y)

    assert isinstance(clf.classifier_, LinearDiscriminantAnalysis)
    assert clf.classifier_.solver == "lsqr"
    assert clf.classifier_.shrinkage == "auto"
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0, atol=1e-6)


def test_csp_classifier_invalid_classifier():
    """Test an unknown classifier name raises a ValueError."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    with pytest.raises(ValueError, match="classifier must be one of"):
        CSPClassifier(classifier="knn", n_components=2).fit(X, y)
