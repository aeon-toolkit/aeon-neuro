"""Test CSP from MNE combined with scikit LDA."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.feature_based import CSPLDAClassifier


def test_csp_lda_classifier_multivariate_equal_length():
    """Test CSPLDAClassifier on multivariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPLDAClassifier(n_components=2)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_csp_lda_classifier_multiclass():
    """Test CSPLDAClassifier returns one probability column per class."""
    X, y = make_example_3d_numpy(
        n_cases=30,
        n_channels=4,
        n_timepoints=64,
        n_labels=3,
        random_state=0,
    )

    clf = CSPLDAClassifier(n_components=2)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (30, 3)
    assert set(clf.predict(X)).issubset(set(np.unique(y)))


def test_csp_lda_classifier_reproducible():
    """Test CSPLDAClassifier gives the same output when fitted twice."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    probs1 = CSPLDAClassifier(n_components=2).fit(X, y).predict_proba(X)
    probs2 = CSPLDAClassifier(n_components=2).fit(X, y).predict_proba(X)

    np.testing.assert_allclose(probs1, probs2)


def test_csp_lda_classifier_shrinkage():
    """Test LDA parameters are passed through to the fitted estimator."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPLDAClassifier(n_components=2, solver="lsqr", shrinkage="auto")
    clf.fit(X, y)

    assert clf.lda_.solver == "lsqr"
    assert clf.lda_.shrinkage == "auto"
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0, atol=1e-6)
