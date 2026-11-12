"""Test CSP and LogVariance features combined with a scikit-learn classifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from aeon_neuro.classification.feature_based import CSPClassifier
from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)

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


def test_csp_classifier_composes_csp_and_log_variance():
    """Test features are LogVariance of the CSP-filtered collection."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )
    clf = CSPClassifier(n_components=2).fit(X, y)

    filtered = clf.csp_.transform(X)
    assert filtered.shape == (12, 2, 64)
    np.testing.assert_allclose(clf._csp_features(X), np.log(np.var(filtered, axis=2)))
    np.testing.assert_array_equal(
        clf.predict(X), clf.classifier_.predict(np.log(np.var(filtered, axis=2)))
    )


def test_csp_classifier_log_false():
    """Test log=False passes raw variance features to the classifier."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )
    clf = CSPClassifier(n_components=2, log=False).fit(X, y)

    assert clf.log_variance_.log is False
    np.testing.assert_allclose(
        clf._csp_features(X), np.var(clf.csp_.transform(X), axis=2)
    )


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_csp_classifier_probability_order(classifier):
    """Test predict_proba columns follow classes_, including string labels."""
    X, y = make_example_3d_numpy(
        n_cases=30,
        n_channels=4,
        n_timepoints=64,
        n_labels=3,
        random_state=0,
    )
    y = np.array(["c", "a", "b"])[y.astype(int)]

    clf = CSPClassifier(classifier, n_components=2, random_state=0).fit(X, y)
    probs = clf.predict_proba(X)

    np.testing.assert_array_equal(clf.classes_, ["a", "b", "c"])
    if classifier is None:
        # LDA predict is the argmax of predict_proba, so the columns must align
        np.testing.assert_array_equal(
            clf.classes_[probs.argmax(axis=1)], clf.predict(X)
        )
    features = clf._csp_features(X)
    raw = clf.classifier_.predict_proba(features)
    for i, label in enumerate(clf.classifier_.classes_):
        col = list(clf.classes_).index(label)
        np.testing.assert_allclose(probs[:, col], raw[:, i])


def test_csp_classifier_filters_learned_from_training_data_only():
    """Test filters come from the training data and predict leaves them fixed."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=4, n_timepoints=64, random_state=0
    )
    X_test, _ = make_example_3d_numpy(
        n_cases=8, n_channels=4, n_timepoints=64, random_state=1
    )
    clf = CSPClassifier(n_components=2).fit(X, y)
    filters = clf.csp_.filters_.copy()

    reference = CommonSpatialPatterns(n_components=2).fit(X, y)
    np.testing.assert_array_equal(filters, reference.filters_)

    clf.predict(X_test)
    clf.predict_proba(X_test)
    np.testing.assert_array_equal(clf.csp_.filters_, filters)
    np.testing.assert_allclose(
        clf._csp_features(X_test),
        np.log(np.var(reference.transform(X_test), axis=2)),
    )


def test_csp_classifier_caps_n_components_at_available_filters():
    """Test n_components above the number of filters is capped with a warning."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=3, n_timepoints=64, random_state=0
    )
    clf = CSPClassifier(n_components=6, reg="oas", cov_est="epoch")
    with pytest.warns(UserWarning, match="using n_components=3"):
        clf.fit(X, y)

    assert clf.n_components == 6
    assert clf.n_components_ == 3
    assert clf.csp_.n_components == 3
    assert clf._csp_features(X).shape == (20, 3)
    assert clf.predict_proba(X).shape == (20, 2)

    reference = CSPClassifier(n_components=3, reg="oas", cov_est="epoch").fit(X, y)
    np.testing.assert_array_equal(clf.predict_proba(X), reference.predict_proba(X))


def test_csp_classifier_caps_n_components_at_data_rank():
    """Test average-referenced (rank n_channels - 1) data caps at the rank."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=4, n_timepoints=64, random_state=0
    )
    X = X - X.mean(axis=1, keepdims=True)
    with pytest.warns(UserWarning, match="using n_components=3"):
        clf = CSPClassifier(n_components=4).fit(X, y)
    assert clf.n_components_ == 3


def test_csp_classifier_no_warning_when_n_components_fits():
    """Test no warning and n_components_ == n_components when enough filters."""
    import warnings

    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=4, n_timepoints=64, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clf = CSPClassifier(n_components=4).fit(X, y)
    assert clf.n_components_ == 4


@pytest.mark.parametrize("n_components", [0, -1, 2.0, True])
def test_csp_classifier_invalid_n_components(n_components):
    """Test non-positive or non-int n_components raises."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=4, n_timepoints=64, random_state=0
    )
    with pytest.raises(ValueError, match="n_components"):
        CSPClassifier(n_components=n_components).fit(X, y)
