"""Tests for the FeatureClassifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from aeon_neuro.classification.feature_based import FeatureClassifier


class _MeanOverTime:
    """Toy transformer: (n_cases, n_channels, n_timepoints) -> (n_cases, n_channels)."""

    def __init__(self):
        self.n_fit = 0

    def fit(self, X, y=None):
        self.n_fit += 1
        return self

    def transform(self, X):
        return np.asarray(X).mean(axis=2)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class _StdOverTime:
    """Toy transformer returning a single std feature per case: (n_cases, 1)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X).std(axis=(1, 2), keepdims=False)[:, None]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class _Identity3D:
    """Toy transformer that returns the 3D input unchanged."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _data(n_cases=12, n_channels=4, n_timepoints=64, n_labels=2):
    return make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_labels=n_labels,
        random_state=0,
    )


@pytest.mark.parametrize("extraction", ["side_by_side", "chained"])
def test_feature_classifier_fit_predict(extraction):
    """Both extraction modes fit and predict with the expected shapes."""
    X, y = _data()
    clf = FeatureClassifier(
        transformers=[_MeanOverTime()], extraction=extraction, random_state=0
    ).fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert preds.shape == (12,)
    assert probs.shape == (12, len(np.unique(y)))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(preds).issubset(set(np.unique(y)))


def test_side_by_side_concatenates_different_shapes():
    """Side-by-side concatenates transformers with differing feature counts."""
    X, y = _data()
    # _MeanOverTime -> (12, 4), _StdOverTime -> (12, 1), _Identity3D -> (12, 4*64)
    clf = FeatureClassifier(
        transformers=[_MeanOverTime(), _StdOverTime(), _Identity3D()],
        extraction="side_by_side",
    ).fit(X, y)

    feats = clf._extract(X, fit=False)
    assert feats.shape == (12, 4 + 1 + 4 * 64)


def test_chained_feeds_output_forward():
    """Chained feeds each output into the next and flattens at the end."""
    X, y = _data()
    clf = FeatureClassifier(
        transformers=[_MeanOverTime(), StandardScaler()],
        extraction="chained",
    ).fit(X, y)

    feats = clf._extract(X, fit=False)
    assert feats.shape == (12, 4)  # mean over time -> (12, 4), scaled in place


def test_single_transformer_not_in_list():
    """A single transformer (not wrapped in a list) is accepted."""
    X, y = _data()
    clf = FeatureClassifier(transformers=_MeanOverTime()).fit(X, y)
    assert clf.predict(X).shape == (12,)


def test_transformers_fitted_once_reused_on_predict():
    """Transformers are fitted during fit and only transformed on predict."""
    X, y = _data()
    t = _MeanOverTime()
    clf = FeatureClassifier(transformers=[t]).fit(X, y)
    clf.predict(X)
    clf.predict(X)

    assert t.n_fit == 1  # fit once in fit(), never in predict()


def test_invalid_extraction_raises():
    """An unknown extraction mode raises a ValueError."""
    X, y = _data()
    with pytest.raises(ValueError, match="extraction must be"):
        FeatureClassifier(transformers=[_MeanOverTime()], extraction="both").fit(X, y)


def test_non_transformer_raises():
    """An object without fit/transform raises a TypeError."""
    X, y = _data()
    with pytest.raises(TypeError, match="not a fit/transform transformer"):
        FeatureClassifier(transformers=[object()]).fit(X, y)


def test_chained_incompatible_shape_raises():
    """A chained stage that rejects the previous output gives a clear error."""
    X, y = _data()
    # StandardScaler cannot consume the raw 3D input directly.
    with pytest.raises(ValueError, match="chained extraction"):
        FeatureClassifier(transformers=[StandardScaler()], extraction="chained").fit(
            X, y
        )


def test_default_classifier_is_lda():
    """The default classifier is LinearDiscriminantAnalysis."""
    X, y = _data()
    clf = FeatureClassifier(transformers=[_MeanOverTime()]).fit(X, y)
    assert isinstance(clf.classifier_, LinearDiscriminantAnalysis)


def test_without_predict_proba_gives_one_hot():
    """A classifier without predict_proba yields one-hot probabilities."""
    X, y = _data()
    clf = FeatureClassifier(
        transformers=[_MeanOverTime()], classifier=RidgeClassifier()
    ).fit(X, y)
    probs = clf.predict_proba(X)

    assert set(np.unique(probs)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(clf.classes_[probs.argmax(axis=1)], clf.predict(X))
