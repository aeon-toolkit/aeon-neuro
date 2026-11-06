"""Preserve aeon's lifecycle while validating optional case information."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from aeon_neuro.classification.base import BaseEEGClassifier
from aeon_neuro.datasets import EEGDatasetInfo


class _InfoClassifier(BaseEEGClassifier):
    """Small real consumer used to observe information inside protected methods."""

    _tags = {"capability:multivariate": True, "X_inner_type": "numpy3D"}

    def __init__(self, sfreq=None, channel_names=None, fail=False):
        self.sfreq = sfreq
        self.channel_names = channel_names
        self.fail = fail
        super().__init__()

    def _fit(self, X, y):
        self.seen_info_ = self.info_.to_dict() if self.info_ is not None else None
        if self.fail:
            raise ValueError("fit failed")
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def _predict_proba(self, X):
        self.last_prediction_ids_ = (
            self._prediction_info.case_ids
            if self._prediction_info is not None
            else None
        )
        if self.fail:
            raise ValueError("prediction failed")
        return np.tile(self.prior_, (len(X), 1))

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]


def data():
    """Each subject has both classes, so the grouped split is feasible."""
    X = np.random.default_rng(42).normal(size=(12, 2, 64))
    y = np.tile([0, 1], 6)
    info = EEGDatasetInfo.from_arrays(
        X,
        sfreq=32,
        channel_names=["C3", "C4"],
        units=["V", "V"],
        subjects=[f"subject-{i//2}" for i in range(12)],
        case_ids=[f"case-{i}" for i in range(12)],
    )
    return X, y, info


def test_info_survives_aeon_reset_and_is_detached_from_inputs():
    """Protected fit sees validated information after the parent lifecycle reset."""
    X, y, info = data()
    clf = _InfoClassifier(sfreq=32, channel_names=["C3", "C4"])
    params = clf.get_params(deep=False)
    clf.fit(X, y, info=info)
    assert clf.is_fitted and clf.seen_info_ == info.to_dict()
    assert clf.info_ is not info
    assert clf.sfreq_ == 32 and clf.channel_names_ == ("C3", "C4")
    assert clf.get_params(deep=False) == params
    assert clone(clf).get_params(deep=False) == params
    assert not hasattr(clf, "_pending_info")
    clf.fit(X, y)
    assert clf.info_ is None and clf.subject_ids_ is None
    assert clf.seen_info_ is None


@pytest.mark.parametrize("fault", ["rate", "names", "count", "units"])
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_prediction_rejects_rate_order_count_and_unit_mismatches(fault, method):
    """Equal-shaped inputs are insufficient when physical layout disagrees."""
    X, y, info = data()
    clf = _InfoClassifier().fit(X, y, info=info)
    options = dict(sfreq=32, channel_names=["C3", "C4"], units=["V", "V"])
    if fault == "rate":
        options["sfreq"] = 64
    elif fault == "names":
        options["channel_names"] = ["C4", "C3"]
    elif fault == "units":
        options["units"] = ["uV", "uV"]
    changed = EEGDatasetInfo.from_arrays(X[:-1] if fault == "count" else X, **options)
    with pytest.raises(ValueError):
        getattr(clf, method)(X, info=changed)
    assert not hasattr(clf, "_prediction_info")


@pytest.mark.parametrize("options", [{"sfreq": 64}, {"channel_names": ["C4", "C3"]}])
def test_explicit_constructor_conflicts_are_rejected_without_mutation(options):
    """Metadata never silently replaces an explicitly supplied configuration."""
    X, y, info = data()
    clf = _InfoClassifier(**options)
    with pytest.raises(ValueError, match="disagrees"):
        clf.fit(X, y, info=info)
    assert clf.get_params(deep=False) == _InfoClassifier(**options).get_params(
        deep=False
    )
    assert not clf.is_fitted


def test_prediction_uses_call_specific_info_and_cleans_up_after_failure():
    """Prediction subjects remain distinct from the retained fitting subjects."""
    X, y, info = data()
    clf = _InfoClassifier().fit(X[:8], y[:8], info=info[:8])
    clf.predict_proba(X[8:], info=info[8:])
    assert clf.last_prediction_ids_ == info[8:].case_ids
    assert clf.info_.case_ids == info[:8].case_ids
    clf.fail = True
    with pytest.raises(ValueError, match="prediction failed"):
        clf.predict_proba(X[8:], info=info[8:])
    assert not hasattr(clf, "_prediction_info")


def test_no_info_pipeline_cv_and_fit_estimates_keep_working():
    """The optional extension preserves ordinary sklearn and aeon entrypoints."""
    X, y, _ = data()
    pipeline = Pipeline([("clf", _InfoClassifier())])
    assert cross_val_score(pipeline, X, y, cv=3).shape == (3,)
    clf = _InfoClassifier().fit(X, y)
    assert clf.predict(X).shape == (12,)
    assert clf.predict_proba(X).shape == (12, 2)
    assert _InfoClassifier().fit_predict(X, y, cv_size=2).shape == (12,)


def test_validation_split_uses_subjects_only_within_current_fit_scope():
    """Inner validation cannot leak subjects or introduce outer held-out cases."""
    X, y, info = data()
    fit = np.arange(10)
    clf = _InfoClassifier().fit(X[fit], y[fit], info=info[fit])
    train, val = clf.validation_split(y[fit], validation_size=0.4, random_state=1)
    assert not {info[fit].participant_ids[i] for i in train} & {
        info[fit].participant_ids[i] for i in val
    }
    assert set(train) | set(val) == set(range(10))
    assert set(clf.subject_ids_).isdisjoint(set(info[10:].participant_ids))
    ordinary = _InfoClassifier().fit(X, y)
    train, val = ordinary.validation_split(y, random_state=1)
    assert set(y[train]) == set(y[val]) == {0, 1}


def test_impossible_subject_validation_does_not_fall_back_to_case_splitting():
    """Missing identities and class-confounded groups require an explicit decision."""
    X, y, _ = data()
    for subjects in ([None] * 12, ["one"] * 12, [str(v) for v in y]):
        info = EEGDatasetInfo.from_arrays(
            X, sfreq=32, channel_names=["C3", "C4"], subjects=subjects
        )
        clf = _InfoClassifier().fit(X, y, info=info)
        with pytest.raises(ValueError):
            clf.validation_split(y, random_state=1)


@pytest.mark.parametrize("name", ["CBraModClassifier", "LabramClassifier"])
def test_pretrained_encoder_rate_mismatch_fails_before_loading_weights(
    name, monkeypatch
):
    """Known pretrained input rates are enforced before any remote model access."""
    from aeon_neuro.classification.deep_learning import _cbramod, _labram

    cls = getattr(_cbramod if name == "CBraModClassifier" else _labram, name)

    def unexpected(*args, **kwargs):
        raise AssertionError("Model weights were opened before checking sampling rate")

    monkeypatch.setattr(cls, "_build_network", unexpected)
    X, y, info = data()
    with pytest.raises(ValueError, match="200 Hz"):
        cls(pretrained=True).fit(X, y, info=info)


def test_within_subject_validation_is_an_explicit_choice_with_information():
    """A within-person protocol can retain physical metadata without guessing."""
    X, y, _ = data()
    info = EEGDatasetInfo.from_arrays(
        X, sfreq=32, channel_names=["C3", "C4"], subjects=["one"] * 12
    )
    clf = _InfoClassifier().fit(X, y, info=info)
    with pytest.raises(ValueError, match="two known subjects"):
        clf.validation_split(y)
    train, val = clf.validation_split(y, strategy="stratified", random_state=1)
    assert set(y[train]) == set(y[val]) == {0, 1}
    assert clf.info_.participant_ids == ("one",) * 12
    with pytest.raises(ValueError, match="strategy"):
        clf.validation_split(y, strategy="infer")
    with pytest.raises(ValueError, match="requires"):
        _InfoClassifier().fit(X, y).validation_split(y, strategy="subject")
