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


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("change", ["rate", "channels"])
def test_array_fitting_layout_is_not_redefined_by_later_constructor_changes(
    method, change
):
    """Prediction metadata must match the layout used to fit the existing model."""
    X, y, _ = data()
    clf = _InfoClassifier(sfreq=32, channel_names=["C3", "C4"]).fit(X, y)
    rate, names = 32, ["C3", "C4"]
    if change == "rate":
        rate = 64
        clf.set_params(sfreq=rate)
    else:
        names = ["C4", "C3"]
        clf.set_params(channel_names=names)
    changed = EEGDatasetInfo.from_arrays(X, sfreq=rate, channel_names=names)
    assert clf.is_fitted
    with pytest.raises(ValueError, match="from fit"):
        getattr(clf, method)(X, info=changed)
    assert not hasattr(clf, "_prediction_info")


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("known_layout", [False, True])
def test_prediction_info_after_array_fitting_retains_known_and_unknown_facts(
    method, known_layout
):
    """Aligned prediction facts remain usable without requiring fitting metadata."""
    X, y, info = data()
    options = dict(sfreq=32, channel_names=["C3", "C4"]) if known_layout else {}
    clf = _InfoClassifier(**options).fit(X, y)
    result = getattr(clf, method)(X, info=info)
    assert len(result) == len(X)
    assert clf.info_ is None
    assert not hasattr(clf, "_prediction_info")
