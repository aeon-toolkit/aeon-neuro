"""Exercise shared EEG information through the merged Braindecode adapter."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

from aeon_neuro.classification.deep_learning import BraindecodeClassifier
from aeon_neuro.datasets import EEGDatasetInfo


def _data():
    X = np.random.default_rng(5).normal(size=(12, 2, 128))
    y = np.tile([0, 1], 6)
    info = EEGDatasetInfo.from_arrays(
        X,
        sfreq=128,
        channel_names=["C4", "C3"],
        units=["V", "V"],
        subjects=[f"person-{i // 2}" for i in range(12)],
    )
    return X, y, info


def test_adapter_uses_current_information_and_preserves_it_through_pickle():
    """Network inputs and restored predictions follow data, not constructor edits."""
    X, y, info = _data()
    X, info = info.process(X, "resample", sfreq=64)
    clf = BraindecodeClassifier(n_epochs=1, batch_size=4, random_state=1, device="cpu")
    params = clf.get_params(deep=False)
    clf.fit(X[:8], y[:8], info=info[:8])
    assert clf.network_.sfreq == 64
    assert [row["ch_name"] for row in clf.network_.chs_info] == ["C4", "C3"]
    assert clf.channel_geometry_ == "names_only"
    assert all(np.isnan(row["loc"][:3]).all() for row in clf.network_.chs_info)
    assert clf.get_params(deep=False) == params == clone(clf).get_params(deep=False)
    probabilities = clf.predict_proba(X[8:], info=info[8:])
    loaded = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(
        loaded.predict_proba(X[8:], info=info[8:]), probabilities, atol=1e-6
    )
    assert loaded.info_.to_dict() == info[:8].to_dict()


def test_nonstandard_metadata_names_do_not_require_template_positions():
    """Ordinary models can retain source names without inventing a montage."""
    X, y, _ = _data()
    info = EEGDatasetInfo.from_arrays(
        X, sfreq=128, channel_names=["sensor-a", "sensor-b"]
    )
    clf = BraindecodeClassifier(n_epochs=1, random_state=1, device="cpu").fit(
        X, y, info=info
    )
    assert [row["ch_name"] for row in clf.network_.chs_info] == ["sensor-a", "sensor-b"]
    assert clf.channel_geometry_ == "names_only"
    assert clf.predict(X, info=info).shape == (12,)


@pytest.mark.parametrize("as_class", [False, True])
def test_spatial_models_use_declared_template_policy_with_information(as_class):
    """Required graph coordinates use template geometry, not measurements."""
    from braindecode.models import DGCNN

    names = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4"]
    X = np.random.default_rng(2).normal(size=(6, 8, 128))
    info = EEGDatasetInfo.from_arrays(X, sfreq=128, channel_names=names)
    before = info.to_dict()
    clf = BraindecodeClassifier(
        model=DGCNN if as_class else "DGCNN", n_epochs=1, random_state=1, device="cpu"
    ).fit(X, np.tile([0, 1], 3), info=info)
    assert clf.channel_geometry_ == "standard_1005"
    assert not np.allclose(clf.network_.chs_info[0]["loc"][:3], 0)
    assert info.to_dict() == before
    assert info.recordings == ()


@pytest.mark.parametrize("conflict", [{"sfreq": 250}, {"channel_names": ["C3", "C4"]}])
def test_adapter_rejects_explicit_layout_conflicts_before_building(
    conflict, monkeypatch
):
    """Equal array shapes cannot excuse an inconsistent physical layout."""
    X, y, info = _data()

    def unexpected(*args, **kwargs):
        raise AssertionError("Network built before checking the information")

    monkeypatch.setattr(BraindecodeClassifier, "_build_network", unexpected)
    with pytest.raises(ValueError, match="disagrees"):
        BraindecodeClassifier(**conflict).fit(X, y, info=info)


def test_adapter_validates_by_subject_and_retains_inference_and_pickle():
    """The shared loop keeps held-out subjects out of all optimiser batches."""
    X, y, info = _data()
    clf = BraindecodeClassifier(
        n_epochs=2,
        batch_size=4,
        random_state=1,
        device="cpu",
        validation_size=0.34,
        patience=1,
    ).fit(X[:10], y[:10], info=info[:10])
    subjects = np.asarray(info[:10].participant_ids)
    assert set(subjects[clf.training_indices_]).isdisjoint(
        set(subjects[clf.validation_indices_])
    )
    assert set(clf.training_indices_) | set(clf.validation_indices_) == set(range(10))
    assert clf.validation_policy_ == "subject"
    assert not clf.network_.training
    loaded = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(
        loaded.predict_proba(X[10:], info=info[10:]),
        clf.predict_proba(X[10:], info=info[10:]),
        atol=1e-6,
    )


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_adapter_checks_prediction_information(method):
    """A fitted network cannot silently accept reordered channel information."""
    X, y, info = _data()
    clf = BraindecodeClassifier(n_epochs=1, random_state=1, device="cpu").fit(
        X, y, info=info
    )
    wrong = EEGDatasetInfo.from_arrays(X, sfreq=128, channel_names=["C3", "C4"])
    with pytest.raises(ValueError, match="channel names/order"):
        getattr(clf, method)(X, info=wrong)
    assert not hasattr(clf, "_prediction_info")
