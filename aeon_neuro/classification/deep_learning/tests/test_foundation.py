"""Tests for the training options of _BaseFoundationClassifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._cbramod import CBraModClassifier
from aeon_neuro.classification.deep_learning._eegpt import EEGPTClassifier
from aeon_neuro.classification.deep_learning._labram import LabramClassifier

PARAMS = {**CBraModClassifier._get_test_params(), "n_layer": 2}


def _initial_state(clf, X, y):
    """Return the network weights before training, by fitting zero epochs."""
    import copy

    clf = copy.deepcopy(clf).set_params(n_epochs=0)
    clf.fit(X, y)
    return {k: v.clone() for k, v in clf.network_.state_dict().items()}


def test_freeze_encoder_trains_only_the_head():
    """With freeze_encoder=True, linear probing leaves the encoder unchanged."""
    X, y = make_example_3d_numpy(n_cases=8, n_channels=3, n_timepoints=200)
    clf = CBraModClassifier(**{**PARAMS, "n_epochs": 2, "freeze_encoder": True})
    before = _initial_state(clf, X, y)
    clf.fit(X, y)

    after = clf.network_.state_dict()
    changed = {k for k in before if not np.allclose(before[k], after[k])}
    assert changed and all(k.startswith("final_layer.") for k in changed)
    assert len(clf.history_) == 2
    assert clf.predict_proba(X).shape == (8, 2)


def test_fine_tuning_trains_the_encoder():
    """By default the whole network, encoder included, is trained."""
    X, y = make_example_3d_numpy(n_cases=8, n_channels=3, n_timepoints=200)
    clf = CBraModClassifier(**{**PARAMS, "n_epochs": 1})
    before = _initial_state(clf, X, y)
    clf.fit(X, y)

    after = clf.network_.state_dict()
    assert len(clf.history_) == 1
    assert not np.allclose(
        before["encoder.layers.0.linear1.weight"],
        after["encoder.layers.0.linear1.weight"],
    )
    assert all(p.requires_grad for p in clf.network_.parameters())


def test_param_groups_layer_decay_and_head_lr():
    """Learning rates decay towards the input; the head uses head_lr."""
    X, y = make_example_3d_numpy(n_cases=4, n_channels=3, n_timepoints=200)
    clf = CBraModClassifier(**{**PARAMS, "n_epochs": 0})
    clf.fit(X, y)
    clf.set_params(lr=1.0, layer_decay=0.5, head_lr=0.1)

    lr_of = {
        id(p): group["lr"]
        for group in clf._param_groups(head_only=False)
        for p in group["params"]
    }
    lr = {name: lr_of[id(p)] for name, p in clf.network_.named_parameters()}
    assert lr["patch_embedding.proj_in.0.weight"] == 0.125
    assert lr["encoder.layers.0.linear1.weight"] == 0.25
    assert lr["encoder.layers.1.linear1.weight"] == 0.5
    assert lr["proj_out.0.weight"] == 1.0
    assert lr["final_layer.1.weight"] == 0.1


def test_cosine_schedule():
    """Linear warmup to the full rate, then cosine decay to zero."""
    factor = CBraModClassifier(lr_schedule="cosine", warmup_epochs=1)._lr_factor(
        n_steps=10, steps_per_epoch=2
    )
    assert [factor(0), factor(1)] == [0.5, 1.0]
    assert factor(2) == 1.0
    assert factor(6) == pytest.approx(0.5)
    assert factor(10) == pytest.approx(0.0)


def test_validation_data_keeps_best_epoch():
    """With validation data, the weights of the best epoch are kept."""
    from sklearn.metrics import balanced_accuracy_score

    X, y = make_example_3d_numpy(n_cases=20, n_channels=3, n_timepoints=200)
    X_val, y_val = make_example_3d_numpy(
        n_cases=20, n_channels=3, n_timepoints=200, random_state=1
    )
    clf = CBraModClassifier(
        **{**PARAMS, "n_epochs": 6, "lr": 1e-2},
        validation_data=(X_val, y_val),
    )
    clf.fit(X, y)

    assert len(clf.validation_scores_) == 6
    assert clf.best_epoch_ == 1 + int(np.argmax(clf.validation_scores_))
    assert balanced_accuracy_score(y_val, clf.predict(X_val)) == pytest.approx(
        max(clf.validation_scores_)
    )


CASES = {
    "CBraMod": (CBraModClassifier, {}, 200),
    "LaBraM": (LabramClassifier, {"ch_names": ["C3", "Cz", "C4"]}, 200),
    "EEGPT": (EEGPTClassifier, {"ch_names": ["C3", "Cz", "C4"]}, 128),
}


def _fake_checkpoint(clf, n_timepoints, tmp_path, monkeypatch, drop_key=False):
    """Serve random weights shaped like clf's checkpoint, without downloading."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(123)
    clf.n_classes_ = 2
    state = {
        k: v.contiguous()
        for k, v in clf._build_network(3, n_timepoints).state_dict().items()
        if k.startswith(clf._checkpoint_prefix) and not k.startswith("final_layer.")
    }
    if drop_key:
        state.pop(next(iter(state)))
    path = tmp_path / "model.safetensors"
    save_file(state, path)

    def download(repo_id, filename):
        assert (repo_id, filename) == (clf._checkpoint, "model.safetensors")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return state


@pytest.mark.parametrize("name", CASES)
def test_pretrained_loading(name, tmp_path, monkeypatch):
    """The checkpoint is loaded into all but the head; a missing key raises."""
    cls, params, n_timepoints = CASES[name]
    X, y = make_example_3d_numpy(n_cases=4, n_channels=3, n_timepoints=n_timepoints)
    clf = cls(**params, pretrained=True, n_epochs=0, random_state=0)
    state = _fake_checkpoint(clf, n_timepoints, tmp_path, monkeypatch)

    clf.fit(X, y)
    loaded = clf.network_.state_dict()
    for key, value in state.items():
        np.testing.assert_array_equal(loaded[key], value)

    _fake_checkpoint(clf, n_timepoints, tmp_path, monkeypatch, drop_key=True)
    with pytest.raises(RuntimeError, match="missing keys"):
        clf.fit(X, y)


@pytest.mark.parametrize("name", CASES)
def test_unpickling_does_not_load_the_checkpoint(name, tmp_path, monkeypatch):
    """A fitted pretrained classifier is restored from its own weights only."""
    import pickle

    cls, params, n_timepoints = CASES[name]
    X, y = make_example_3d_numpy(n_cases=4, n_channels=3, n_timepoints=n_timepoints)
    clf = cls(**params, pretrained=True, n_epochs=1, random_state=0)
    _fake_checkpoint(clf, n_timepoints, tmp_path, monkeypatch)
    clf.fit(X, y)
    data = pickle.dumps(clf)

    def no_download(*args, **kwargs):
        raise AssertionError("the checkpoint was downloaded")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    restored = pickle.loads(data)
    np.testing.assert_array_equal(restored.predict_proba(X), clf.predict_proba(X))


@pytest.mark.parametrize("name", CASES)
def test_estimator_on_frozen_embeddings(name):
    """With estimator, it is fitted on the embeddings and the network is unchanged."""
    from sklearn.linear_model import LogisticRegression

    cls, params, n_timepoints = CASES[name]
    X, y = make_example_3d_numpy(n_cases=8, n_channels=3, n_timepoints=n_timepoints)
    clf = cls(**params, pretrained=False, random_state=0)
    before = _initial_state(clf, X, y)
    clf.set_params(estimator=LogisticRegression()).fit(X, y)

    after = clf.network_.state_dict()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert clf.history_ == []
    features = clf._features(clf._pad(X))
    assert features.shape[0] == 8
    assert clf.estimator_.n_features_in_ == features.shape[1]
    assert clf.predict_proba(X).shape == (8, 2)
    assert clf.predict(X).shape == (8,)
