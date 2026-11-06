"""Test the general braindecode classifier adaptor."""

import pickle

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.estimator_checking import check_estimator
from sklearn.base import clone
from torch import nn

from aeon_neuro.classification.deep_learning import BraindecodeClassifier
from aeon_neuro.classification.deep_learning._braindecode import (
    _NEEDS_CHANNEL_NAMES,
    SUPPORTED_MODELS,
)

# Settings that keep the tests quick; all other models use their defaults.
_SMALL_PARAMS = {"EEGConformer": {"num_layers": 1, "num_heads": 2}}
_CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4"]


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_supported_models_fit_predict(model):
    """Check each supported model fits, predicts and survives pickling."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=8, n_timepoints=256, random_state=0
    )
    clf = BraindecodeClassifier(
        model=model,
        model_params=_SMALL_PARAMS.get(model),
        sfreq=128.0,
        channel_names=_CHANNEL_NAMES if model in _NEEDS_CHANNEL_NAMES else None,
        n_epochs=1,
        batch_size=5,
        random_state=0,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (10, len(np.unique(y)))
    assert preds.shape == (10,)
    assert set(preds).issubset(set(clf.classes_))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert len(clf.history_) == 1

    # some layers (e.g. EEGNet's max-norm constraint) use torch
    # parametrizations, which torch cannot pickle directly
    loaded = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(loaded.predict_proba(X), probs, atol=1e-6)


def test_supported_models_exist_in_braindecode():
    """Check every supported model name is a braindecode model."""
    import braindecode.models

    for model in SUPPORTED_MODELS:
        assert issubclass(getattr(braindecode.models, model), nn.Module)


@pytest.mark.parametrize("model", ["NotAModel", "EEGClassifier", 3])
def test_unknown_model_raises(model):
    """Check an unknown name, or something that is not a model, raises."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model=model, n_epochs=1)
    with pytest.raises(ValueError, match="model"):
        clf.fit(X, y)


def test_model_class_accepted():
    """Check a braindecode model class can be passed instead of a name."""
    from braindecode.models import ShallowFBCSPNet

    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model=ShallowFBCSPNet, n_epochs=1, random_state=0)
    clf.fit(X, y)

    assert isinstance(clf.network_, ShallowFBCSPNet)


def test_model_params_passed_and_not_modified():
    """Check model_params reach the network and are left unchanged."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    params = {"n_filters_time": 7}
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet", model_params=params, n_epochs=1, random_state=0
    )
    clf.fit(X, y)

    assert params == {"n_filters_time": 7}
    assert clf.network_.n_filters_time == 7
    assert clone(clf).get_params()["model_params"] == params


@pytest.mark.parametrize("key", ["n_chans", "n_outputs", "n_times", "sfreq"])
def test_reserved_model_params_raise(key):
    """Check model_params cannot override values set from the data."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model_params={key: 1}, n_epochs=1)
    with pytest.raises(ValueError, match="model_params cannot set"):
        clf.fit(X, y)


def test_sfreq_passed_to_network():
    """Check sfreq reaches models that accept it."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=256, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet", sfreq=128.0, n_epochs=1, random_state=0
    )
    clf.fit(X, y)

    assert clf.network_.sfreq == 128.0


def test_channel_names_give_electrode_positions():
    """Check channel_names are turned into chs_info with montage positions."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=3, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        channel_names=["C3", "Cz", "C4"],
        n_epochs=1,
        random_state=0,
    )
    clf.fit(X, y)

    chs_info = clf.network_.chs_info
    assert [ch["ch_name"] for ch in chs_info] == ["C3", "Cz", "C4"]
    assert not np.allclose(chs_info[0]["loc"][:3], 0)


@pytest.mark.parametrize("model", _NEEDS_CHANNEL_NAMES)
def test_missing_channel_names_raise(model):
    """Check models that need electrode positions ask for channel_names."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=3, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model=model, n_epochs=1)
    with pytest.raises(ValueError, match="pass channel_names"):
        clf.fit(X, y)


def test_channel_names_wrong_length_raises():
    """Check channel_names must match the number of channels."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=3, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(channel_names=["C3", "C4"], n_epochs=1)
    with pytest.raises(ValueError, match="channel_names has 2 names"):
        clf.fit(X, y)


def test_unknown_channel_names_raise():
    """Check channel names without standard positions raise a clear error."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet", channel_names=["C3", "NotAChannel"], n_epochs=1
    )
    with pytest.raises(ValueError, match="standard_1005"):
        clf.fit(X, y)


class _PerTimepointNetwork(nn.Module):
    """Toy network giving one output per time point, not one per case."""

    def __init__(self, n_chans, n_outputs, n_times):
        super().__init__()
        self.conv = nn.Conv1d(n_chans, n_outputs, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_non_classifier_output_raises():
    """Check a model whose output is not one score per class raises."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=32, random_state=0
    )
    clf = BraindecodeClassifier(model=_PerTimepointNetwork, n_epochs=1)
    with pytest.raises(ValueError, match="one score per class"):
        clf.fit(X, y)


def test_predict_different_length_raises():
    """Check predict on a different series length raises through aeon."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model="ShallowFBCSPNet", n_epochs=1, random_state=0)
    clf.fit(X, y)

    X_long, _ = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=200, random_state=0
    )
    with pytest.raises(ValueError, match="different length"):
        clf.predict(X_long)


def test_reproducible():
    """Check a fixed random_state gives identical probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=128, random_state=0
    )
    params = {"model": "ShallowFBCSPNet", "n_epochs": 2, "random_state": 0}

    probs1 = BraindecodeClassifier(**params).fit(X, y).predict_proba(X)
    probs2 = BraindecodeClassifier(**params).fit(X, y).predict_proba(X)

    np.testing.assert_allclose(probs1, probs2)


def _record_batch_sizes(coerce_logits, sizes):
    """Wrap _coerce_logits to record the size of each batch it sees."""

    def wrapper(logits):
        sizes.append(logits.shape[0])
        return coerce_logits(logits)

    return wrapper


def test_incomplete_last_batch_is_dropped():
    """Check training drops the incomplete last batch but predicts every case."""
    X, y = make_example_3d_numpy(
        n_cases=9, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet", batch_size=4, n_epochs=2, random_state=0
    )
    batch_sizes = []
    clf._coerce_logits = _record_batch_sizes(clf._coerce_logits, batch_sizes)
    clf.fit(X, y)

    # 9 cases in batches of 4: two full batches per epoch, the last case dropped
    assert batch_sizes == [4, 4, 4, 4]
    assert clf.predict_proba(X).shape == (9, 2)


def test_summary():
    """Check summary returns the training losses, or None before fit."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model="ShallowFBCSPNet", n_epochs=2, random_state=0)

    assert clf.summary() is None

    clf.fit(X, y)
    assert clf.summary() == {"loss": clf.history_}
    assert len(clf.summary()["loss"]) == 2


def test_early_stopping_keeps_best_epoch():
    """Check early stopping stops on patience and restores the best weights."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        n_epochs=30,
        validation_size=0.25,
        early_stopping_patience=2,
        random_state=0,
    )
    clf.fit(X, y)

    n_run = len(clf.history_)
    assert len(clf.validation_history_) == n_run
    assert clf.validation_history_[clf.best_epoch_] == max(clf.validation_history_)
    if n_run < 30:
        # stopped because the last `patience` epochs did not improve
        assert n_run - 1 - clf.best_epoch_ == 2


def test_validation_trains_on_remaining_cases():
    """Check the held out validation cases are not used for training."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        batch_size=64,
        n_epochs=1,
        validation_size=0.25,
        random_state=0,
    )
    batch_sizes = []
    clf._coerce_logits = _record_batch_sizes(clf._coerce_logits, batch_sizes)
    clf.fit(X, y)

    # one training batch of the 15 remaining cases, then the 5 validation cases
    assert batch_sizes == [15, 5]


def test_validation_holds_out_whole_groups():
    """Check train_groups make the validation set whole groups."""
    X, y = make_example_3d_numpy(
        n_cases=24, n_channels=2, n_timepoints=128, random_state=0
    )
    groups = np.repeat(np.arange(8), 3)
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        n_epochs=1,
        validation_size=0.25,
        train_groups=groups,
        random_state=0,
    )
    batch_sizes = []
    clf._coerce_logits = _record_batch_sizes(clf._coerce_logits, batch_sizes)
    clf.fit(X, y)

    # 2 of the 8 groups (6 cases) validate; the other 18 cases train
    assert len(clf.validation_groups_) == 2
    assert batch_sizes == [18, 6]


def test_group_split_is_stratified_when_groups_have_one_class():
    """Check subject-level labels give a validation set with every class."""
    X, _ = make_example_3d_numpy(
        n_cases=32, n_channels=2, n_timepoints=128, random_state=0
    )
    groups = np.repeat(np.arange(8), 4)
    y = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 4)
    for seed in range(5):
        clf = BraindecodeClassifier(
            model="ShallowFBCSPNet",
            n_epochs=1,
            validation_size=0.25,
            train_groups=groups,
            random_state=seed,
        )
        clf.fit(X, y)
        val_labels = {y[groups == g][0] for g in clf.validation_groups_}
        assert val_labels == {0, 1}


def test_single_group_falls_back_to_case_split():
    """Check one group, or no groups, splits by case."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=128, random_state=0
    )
    for groups in (np.zeros(20), None):
        clf = BraindecodeClassifier(
            model="ShallowFBCSPNet",
            n_epochs=1,
            validation_size=0.25,
            train_groups=groups,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.validation_groups_ is None
        assert len(clf.validation_history_) == 1


def test_train_groups_wrong_length_raises():
    """Check train_groups must give one group per training case."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        n_epochs=1,
        validation_size=0.25,
        train_groups=np.arange(10),
    )
    with pytest.raises(ValueError, match="train_groups has 10 entries"):
        clf.fit(X, y)


def test_balanced_class_weights():
    """Check balanced weights are n_cases / (n_classes * class count)."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=128, random_state=0
    )
    y = np.array([0] * 9 + [1] * 3)
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        n_epochs=1,
        class_weight="balanced",
        label_smoothing=0.1,
        random_state=0,
    )
    clf.fit(X, y)

    np.testing.assert_allclose(
        clf.loss_.weight.cpu().numpy(), [12 / (2 * 9), 12 / (2 * 3)], rtol=1e-6
    )
    assert clf.loss_.label_smoothing == 0.1


def test_onecycle_schedule_and_adamw():
    """Check AdamW and the one-cycle schedule, stepped once per batch."""
    import torch

    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(
        model="ShallowFBCSPNet",
        batch_size=4,
        n_epochs=3,
        lr=1e-3,
        optimizer="adamw",
        lr_scheduler="onecycle",
        gradient_clip=1.0,
        random_state=0,
    )
    clf.fit(X, y)

    assert isinstance(clf.optimizer_, torch.optim.AdamW)
    assert clf.scheduler_.total_steps == 3 * 4
    assert clf.scheduler_.last_epoch == 3 * 4
    # the clone and the pickled classifier keep working
    assert clone(clf).get_params()["lr_scheduler"] == "onecycle"
    loaded = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))


@pytest.mark.parametrize(
    "params, match",
    [
        ({"validation_size": 1.5}, "validation_size"),
        ({"class_weight": "inverse"}, "class_weight"),
        ({"lr_scheduler": "cosine"}, "lr_scheduler"),
    ],
)
def test_invalid_training_options_raise(params, match):
    """Check unknown training options raise a ValueError."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = BraindecodeClassifier(model="ShallowFBCSPNet", n_epochs=1, **params)
    with pytest.raises(ValueError, match=match):
        clf.fit(X, y)


def test_aeon_estimator_checks():
    """Run aeon's estimator checks on the adaptor."""
    check_estimator(BraindecodeClassifier, raise_exceptions=True)
