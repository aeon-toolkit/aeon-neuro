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


def test_aeon_estimator_checks():
    """Run aeon's estimator checks on the adaptor."""
    check_estimator(BraindecodeClassifier, raise_exceptions=True)
