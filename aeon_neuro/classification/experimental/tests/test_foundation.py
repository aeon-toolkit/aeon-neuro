"""Tests for the fine-tuned foundation model classifiers."""

import os
import pickle

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.estimator_checking import check_estimator

from aeon_neuro.classification.experimental import (
    CBraModFineTuneClassifier,
    LabramFineTuneClassifier,
)

SMALL = {
    "sfreq": 250.0,
    "pretrained": False,
    "n_epochs": 1,
    "validation_size": 0.25,
    "batch_size": 4,
    "random_state": 0,
}


def _data(n_timepoints=250, n_channels=3):
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=n_channels, n_timepoints=n_timepoints, random_state=0
    )
    return X * 50, y


@pytest.mark.parametrize("cls", [CBraModFineTuneClassifier, LabramFineTuneClassifier])
def test_foundation_estimator_checks(cls):
    """Run aeon's estimator checks with random weights."""
    check_estimator(cls, raise_exceptions=True)


@pytest.mark.parametrize(
    "clf",
    [
        CBraModFineTuneClassifier(**SMALL),
        LabramFineTuneClassifier(ch_names=["c3", "Cz", "C4"], **SMALL),
    ],
)
def test_foundation_fit_predict(clf):
    """Data is resampled to 200 Hz, padded to whole patches and classified."""
    X, y = _data()
    clf.fit(X, y)
    # 250 samples at 250 Hz are 200 at 200 Hz: one patch
    assert clf.n_times_ == 200
    proba = clf.predict_proba(X)
    assert proba.shape == (16, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert clf.network_.head.out_features == 2
    # NeuralBench's optimiser settings
    assert clf.optimizer == "adamw"
    assert clf.lr_scheduler == "onecycle"
    assert clf.class_weight == "balanced"
    assert clf.label_smoothing == 0.1
    assert clf.gradient_clip == 1.0


def test_foundation_pads_to_whole_patches():
    """1.5 s at 100 Hz is 300 samples at 200 Hz, padded to two patches."""
    X, y = _data(n_timepoints=150)
    clf = CBraModFineTuneClassifier(**{**SMALL, "sfreq": 100.0}).fit(X, y)
    assert clf.n_times_ == 400
    assert clf._scale_and_pad(clf.preprocessor_.transform(X)).shape == (16, 3, 400)


@pytest.mark.parametrize(
    "units, scale", [("uV", 1e-2), ("mV", 10.0), ("V", 1e4), (0.5, 0.5)]
)
def test_foundation_input_units(units, scale):
    """Input is scaled to the 0.1 mV pretraining units."""
    X, y = _data()
    clf = CBraModFineTuneClassifier(**SMALL, input_units=units).fit(X, y)
    assert clf.unit_scale_ == scale
    preprocessed = clf.preprocessor_.transform(X)
    np.testing.assert_allclose(
        clf._scale_and_pad(preprocessed), preprocessed * scale, rtol=1e-5, atol=1e-6
    )


def test_foundation_validation_split_by_subject():
    """train_groups holds out whole subjects for early stopping."""
    X, y = _data()
    groups = np.repeat(np.arange(4), 4)
    clf = CBraModFineTuneClassifier(**SMALL, train_groups=groups).fit(X, y)
    assert len(clf.validation_groups_) == 1


def test_foundation_random_control_is_seeded():
    """pretrained=False with the same seed gives the same model."""
    X, y = _data()
    a = CBraModFineTuneClassifier(**SMALL).fit(X, y).predict_proba(X)
    b = CBraModFineTuneClassifier(**SMALL).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_foundation_pickle_round_trip():
    """A pickled CBraMod fine-tune predicts the same after unpickling."""
    X, y = _data()
    clf = CBraModFineTuneClassifier(**SMALL).fit(X, y)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X))


def test_foundation_loads_weights_from_directory(tmp_path):
    """weights_path loads a local checkpoint without the network."""
    import torch

    X, y = _data()
    clf = CBraModFineTuneClassifier(**{**SMALL, "n_epochs": 1}).fit(X, y)
    state = {
        k[len("encoder.") :]: v
        for k, v in clf.network_.state_dict().items()
        if k.startswith("encoder.")
    }
    torch.save(state, os.path.join(tmp_path, "pytorch_model.bin"))

    loaded = CBraModFineTuneClassifier(
        **{**SMALL, "pretrained": True, "weights_path": str(tmp_path)}
    )
    loaded.n_classes_ = 2
    encoder = loaded._build_encoder(3, 200)
    loaded._load_weights(encoder)
    for key, value in encoder.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), state[key].numpy())


def test_foundation_missing_weights(tmp_path):
    """An empty weights_path raises."""
    X, y = _data()
    clf = CBraModFineTuneClassifier(
        **{**SMALL, "pretrained": True, "weights_path": str(tmp_path)}
    )
    with pytest.raises(FileNotFoundError, match="weights_path"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"sfreq": None}, "sfreq"),
        ({"input_units": "nV"}, "input_units"),
        ({"input_units": -1.0}, "positive"),
    ],
)
def test_foundation_invalid_parameters(params, match):
    """Invalid parameters raise informative errors."""
    X, y = _data()
    with pytest.raises(ValueError, match=match):
        CBraModFineTuneClassifier(**{**SMALL, **params}).fit(X, y)


@pytest.mark.parametrize(
    "params, n_timepoints, match",
    [
        ({"pretrained": True}, 250, "ch_names is required"),
        ({"ch_names": ["C3", "CZ"]}, 250, "2 names"),
        ({"ch_names": ["C3", "CZ", "XX"]}, 250, "XX"),
        ({"ch_names": ["C3", "CZ", "C4"], "sfreq": 200.0}, 17 * 200, "at most 16"),
    ],
)
def test_labram_invalid_inputs(params, n_timepoints, match):
    """LaBraM needs known channel names and at most 16 patches."""
    X, y = _data(n_timepoints=n_timepoints)
    with pytest.raises(ValueError, match=match):
        LabramFineTuneClassifier(**{**SMALL, **params}).fit(X, y)
