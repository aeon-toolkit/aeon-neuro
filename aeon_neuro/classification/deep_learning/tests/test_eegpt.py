"""Minimal testing for EEGPT."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._eegpt import EEGPTClassifier


def test_eegpt_classifier_named_channels():
    """Fit/predict with named channels and a series shorter than one patch."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=50)

    clf = EEGPTClassifier(ch_names=["C3", "Cz", "C4"], pretrained=False)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (10, len(np.unique(y)))
    assert clf.predict(X).shape == (10,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_eegpt_classifier_input_errors():
    """Missing names when pretrained, and unknown names, raise."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=128)

    with pytest.raises(ValueError, match="ch_names is required"):
        EEGPTClassifier(pretrained=True).fit(X, y)

    with pytest.raises(ValueError, match="Unknown channel names"):
        EEGPTClassifier(ch_names=["C3", "Cz", "X9"], pretrained=False).fit(X, y)

    with pytest.raises(ValueError, match="has 2 names but X has 3"):
        EEGPTClassifier(ch_names=["C3", "Cz"], pretrained=False).fit(X, y)

    with pytest.raises(ValueError, match="Duplicate channel names"):
        EEGPTClassifier(ch_names=["C3", "Cz", "CZ"], pretrained=False).fit(X, y)


def _fake_checkpoint(tmp_path, monkeypatch, drop_key=None):
    """Save a random encoder as a fake hub checkpoint; return its state dict."""
    import huggingface_hub
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(123)
    state = (
        EEGPTClassifier(ch_names=["C3", "Cz", "C4"], pretrained=False)
        ._build_network(n_channels=3, n_timepoints=128)
        .state_dict()
    )
    state = {k: v.contiguous() for k, v in state.items() if k != "chans_id"}
    state.pop(drop_key, None)
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a: str(path))
    return state


def test_eegpt_pretrained_loading(tmp_path, monkeypatch):
    """Encoder weights come from the checkpoint; missing weights raise."""
    import torch

    state = _fake_checkpoint(tmp_path, monkeypatch)
    net = EEGPTClassifier(ch_names=["C3", "Cz", "C4"])._build_network(3, 128)
    loaded = net.state_dict()
    for k, v in state.items():
        assert torch.equal(loaded[k], v), k

    _fake_checkpoint(tmp_path, monkeypatch, drop_key=next(iter(state)))
    with pytest.raises(RuntimeError, match="missing keys"):
        EEGPTClassifier(ch_names=["C3", "Cz", "C4"])._build_network(3, 128)
