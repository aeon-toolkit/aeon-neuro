"""Check pretrained input requirements without downloading model weights."""

import numpy as np
import pytest

from aeon_neuro.datasets import EEGDatasetInfo


@pytest.mark.parametrize("name", ["CBraModClassifier", "LabramClassifier"])
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("finetune", [False, True])
def test_prediction_respects_fitted_pretrained_rate(
    name, method, pretrained, finetune, monkeypatch
):
    """Model requirements remain known even when no training rate was supplied."""
    from torch import nn

    from aeon_neuro.classification import deep_learning

    def build(self, n_channels, n_timepoints):
        return nn.Sequential(nn.Flatten(), nn.Linear(n_channels * n_timepoints, 2))

    cls = getattr(deep_learning, name)
    monkeypatch.setattr(cls, "_build_network", build)
    monkeypatch.setattr(cls, "_load_checkpoint", lambda self, network: None)
    monkeypatch.setattr(cls, "_features", lambda self, X: X.mean(axis=-1))
    options = dict(pretrained=pretrained, finetune=finetune, n_epochs=1, random_state=0)
    if name == "LabramClassifier":
        options["ch_names"] = ["C3", "C4"]
    X = np.random.default_rng(1).normal(size=(12, 2, 200))
    y = np.tile([0, 1], 6)
    info = EEGDatasetInfo.from_arrays(X, sfreq=128, channel_names=["C3", "C4"])
    if pretrained:
        with pytest.raises(ValueError, match="200 Hz"):
            cls(**options).fit(X, y, info=info)
    classifier = cls(**options).fit(X, y)
    assert classifier.sfreq_ is None
    # Later parameter changes must not redefine the network fitted above.
    classifier.set_params(pretrained=not pretrained)
    if pretrained:
        with pytest.raises(ValueError, match="200 Hz"):
            getattr(classifier, method)(X, info=info)
    else:
        assert len(getattr(classifier, method)(X, info=info)) == len(X)
    valid = EEGDatasetInfo.from_arrays(X, sfreq=200, channel_names=["C3", "C4"])
    assert len(getattr(classifier, method)(X, info=valid)) == len(X)
    assert len(getattr(classifier, method)(X)) == len(X)
    assert not hasattr(classifier, "_prediction_info")


def test_labram_uses_fitted_metadata_channels_through_pickle(monkeypatch):
    """Use fitted channel names when constructing and restoring the network."""
    import pickle

    from aeon_neuro.classification.deep_learning import LabramClassifier

    X = np.random.default_rng(1).normal(size=(6, 2, 200))
    y = np.tile([0, 1], 3)
    info = EEGDatasetInfo.from_arrays(X, sfreq=200, channel_names=["C3", "C4"])
    monkeypatch.setattr(
        LabramClassifier, "_load_checkpoint", lambda self, network: None
    )
    classifier = LabramClassifier(pretrained=True, random_state=0).fit(X, y, info=info)
    assert classifier.ch_names_ == ["C3", "C4"]
    probabilities = classifier.predict_proba(X, info=info)

    classifier.set_params(ch_names=["C4", "C3"])
    restored = pickle.loads(pickle.dumps(classifier))
    assert restored.ch_names_ == ["C3", "C4"]
    np.testing.assert_array_equal(restored.predict_proba(X, info=info), probabilities)
