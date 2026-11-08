"""Check pretrained input requirements without downloading model weights."""

import numpy as np
import pytest

from aeon_neuro.datasets import EEGDatasetInfo


@pytest.mark.parametrize("name", ["CBraModClassifier", "LabramClassifier"])
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("pretrained", [True, False])
def test_prediction_respects_fitted_pretrained_rate(
    name, method, pretrained, monkeypatch
):
    """Model requirements remain known even when no training rate was supplied."""
    from torch import nn

    from aeon_neuro.classification import deep_learning

    class Encoder(nn.Module):
        def forward(self, X, **kwargs):
            if name == "CBraModClassifier":
                return X
            return {"features": X.unflatten(-1, (-1, 200)).flatten(1, 2)}

    def build(self, **kwargs):
        if name == "LabramClassifier":
            self.ch_names_ = list(self.ch_names)
        return Encoder()

    cls = getattr(deep_learning, name)
    monkeypatch.setattr(cls, "_build_network", build)
    options = dict(pretrained=pretrained, random_state=0)
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
