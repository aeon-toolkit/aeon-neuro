"""Minimal testing for EEGPT."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._eegpt import EEGPTClassifier


def test_eegpt_classifier_named_channels():
    """Fit/predict with named channels and a series shorter than one patch."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=50)

    clf = EEGPTClassifier(
        ch_names=["C3", "Cz", "C4"], pretrained=False, n_epochs=1, random_state=0
    )
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


def test_eegpt_classifier_old_1020_names():
    """T3/T4/T5/T6 are accepted as their 10-10 names."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=4, n_timepoints=128)
    clf = EEGPTClassifier(
        ch_names=["T3", "t4", "T5", "T6"], pretrained=False, n_epochs=1
    ).fit(X, y)
    assert clf.ch_names_ == ["T7", "T8", "P7", "P8"]
