"""Test EEGNet classifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._eegnet import EEGNetClassifier


def test_eeg_net_classifier_multivariate_equal_length():
    """Basic EEGNet test placeholder."""
    X, y = make_example_3d_numpy(
        n_cases=10,
        n_channels=3,
        n_timepoints=256,
        random_state=0,
    )

    clf = EEGNetClassifier(
        n_epochs=1,
        batch_size=4,
        random_state=0,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (10, len(np.unique(y)))
    assert preds.shape == (10,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_eegnet_predict_different_length_raises():
    """Test predict raises when series length differs from fit."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = EEGNetClassifier(n_epochs=1, batch_size=4, random_state=0)
    clf.fit(X, y)

    X_long, _ = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=200, random_state=0
    )
    # aeon skips its own length check when fit metadata is missing (e.g. a
    # loaded model), so the classifier's check must catch the mismatch
    clf.metadata_ = {}
    with pytest.raises(ValueError, match="was fitted on series of length"):
        clf.predict(X_long)


def test_eegnet_lightweight_test_parameters():
    """Check that aeon receives the intended lightweight test settings."""
    params = EEGNetClassifier._get_test_params()

    assert params.get("n_epochs") == 1
    assert params.get("batch_size") == 4
