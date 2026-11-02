"""Test EEGConformer classifier."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._eegconformer import (
    EEGConformerClassifier,
)


def test_eeg_conformer_classifier_multivariate_equal_length():
    """Check fit, predict and predict_proba on multivariate data."""
    X, y = make_example_3d_numpy(
        n_cases=10,
        n_channels=3,
        n_timepoints=200,
        random_state=0,
    )

    clf = EEGConformerClassifier(**EEGConformerClassifier._get_test_params())
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (10, len(np.unique(y)))
    assert preds.shape == (10,)
    assert set(preds).issubset(set(clf.classes_))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert len(clf.history_) == 1


def test_eeg_conformer_classifier_reproducible():
    """Check that a fixed random_state gives identical probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=120, random_state=0
    )
    params = EEGConformerClassifier._get_test_params()

    probs1 = EEGConformerClassifier(**params).fit(X, y).predict_proba(X)
    probs2 = EEGConformerClassifier(**params).fit(X, y).predict_proba(X)

    np.testing.assert_allclose(probs1, probs2)


def test_eeg_conformer_classifier_too_short_series():
    """Check that series shorter than the minimum length raise a clear error."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=50, random_state=0
    )

    clf = EEGConformerClassifier(n_epochs=1, num_layers=1, num_heads=2)
    with pytest.raises(ValueError, match="at least 99 timepoints"):
        clf.fit(X, y)


def test_eeg_conformer_classifier_invalid_optimizer():
    """Check that an unknown optimizer name raises a ValueError."""
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=120, random_state=0
    )

    clf = EEGConformerClassifier(
        optimizer="rmsprop", **EEGConformerClassifier._get_test_params()
    )
    with pytest.raises(ValueError, match="Unknown optimizer"):
        clf.fit(X, y)


def test_eeg_conformer_lightweight_test_parameters():
    """Check that aeon receives the intended lightweight test settings."""
    params = EEGConformerClassifier._get_test_params()

    assert params.get("n_epochs") == 1
    assert params.get("batch_size") == 4
    assert params.get("num_layers") == 1


def test_eegconformer_predict_different_length_raises():
    """Test predict raises when series length differs from fit."""
    X, y = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=128, random_state=0
    )
    clf = EEGConformerClassifier(**EEGConformerClassifier._get_test_params())
    clf.fit(X, y)

    X_long, _ = make_example_3d_numpy(
        n_cases=8, n_channels=2, n_timepoints=200, random_state=0
    )
    # aeon skips its own length check when fit metadata is missing (e.g. a
    # loaded model), so the classifier's check must catch the mismatch
    clf.metadata_ = {}
    with pytest.raises(ValueError, match="was fitted on series of length"):
        clf.predict(X_long)
