"""Test EEGNet classifier."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification._eegnet import EEGNetClassifier


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
