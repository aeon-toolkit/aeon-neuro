"""Minimal testing for CBraMod."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._cbramod import CBraModClassifier


def test_cbramod_classifier_multivariate_equal_length():
    """Fit/predict with a length that is not a multiple of patch_size."""
    X, y = make_example_3d_numpy(
        n_cases=10,
        n_channels=3,
        n_timepoints=250,
        random_state=0,
    )

    clf = CBraModClassifier(**CBraModClassifier._get_test_params())
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (10, len(np.unique(y)))
    assert preds.shape == (10,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
