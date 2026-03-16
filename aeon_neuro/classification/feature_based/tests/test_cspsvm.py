"""Test CSP from MNE combined with scikit SVM."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.feature_based import CSPSVMClassifier


def test_csp_svm_classifier_multivariate_equal_length():
    """Test CSPSVMClassifier on multivariate equal-length data."""
    X, y = make_example_3d_numpy(
        n_cases=12,
        n_channels=4,
        n_timepoints=64,
        random_state=0,
    )

    clf = CSPSVMClassifier(
        n_components=2,
        kernel="linear",
        random_state=0,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (12, len(np.unique(y)))
    assert preds.shape == (12,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
