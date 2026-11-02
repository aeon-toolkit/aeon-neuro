"""Minimal testing for LaBraM."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._labram import LabramClassifier


def test_labram_classifier_named_channels():
    """Fit/predict with named channels and a length that is not a multiple of 200."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=3, n_timepoints=900, random_state=0
    )

    clf = LabramClassifier(ch_names=["C3", "Cz", "C4"], pretrained=False)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (10, len(np.unique(y)))
    assert clf.predict(X).shape == (10,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_labram_classifier_input_errors():
    """Missing names when pretrained, and inputs longer than 16 patches, raise."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=400)

    with pytest.raises(ValueError, match="ch_names is required"):
        LabramClassifier(pretrained=True).fit(X, y)

    X_long, y_long = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=3400)
    with pytest.raises(ValueError, match="at most 16 patches"):
        LabramClassifier(pretrained=False).fit(X_long, y_long)
