"""Test data loading with shipped data."""

from aeon_neuro.datasets._data_loaders import load_eeg_classification


def test_load_eeg():
    """Test data loading from provided datasets."""
    X, y = load_eeg_classification("SelfRegulationSCP1")
    assert X.shape == (561, 6, 896)
    X, y, meta = load_eeg_classification("SelfRegulationSCP1", return_metadata=True)
    assert meta["problemname"] == "selfregulationscp1"
