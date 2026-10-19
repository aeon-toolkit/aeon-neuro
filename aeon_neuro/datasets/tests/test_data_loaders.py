"""Test data loading with shipped data."""

import os

import pytest

from aeon_neuro.datasets._data_loaders import load_eeg_classification
from aeon_neuro.datasets.classification_datasets import dataset_map, dataset_names


def test_load_eeg():
    """Test data loading from provided datasets."""
    X, y = load_eeg_classification("SelfRegulationSCP1")
    assert X.shape == (561, 6, 896)
    X, y, meta = load_eeg_classification("SelfRegulationSCP1", return_metadata=True)
    assert meta["problemname"] == "selfregulationscp1"


@pytest.mark.skipif(
    not os.environ.get("AEON_NEURO_DOWNLOAD_TESTS"),
    reason="Downloads every dataset on Zenodo; set AEON_NEURO_DOWNLOAD_TESTS=1 to run",
)
def test_load_eeg_for_all_datasets(tmp_path):
    """Test data loading for all datasets that are available on Zenodo."""
    for name, zenodo_id in dataset_map.items():
        if zenodo_id != 49:
            X, y = load_eeg_classification(name, extract_path=str(tmp_path))
            assert X.ndim == 3
            assert y.ndim == 1


def test_dataset_map_matches_names():
    """Test every listed dataset has an entry in the Zenodo map."""
    assert set(dataset_map) == set(dataset_names)


def test_load_eeg_unavailable(tmp_path):
    """Test loading a dataset that is not yet on Zenodo raises a clear error."""
    name = next(n for n, i in dataset_map.items() if i == 49)
    with pytest.raises(ValueError, match="not yet available"):
        load_eeg_classification(name, extract_path=str(tmp_path))


def test_load_eeg_invalid_name(tmp_path):
    """Test loading an unknown dataset raises an error."""
    with pytest.raises(ValueError, match="not in the list of valid files"):
        load_eeg_classification("NotADataset", extract_path=str(tmp_path))


def test_load_eeg_from_zenodo(tmp_path):
    """Test downloading a small dataset from Zenodo."""
    X, y = load_eeg_classification("Sleep", extract_path=str(tmp_path))  # ~6 MB
    assert X.shape == (3599, 1, 178)
