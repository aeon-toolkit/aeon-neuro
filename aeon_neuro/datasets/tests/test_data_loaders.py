"""Test local EEG data loading and the optional Zenodo downloads."""

import os
from unittest.mock import Mock

import numpy as np
import pytest

from aeon_neuro.datasets import _data_loaders
from aeon_neuro.datasets._data_loaders import load_eeg_classification
from aeon_neuro.datasets.classification_datasets import dataset_map, dataset_names

_requires_zenodo = pytest.mark.skipif(
    not os.environ.get("AEON_NEURO_DOWNLOAD_TESTS"),
    reason="Requires Zenodo access. Set AEON_NEURO_DOWNLOAD_TESTS=1 to run.",
)


def test_load_eeg():
    """Test data loading from provided datasets."""
    X, y = load_eeg_classification("SelfRegulationSCP1")
    assert X.shape == (561, 6, 896)
    X, y, meta = load_eeg_classification("SelfRegulationSCP1", return_metadata=True)
    assert meta["problemname"] == "selfregulationscp1"


@_requires_zenodo
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


def test_load_eeg_uses_cached_files(tmp_path, monkeypatch):
    """Load both cached splits by default and support selecting just TEST."""
    cache = tmp_path / "Sleep"
    cache.mkdir()
    header = (
        "@problemName Sleep\n@timestamps false\n@univariate true\n"
        "@equalLength true\n@seriesLength 3\n@classLabel true awake asleep\n@data\n"
    )
    # Cache discovery requires both split files, even when loading just TEST.
    (cache / "Sleep_TRAIN.ts").write_text(header + "1,2,3:awake\n", encoding="utf-8")
    (cache / "Sleep_TEST.ts").write_text(header + "4,5,6:asleep\n", encoding="utf-8")
    download = Mock(side_effect=AssertionError("Cache hit attempted a download"))
    monkeypatch.setattr(_data_loaders, "load_classification", download)

    X, y = load_eeg_classification("Sleep", extract_path=str(tmp_path))

    np.testing.assert_array_equal(X, [[[1, 2, 3]], [[4, 5, 6]]])
    np.testing.assert_array_equal(y, ["awake", "asleep"])

    X, y, meta = load_eeg_classification(
        "Sleep", split="test", extract_path=str(tmp_path), return_metadata=True
    )

    np.testing.assert_array_equal(X, [[[4, 5, 6]]])
    np.testing.assert_array_equal(y, ["asleep"])
    assert meta["problemname"] == "sleep"
    download.assert_not_called()


def test_load_eeg_download_fallback(tmp_path, monkeypatch):
    """Forward the requested split and metadata options on a cache miss."""
    expected = (np.array([[[4, 5, 6]]]), np.array(["asleep"]), {"problemname": "sleep"})
    download = Mock(return_value=expected)
    monkeypatch.setattr(_data_loaders, "load_classification", download)

    X, y, meta = load_eeg_classification(
        "Sleep", split="test", extract_path=str(tmp_path), return_metadata=True
    )

    download.assert_called_once_with(
        "Sleep", split="test", extract_path=str(tmp_path), return_metadata=True
    )
    np.testing.assert_array_equal(X, expected[0])
    np.testing.assert_array_equal(y, expected[1])
    assert meta == expected[2]


@_requires_zenodo
def test_load_eeg_from_zenodo(tmp_path):
    """Test downloading a small dataset from Zenodo."""
    X, y = load_eeg_classification("Sleep", extract_path=str(tmp_path))  # ~6 MB
    assert X.shape == (3599, 1, 178)
