"""Offline checks for the COVID19 archive and its download registration."""

import numpy as np
import pytest
from aeon.datasets import save_to_ts_file

from aeon_neuro.datasets import _data_loaders
from aeon_neuro.datasets.classification_datasets import dataset_map, dataset_names


@pytest.fixture
def covid19_archive(tmp_path):
    """Small synthetic archive with the published channel and time dimensions."""
    signals = np.random.default_rng(42).normal(size=(4, 19, 400)).astype(np.float32)
    labels = np.array(["close_contact", "covid", "covid", "close_contact"])
    for split, selection in (("TRAIN", slice(0, 2)), ("TEST", slice(2, 4))):
        save_to_ts_file(
            signals[selection],
            labels[selection],
            label_type="classification",
            path=tmp_path / "COVID19",
            problem_name="COVID19",
            file_suffix=f"_{split}",
        )
    return tmp_path, signals, labels


@pytest.mark.parametrize("split", ["TRAIN", "TEST", None])
def test_covid19_local_archive(covid19_archive, monkeypatch, split):
    """Local loading preserves every value, label and partition without network."""
    root, signals, labels = covid19_archive

    def unexpected_download(**kwargs):
        pytest.fail("Local COVID19 archive triggered a download")

    monkeypatch.setattr(_data_loaders, "_load_collection", unexpected_download)
    actual, actual_labels, metadata = _data_loaders.load_eeg_classification(
        "COVID19", split=split, extract_path=str(root), return_metadata=True
    )
    selection = {"TRAIN": slice(0, 2), "TEST": slice(2, 4), None: slice(None)}[split]
    np.testing.assert_array_equal(actual.astype(np.float32), signals[selection])
    np.testing.assert_array_equal(actual_labels, labels[selection])
    assert metadata["problemname"] == "covid19"
    assert metadata["classlabel"]
    assert set(metadata["class_values"]) == {"close_contact", "covid"}


def test_covid19_unpublished(tmp_path, monkeypatch):
    """An unpublished archive gives local-loading instructions without network."""
    assert "COVID19" in dataset_names
    monkeypatch.setitem(dataset_map, "COVID19", 49)
    with pytest.raises(ValueError, match="not yet available.*Zenodo"):
        _data_loaders.load_eeg_classification("COVID19", extract_path=str(tmp_path))


def test_download_uses_neuro_registry(tmp_path, monkeypatch):
    """Pass the neuro catalogue to aeon, including EEG-only dataset names."""
    monkeypatch.setitem(dataset_map, "COVID19", 123456)  # Synthetic test ID only.
    expected = np.zeros((2, 19, 400))
    labels = np.array(["close_contact", "covid"])
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return expected, labels, {"classlabel": True}

    monkeypatch.setattr(_data_loaders, "_load_collection", download)
    actual, actual_labels = _data_loaders.load_eeg_classification(
        "COVID19", split="TRAIN", extract_path=str(tmp_path)
    )
    assert actual is expected
    assert actual_labels is labels
    assert calls == [
        {
            "name": "COVID19",
            "problem_dict": dataset_map,
            "split": "TRAIN",
            "extract_path": str(tmp_path),
            "return_metadata": True,
        }
    ]


def test_download_rejects_regression(tmp_path, monkeypatch):
    """Apply the classification guard to downloaded files as well as local ones."""
    monkeypatch.setitem(dataset_map, "COVID19", 123456)
    monkeypatch.setattr(
        _data_loaders,
        "_load_collection",
        lambda **kwargs: (
            np.zeros((2, 19, 400)),
            np.array([0.1, 0.2]),
            {"classlabel": False},
        ),
    )
    with pytest.raises(ValueError, match="regression problem"):
        _data_loaders.load_eeg_classification("COVID19", extract_path=str(tmp_path))
