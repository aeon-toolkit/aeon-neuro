"""Tests for loading EEG recordings from BIDS datasets."""

import os

import mne
import numpy as np
import pytest

from aeon_neuro.datasets.load_bids import load_bids_data


def create_recording(
    root,
    subject="01",
    session="01",
    task="rest",
    acq=None,
    run="01",
    n_times=100,
    sfreq=100,
    ch_names=("EEG01", "EEG02"),
):
    """Create a synthetic EDF recording in a BIDS-like directory."""
    eeg_dir = root / f"sub-{subject}"

    if session is not None:
        eeg_dir /= f"ses-{session}"

    eeg_dir /= "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    entities = [f"sub-{subject}"]

    if session is not None:
        entities.append(f"ses-{session}")
    if task is not None:
        entities.append(f"task-{task}")
    if acq is not None:
        entities.append(f"acq-{acq}")
    if run is not None:
        entities.append(f"run-{run}")

    filepath = eeg_dir / ("_".join(entities) + "_eeg.edf")

    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(ch_names), n_times)) * 1e-5

    info = mne.create_info(
        ch_names=list(ch_names),
        sfreq=sfreq,
        ch_types=["eeg"] * len(ch_names),
    )
    raw = mne.io.RawArray(data, info)
    raw.export(filepath, fmt="edf", overwrite=True)

    return filepath


def test_equal_length_recordings(tmp_path):
    """Return a 3D array when recordings are compatible."""
    create_recording(tmp_path, run="01", sfreq=100, ch_names=("EEG01", "EEG02"))
    create_recording(tmp_path, run="02", sfreq=100, ch_names=("EEG01", "EEG02"))

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2, 100)
    assert len(metadata) == 2
    assert all(recording.shape == (2, 100) for recording in X)
    assert metadata[0]["channel_names"] == metadata[1]["channel_names"]
    assert metadata[0]["sampling_frequency"] == pytest.approx(
        metadata[1]["sampling_frequency"]
    )


def test_unequal_length_recordings(tmp_path):
    """Return 2D arrays in a list when recording lengths differ."""
    create_recording(tmp_path, run="01", n_times=100)
    create_recording(tmp_path, run="02", n_times=200)

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, list)
    assert len(X) == 2
    assert [recording.shape for recording in X] == [
        (2, 100),
        (2, 200),
    ]
    assert len(metadata) == 2


def test_different_channel_names(tmp_path):
    """Return 2D arrays when channel names differ."""
    create_recording(tmp_path, run="01", ch_names=("EEG01", "EEG02"))
    create_recording(tmp_path, run="02", ch_names=("EEG01", "EEG03"))

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, list)
    assert len(X) == 2
    assert all(recording.shape == (2, 100) for recording in X)
    assert metadata[0]["channel_names"] != metadata[1]["channel_names"]


def test_different_channel_order(tmp_path):
    """Return 2D arrays when channel order differs."""
    create_recording(tmp_path, run="01", ch_names=("EEG01", "EEG02"))
    create_recording(tmp_path, run="02", ch_names=("EEG02", "EEG01"))

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, list)
    assert len(X) == 2
    assert all(recording.shape == (2, 100) for recording in X)
    assert metadata[0]["channel_names"] != metadata[1]["channel_names"]


def test_different_sampling_frequencies(tmp_path):
    """Return 2D arrays when sampling frequencies differ."""
    create_recording(tmp_path, run="01", sfreq=100)
    create_recording(tmp_path, run="02", sfreq=200)

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, list)
    assert len(X) == 2
    assert all(recording.ndim == 2 for recording in X)
    assert all(recording.shape[0] == 2 for recording in X)
    assert metadata[0]["sampling_frequency"] == pytest.approx(100.0)
    assert metadata[1]["sampling_frequency"] == pytest.approx(200.0)


def test_multiple_subjects_sessions_and_runs(tmp_path):
    """Load recordings across subjects, sessions, and runs."""
    create_recording(tmp_path, subject="01", session="01", task="rest", run="01")
    create_recording(tmp_path, subject="01", session="02", task="rest", run="01")
    create_recording(tmp_path, subject="02", session="01", task="task", run="01")
    create_recording(tmp_path, subject="02", session="01", task="task", run="02")

    X, metadata = load_bids_data(path=tmp_path)

    assert len(X) == 4
    assert len(metadata) == 4

    entities = {(m["subject"], m["session"], m["task"], m["run"]) for m in metadata}

    assert entities == {
        ("01", "01", "rest", "01"),
        ("01", "02", "rest", "01"),
        ("02", "01", "task", "01"),
        ("02", "01", "task", "02"),
    }


def test_subject_session_task_and_run_selection(tmp_path):
    """Load only recordings matching all supplied selection criteria."""
    create_recording(tmp_path, subject="01", session="01", task="rest", run="01")
    create_recording(tmp_path, subject="01", session="02", task="task", run="02")
    create_recording(tmp_path, subject="02", session="02", task="task", run="02")

    X, metadata = load_bids_data(
        path=tmp_path,
        subjects=["01"],
        sessions=["02"],
        tasks=["task"],
        runs=["02"],
    )

    assert len(X) == 1
    assert len(metadata) == 1
    assert metadata[0]["subject"] == "01"
    assert metadata[0]["session"] == "02"
    assert metadata[0]["task"] == "task"
    assert metadata[0]["run"] == "02"


def test_acquisition_selection(tmp_path):
    """Filter recordings by acquisition identifier."""
    create_recording(tmp_path, acq="standard", run="01")
    create_recording(tmp_path, acq="alternate", run="02")

    X, metadata = load_bids_data(path=tmp_path, acqs=["alternate"])

    assert len(X) == 1
    assert metadata[0]["acq"] == "alternate"
    assert metadata[0]["run"] == "02"


def test_metadata_matches_recordings(tmp_path):
    """Verify metadata for every loaded recording."""
    create_recording(tmp_path, subject="01", session="01", task="rest", run="01")
    create_recording(tmp_path, subject="02", session="02", task="task", run="02")

    X, metadata = load_bids_data(path=tmp_path)
    recordings = X if isinstance(X, list) else list(X)

    assert len(recordings) == len(metadata)

    for recording, item in zip(recordings, metadata):
        assert recording.ndim == 2
        assert recording.shape[0] == len(item["channel_names"])
        assert item["sampling_frequency"] == pytest.approx(100.0)
        assert item["filename"].endswith("_eeg.edf")
        assert os.path.isfile(tmp_path / item["filepath"])
        assert item["filepath"].endswith(item["filename"])
        assert item["subject"] is not None
        assert item["task"] is not None
        assert item["run"] is not None


def test_sessionless_recording(tmp_path):
    """Load a recording without a session directory."""
    create_recording(tmp_path, subject="01", session=None, task="rest", run="01")

    X, metadata = load_bids_data(path=tmp_path)

    assert len(X) == 1
    assert metadata[0]["subject"] == "01"
    assert metadata[0]["session"] is None


def test_optional_bids_entities_are_none(tmp_path):
    """Record absent optional BIDS entities as None."""
    create_recording(
        tmp_path,
        session=None,
        task="rest",
        acq=None,
        run=None,
    )

    X, metadata = load_bids_data(path=tmp_path)

    assert len(X) == 1
    assert metadata[0]["session"] is None
    assert metadata[0]["task"] == "rest"
    assert metadata[0]["acq"] is None
    assert metadata[0]["run"] is None


def test_custom_formatts(tmp_path):
    """Raise ValueError when no recordings match the specified file extensions."""
    create_recording(tmp_path)

    with pytest.raises(ValueError, match="No EEG recordings found"):
        load_bids_data(path=tmp_path, formatts=(".vhdr",))


def test_no_matching_recordings(tmp_path):
    """Raise ValueError when selection criteria match no recordings."""
    create_recording(tmp_path, task="rest")

    with pytest.raises(ValueError, match="No EEG recordings found"):
        load_bids_data(path=tmp_path, tasks=["task"])


def test_empty_dataset(tmp_path):
    """Raise ValueError when the dataset contains no EEG recordings."""
    with pytest.raises(ValueError, match="No EEG recordings found"):
        load_bids_data(path=tmp_path)


def test_empty_selection(tmp_path):
    """Raise ValueError when an empty selection matches no recordings."""
    create_recording(tmp_path)

    with pytest.raises(ValueError, match="No EEG recordings found"):
        load_bids_data(path=tmp_path, subjects=["99"])


def test_single_recording(tmp_path):
    """Return a 3D array for one compatible recording."""
    create_recording(tmp_path)

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 2, 100)
    assert len(metadata) == 1


def test_relative_dataset_path(tmp_path, monkeypatch):
    """Resolve a relative dataset path from the current directory."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    create_recording(dataset_dir)

    monkeypatch.chdir(tmp_path)
    X, metadata = load_bids_data(path="dataset")

    assert len(X) == 1
    assert len(metadata) == 1
    assert os.path.isfile(dataset_dir / metadata[0]["filepath"])
