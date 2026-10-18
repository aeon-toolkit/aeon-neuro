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

    # Synthetic EEG samples for testing the loader.
    rng = np.random.default_rng(42)
    data = rng.standard_normal((2, n_times)) * 1e-5
    info = mne.create_info(
        ch_names=["EEG01", "EEG02"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info)
    raw.export(filepath, fmt="edf", overwrite=True)

    return filepath


def test_equal_length_recordings_return_3d_array(tmp_path):
    """Return a 3D NumPy array when all recordings have equal shapes."""
    create_recording(tmp_path, run="01")
    create_recording(tmp_path, run="02")

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 2, 100)
    assert len(metadata) == 2


def test_unequal_length_recordings_return_list(tmp_path):
    """Return a list of 2D arrays when recordings have unequal lengths."""
    create_recording(tmp_path, run="01", n_times=100)
    create_recording(tmp_path, run="02", n_times=200)

    X, metadata = load_bids_data(path=tmp_path)

    assert isinstance(X, list)
    assert len(X) == 2
    assert X[0].shape == (2, 100)
    assert X[1].shape == (2, 200)
    assert len(metadata) == 2


def test_multiple_subjects_sessions_and_runs(tmp_path):
    """Load recordings across multiple subjects, sessions and runs."""
    create_recording(tmp_path, subject="01", session="01", task="rest", run="01")
    create_recording(tmp_path, subject="01", session="02", task="rest", run="01")
    create_recording(tmp_path, subject="02", session="01", task="task", run="01")
    create_recording(tmp_path, subject="02", session="01", task="task", run="02")

    X, metadata = load_bids_data(path=tmp_path)

    # Check that every subject, session and run is represented.
    assert len(X) == 4
    assert len(metadata) == 4
    assert {(m["subject"], m["session"], m["task"], m["run"]) for m in metadata} == {
        ("01", "01", "rest", "01"),
        ("01", "02", "rest", "01"),
        ("02", "01", "task", "01"),
        ("02", "01", "task", "02"),
    }


def test_subject_session_task_and_run_selection(tmp_path):
    """Load only recordings matching the specified selection criteria."""
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
    assert metadata[0]["subject"] == "01"
    assert metadata[0]["session"] == "02"
    assert metadata[0]["task"] == "task"
    assert metadata[0]["run"] == "02"


def test_acquisition_selection(tmp_path):
    """Load only recordings matching the specified acquisition."""
    create_recording(tmp_path, acq="standard", run="01")
    create_recording(tmp_path, acq="alternate", run="02")

    X, metadata = load_bids_data(path=tmp_path, acqs=["alternate"])

    assert len(X) == 1
    assert metadata[0]["acq"] == "alternate"
    assert metadata[0]["run"] == "02"


def test_metadata_matches_recordings(tmp_path):
    """Verify that metadata correctly describes each loaded recording."""
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


def test_sessionless_recording(tmp_path):
    """Load recordings without session directories and record a None session."""
    create_recording(tmp_path, subject="01", session=None, task="rest", run="01")

    X, metadata = load_bids_data(path=tmp_path)

    assert len(X) == 1
    assert metadata[0]["subject"] == "01"
    assert metadata[0]["session"] is None


def test_no_matching_recordings_raises_value_error(tmp_path):
    """Raise ValueError when no recordings match the selection criteria."""
    create_recording(tmp_path, task="rest")

    with pytest.raises(ValueError, match="No EEG recordings found"):
        load_bids_data(path=tmp_path, tasks=["task"])
