"""Test native EEG and MNE Raw conversion through actual BIDS read-back."""

from pathlib import Path

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_allclose, assert_array_equal

from aeon_neuro.utils.bids import convert_eeg_to_bids


@pytest.mark.parametrize("source_kind", ["filename", "raw"])
def test_convert_brainvision_roundtrip(tmp_path, source_kind):
    """Preserve the bundled recording through filename and Raw conversion."""
    header = (
        Path(__file__).resolve().parents[3]
        / "example_raw_eeg/basic_classification_task/sub-01/ses-01/eeg"
        / "sub-01_ses-01_task-task_run-01_eeg.vhdr"
    )
    original = mne.io.read_raw_brainvision(header)
    source = str(header) if source_kind == "filename" else original
    destination = BIDSPath(root=tmp_path, subject="01", task="test", datatype="eeg")

    written = convert_eeg_to_bids(source, destination)
    restored = read_raw_bids(written)

    assert written.fpath.is_file()
    assert written.extension == ".vhdr"
    assert original.preload is False
    assert restored.ch_names == original.ch_names
    assert restored.get_channel_types() == original.get_channel_types()
    assert restored.info["sfreq"] == original.info["sfreq"]
    assert restored.n_times == original.n_times
    assert_array_equal(restored.get_data(), original.get_data())
    assert_array_equal(
        restored.annotations.description, original.annotations.description
    )
    assert_allclose(
        restored.annotations.onset, original.annotations.onset, rtol=0, atol=1e-9
    )
    assert_allclose(
        restored.annotations.duration, original.annotations.duration, rtol=0, atol=1e-9
    )


def test_convert_invalid_source(tmp_path):
    """Reject objects outside the converter's filename-or-Raw interface."""
    with pytest.raises(TypeError, match="MNE Raw"):
        convert_eeg_to_bids(123, BIDSPath(root=tmp_path))


def test_convert_unsupported_filename(tmp_path):
    """Do not silently claim native support for an unimplemented format."""
    with pytest.raises(ValueError, match=".vhdr"):
        convert_eeg_to_bids(tmp_path / "recording.edf", BIDSPath(root=tmp_path))


def test_convert_raw_array_roundtrip(tmp_path):
    """Export in-memory samples and acquisition metadata when explicitly enabled."""
    pytest.importorskip("pybv")
    data = np.array([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]])
    raw = mne.io.RawArray(data, mne.create_info(["Fp1", "EOG"], 100, ["eeg", "eog"]))
    raw.info["line_freq"] = 50
    raw.set_annotations(mne.Annotations([0.02], [0.01], ["annotation"]))
    destination = BIDSPath(root=tmp_path, subject="01", task="test", datatype="eeg")

    written = convert_eeg_to_bids(
        raw,
        destination,
        events=np.array([[1, 0, 7]]),
        event_id={"stimulus": 7, "annotation": 9},
        allow_preload=True,
        format="BrainVision",
    )
    restored = read_raw_bids(written)

    # BrainVision export uses float32; compare physical values at that precision.
    assert_allclose(restored.get_data(), data, rtol=1e-7, atol=0)
    assert restored.get_channel_types() == ["eeg", "eog"]
    assert restored.info["line_freq"] == 50
    assert_array_equal(restored.annotations.description, ["stimulus", "annotation"])
    assert_allclose(restored.annotations.onset, [0.01, 0.02], rtol=0, atol=1e-9)
    assert_allclose(restored.annotations.duration, [0, 0.01], rtol=0, atol=1e-9)
    events = np.genfromtxt(
        written.copy().update(suffix="events", extension=".tsv").fpath,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert_array_equal(events["value"], [7, 9])
    assert_array_equal(raw.annotations.description, ["annotation"])
    assert_array_equal(raw.get_data(), data)
