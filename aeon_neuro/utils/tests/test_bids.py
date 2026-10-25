"""Test native EEG and MNE Raw conversion through actual BIDS read-back."""

from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_allclose, assert_array_equal

from aeon_neuro.utils.bids import convert_eeg_to_bids
from aeon_neuro.utils.data_io import load_eeg_to_mne


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
        convert_eeg_to_bids(tmp_path / "recording.abc", BIDSPath(root=tmp_path))


@pytest.mark.parametrize("extension,backend", [("edf", "edfio"), ("set", "eeglabio")])
def test_convert_native_roundtrip(tmp_path, extension, backend):
    """Retain physical samples, channel order and annotations in new input routes."""
    pytest.importorskip(backend)
    raw = mne.io.RawArray(
        np.arange(400).reshape(4, 100) * 1e-7,
        mne.create_info(["Fp1", "Fp2", "Cz", "Pz"], 100, "eeg"),
    )
    # Give the generated EEGLAB fixture defined positions rather than NaN locations.
    raw.set_montage("standard_1020")
    raw.set_annotations(mne.Annotations([0.2], [0.1], ["stimulus"]))
    source = tmp_path / f"recording.{extension}"
    raw.export(source)
    native = load_eeg_to_mne(source)
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="test")

    # Embedded EEGLAB data are preloaded by its reader and need explicit export.
    options = {"allow_preload": True, "format": "EEGLAB"} if native.preload else {}
    written = convert_eeg_to_bids(source, destination, **options)
    restored = read_raw_bids(written)

    assert written.extension == f".{extension}"
    assert restored.ch_names == native.ch_names
    assert restored.get_channel_types() == native.get_channel_types()
    assert restored.info["sfreq"] == native.info["sfreq"]
    assert restored.n_times == native.n_times
    # Compare to the encoded input, since EDF export quantises physical samples.
    assert_array_equal(restored.get_data(), native.get_data())
    assert_array_equal(restored.annotations.description, native.annotations.description)
    assert_allclose(restored.annotations.onset, native.annotations.onset, atol=1e-9)
    assert_allclose(
        restored.annotations.duration, native.annotations.duration, atol=1e-9
    )


@pytest.mark.parametrize(
    "sample,event_id,message",
    [
        (9, {"stimulus": 7}, "samples"),
        (210, {"stimulus": 7}, "samples"),
        (11, {"other": 9}, "missing descriptions"),
        (11, {"stimulus": 7}, "nonzero first_samp"),
    ],
)
def test_convert_invalid_events_before_writing(tmp_path, sample, event_id, message):
    """Reject lost events or incomplete mappings before creating dataset metadata."""
    raw = mne.io.RawArray(
        np.zeros((1, 200)), mne.create_info(["Cz"], 100, "eeg"), first_samp=10
    )
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="test")

    with pytest.raises(ValueError, match=message):
        convert_eeg_to_bids(
            raw,
            destination,
            events=np.array([[sample, 0, 7]]),
            event_id=event_id,
            allow_preload=True,
            format="BrainVision",
        )

    assert not destination.root.exists()
    assert len(raw.annotations) == 0


@pytest.mark.parametrize(
    "events,event_id,error,message",
    [
        (np.array([[10, 7]]), {"stimulus": 7}, ValueError, "shape"),
        (np.array([[10.5, 0, 7]]), {"stimulus": 7}, TypeError, "integer"),
        (np.array([[10, 0, 7]]), None, ValueError, "event_id is required"),
        (
            np.array([[10, 0, 7]]),
            {"stimulus": 7},
            ValueError,
            "all native annotation descriptions",
        ),
    ],
)
def test_convert_invalid_event_metadata_before_writing(
    tmp_path, events, event_id, error, message
):
    """Reject malformed events and missing mappings before writer side effects."""
    raw = mne.io.RawArray(np.zeros((1, 200)), mne.create_info(["Cz"], 100, "eeg"))
    raw.set_annotations(mne.Annotations([0.2], [0], ["annotation"]))
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="test")

    with pytest.raises(error, match=message):
        convert_eeg_to_bids(raw, destination, events=events, event_id=event_id)

    assert not destination.root.exists()
    assert_array_equal(raw.annotations.description, ["annotation"])


@pytest.mark.parametrize("event_source", ["array", "file"])
@pytest.mark.parametrize("first_samp", [0, 100])
def test_convert_event_boundaries_roundtrip(tmp_path, first_samp, event_source):
    """Preserve events at both inclusive sample bounds through BIDS read-back."""
    pytest.importorskip("pybv")
    raw = mne.io.RawArray(
        np.zeros((1, 200)), mne.create_info(["Cz"], 100, "eeg"), first_samp=first_samp
    )
    # Nonzero offsets need a known origin to avoid the writer's timing ambiguity.
    if first_samp:
        raw.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))
    events = np.array([[raw.first_samp, 0, 7], [raw.last_samp, 0, 8]])
    source = events
    if event_source == "file":
        source = tmp_path / "boundary-events.eve"
        np.savetxt(source, events, fmt="%d")
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="test")

    written = convert_eeg_to_bids(
        raw,
        destination,
        events=source,
        event_id={"first": 7, "last": 8},
        allow_preload=True,
        format="BrainVision",
    )
    restored = read_raw_bids(written)
    restored_events, _ = mne.events_from_annotations(
        restored, event_id={"first": 7, "last": 8}
    )

    assert restored.n_times == raw.n_times
    assert_array_equal(restored.annotations.description, ["first", "last"])
    assert_allclose(restored.annotations.onset, [0, 1.99], rtol=0, atol=1e-9)
    assert_array_equal(restored_events, [[0, 0, 7], [199, 0, 8]])
    assert_array_equal(events, [[first_samp, 0, 7], [first_samp + 199, 0, 8]])
    assert len(raw.annotations) == 0


@pytest.mark.parametrize("first_samp", [0, 100])
def test_convert_raw_array_roundtrip(tmp_path, first_samp):
    """Export in-memory samples and acquisition metadata when explicitly enabled."""
    pytest.importorskip("pybv")
    data = np.array([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]])
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Fp1", "EOG"], 100, ["eeg", "eog"]),
        first_samp=first_samp,
    )
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))
    raw.info["line_freq"] = 50
    raw.set_annotations(mne.Annotations([0.02], [0.01], ["annotation"]))
    destination = BIDSPath(root=tmp_path, subject="01", task="test", datatype="eeg")

    written = convert_eeg_to_bids(
        raw,
        destination,
        events=np.array([[first_samp + 1, 0, 7]]),
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
