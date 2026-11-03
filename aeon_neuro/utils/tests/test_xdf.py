"""Exercise actual XDF decoding and the continuous-grid boundary with tiny files."""

import gzip
import hashlib
import struct
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids

from aeon_neuro.utils.bids import convert_eeg_to_bids
from aeon_neuro.utils.data_io import load_eeg_to_mne

pytest.importorskip("pyxdf")


def _integer(value):
    """Encode an XDF variable-width integer using its four-byte form."""
    return b"\x04" + struct.pack("<I", value)


def _chunk(tag, content):
    """Prefix a tagged XDF chunk with its encoded byte count."""
    body = struct.pack("<H", tag) + content
    return _integer(len(body)) + body


def _header(kind, count, rate, channels):
    """Describe a numeric EEG or irregular string marker stream."""
    info = Element("info")
    for key, value in {
        "name": kind,
        "type": kind,
        "channel_count": str(count),
        "nominal_srate": str(rate),
        "channel_format": "string" if kind == "Markers" else "float32",
        "source_id": kind,
    }.items():
        SubElement(info, key).text = value
    descriptors = SubElement(SubElement(info, "desc"), "channels")
    for channel in channels:
        entry = SubElement(descriptors, "channel")
        for key, value in channel.items():
            SubElement(entry, key).text = value
    return tostring(info)


@pytest.fixture
def xdf_file(tmp_path):
    """Write complete XDF chunks; no mocked decoder or participant data is used."""

    def write(
        *,
        times=None,
        channels=None,
        count=2,
        rate=100,
        suffix=".xdf",
        marker_times=(10.25, 10.5),
        offsets=None,
        chunk_size=100,
    ):
        times = 10 + np.arange(100) / 100 if times is None else np.asarray(times)
        channels = (
            channels
            if channels is not None
            else [
                {"label": "Cz", "type": "EEG", "unit": "uV"},
                {"label": "Eye", "type": "EOG", "unit": "mV"},
            ]
        )
        values = np.arange(len(times) * count, dtype=np.float32).reshape(-1, count)
        contents = b"XDF:" + _chunk(1, b"<info><version>1.0</version></info>")
        contents += _chunk(
            2, struct.pack("<I", 1) + _header("EEG", count, rate, channels)
        )
        for first in range(0, len(times), chunk_size):
            stamps = times[first : first + chunk_size]
            rows = values[first : first + chunk_size]
            samples = _integer(len(stamps)) + b"".join(
                b"\x08" + struct.pack("<d", stamp) + row.astype("<f4").tobytes()
                for stamp, row in zip(stamps, rows)
            )
            contents += _chunk(3, struct.pack("<I", 1) + samples)
        if marker_times is not None:
            contents += _chunk(2, struct.pack("<I", 2) + _header("Markers", 1, 0, []))
            samples = _integer(len(marker_times))
            for index, stamp in enumerate(marker_times):
                label = f"event-{index}".encode()
                samples += (
                    b"\x08" + struct.pack("<d", stamp) + _integer(len(label)) + label
                )
            contents += _chunk(3, struct.pack("<I", 2) + samples)
        if offsets is not None:
            for stream_id, offset in enumerate(offsets, 1):
                for timestamp in (90.0, 110.0):
                    contents += _chunk(
                        4, struct.pack("<Idd", stream_id, timestamp, offset)
                    )
        path = tmp_path / ("recording" + suffix)
        path.write_bytes(gzip.compress(contents) if suffix == ".xdfz" else contents)
        return path, values, times

    return write


@pytest.mark.parametrize("suffix", [".xdf", ".xdfz"])
@pytest.mark.parametrize("dejitter", [False, True])
def test_xdf_values_units_and_markers(xdf_file, suffix, dejitter):
    """Keep signal axes, calibrated units and shared-clock annotation positions."""
    path, values, times = xdf_file(suffix=suffix)
    before = path.read_bytes()
    raw = load_eeg_to_mne(path, dejitter_timestamps=dejitter)
    assert raw.preload
    assert raw.ch_names == ["Cz", "Eye"]
    assert raw.get_channel_types() == ["eeg", "eog"]
    np.testing.assert_array_equal(raw.get_data(), values.T * np.array([[1e-6], [1e-3]]))
    np.testing.assert_allclose(raw.times, times - times[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(raw.annotations.onset, [0.25, 0.5], atol=1e-12)
    assert raw.annotations.description.tolist() == ["event-0", "event-1"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("dejitter", [False, True])
@pytest.mark.parametrize("problem", ["long_gap", "missing_sample", "jitter"])
def test_xdf_rejects_irregular_time_before_dejitter(xdf_file, dejitter, problem):
    """Neither disabled dejittering nor relaxed backend thresholds may hide gaps."""
    times = 10 + np.arange(100) / 100
    if problem == "long_gap":
        times[50:] += 2
    elif problem == "missing_sample":
        times = np.delete(times, 50)
    else:
        times[50] += 0.004
    path, _, _ = xdf_file(times=times, marker_times=None, chunk_size=50)
    with pytest.raises(ValueError, match="timestamps|grid|gap"):
        load_eeg_to_mne(
            path,
            dejitter_timestamps=dejitter,
            jitter_break_threshold_seconds=10000,
            jitter_break_threshold_samples=1000000,
        )


@pytest.mark.parametrize(
    "problem", ["duplicate", "backwards", "nan", "empty", "single"]
)
def test_xdf_rejects_unusable_time(xdf_file, problem):
    """Require finite increasing timestamps and enough samples to check the grid."""
    times = 10 + np.arange(100) / 100
    if problem == "duplicate":
        times[50] = times[49]
    elif problem == "backwards":
        times[50] = times[49] - 0.01
    elif problem == "nan":
        times[50] = np.nan
    elif problem == "empty":
        times = times[:0]
    else:
        times = times[:1]
    path, _, _ = xdf_file(times=times, marker_times=None)
    with pytest.raises(ValueError, match="timestamps|samples|grid"):
        load_eeg_to_mne(path, dejitter_timestamps=False)


@pytest.mark.parametrize(
    "problem", ["count", "duplicate", "missing_label", "unit", "type"]
)
def test_xdf_channel_metadata_cannot_broadcast_or_invent_identity(xdf_file, problem):
    """Channel-count mismatches must fail before a unit vector can broadcast."""
    channels = [
        {"label": "Cz", "type": "EEG", "unit": "uV"},
        {"label": "Eye", "type": "EOG", "unit": "mV"},
    ]
    count = 1 if problem == "count" else 2
    if problem == "duplicate":
        channels[1]["label"] = "Cz"
    elif problem == "missing_label":
        channels[0].pop("label")
    elif problem == "unit":
        channels[0]["unit"] = "unknown"
    elif problem == "type":
        channels[0]["type"] = "STIM"
    path, _, _ = xdf_file(channels=channels, count=count)
    with pytest.raises(ValueError, match="channel|label|unit|voltage"):
        load_eeg_to_mne(path)


@pytest.mark.parametrize("marker_times", [(9.99,), (11.01,), (np.nan,)])
def test_xdf_does_not_silently_drop_unrepresentable_markers(xdf_file, marker_times):
    """Annotations must be finite and supported by the selected EEG recording."""
    path, _, _ = xdf_file(marker_times=marker_times)
    with pytest.raises(ValueError, match="marker|Marker"):
        load_eeg_to_mne(path)


def test_xdf_clock_correction_aligns_marker_and_eeg(xdf_file):
    """Independent clock offsets must be applied before subtracting EEG origin."""
    path, _, _ = xdf_file(marker_times=(9.25,), offsets=(2.0, 3.0))
    raw = load_eeg_to_mne(path)
    np.testing.assert_allclose(raw.annotations.onset, [0.25], atol=1e-10)


def test_xdf_selection_can_deliberately_omit_markers(xdf_file):
    """Unselected streams must not contribute annotations or validation failures."""
    path, _, _ = xdf_file(marker_times=(9.0,))
    raw = load_eeg_to_mne(path, select_streams=[1])
    assert len(raw.annotations) == 0


@pytest.mark.parametrize("rate", [0, np.nan])
def test_xdf_rejects_undeclared_regular_rate(xdf_file, rate):
    """An irregular/invalid nominal rate cannot establish a continuous EEG grid."""
    path, _, _ = xdf_file(rate=rate, marker_times=None)
    with pytest.raises(ValueError, match="nominal sampling rate"):
        load_eeg_to_mne(path)


@pytest.mark.parametrize("dejitter", [False, True])
@pytest.mark.parametrize("actual_rate", [98.9, 101.1, 200])
def test_xdf_rejects_rate_inconsistent_with_nominal(xdf_file, dejitter, actual_rate):
    """A regular grid must not conceal incorrect nominal-rate metadata."""
    path, _, _ = xdf_file(
        times=10 + np.arange(100) / actual_rate, rate=100, marker_times=None
    )
    with pytest.raises(ValueError, match="effective sampling rate.*nominal"):
        load_eeg_to_mne(path, dejitter_timestamps=dejitter)


@pytest.mark.parametrize("dejitter", [False, True])
@pytest.mark.parametrize("actual_rate", [99, 101])
def test_xdf_accepts_nominal_rate_clock_drift(xdf_file, dejitter, actual_rate):
    """Allow one-percent clock drift while keeping the effective sample rate."""
    path, values, _ = xdf_file(
        times=10 + np.arange(100) / actual_rate, rate=100, marker_times=None
    )
    before = path.read_bytes()
    raw = load_eeg_to_mne(path, dejitter_timestamps=dejitter)
    assert raw.info["sfreq"] == pytest.approx(actual_rate)
    np.testing.assert_array_equal(raw.get_data(), values.T * np.array([[1e-6], [1e-3]]))
    assert path.read_bytes() == before


def test_xdf_timing_tolerance_is_explicit(xdf_file):
    """Accept only bounded sub-sample deviations; do not interpolate signal values."""
    times = 10 + np.arange(100) / 100
    times[50] += 0.002
    path, values, _ = xdf_file(times=times, marker_times=None)
    raw = load_eeg_to_mne(path, dejitter_timestamps=False)
    np.testing.assert_array_equal(raw.get_data()[0], values[:, 0].astype(float) * 1e-6)
    with pytest.raises(ValueError, match="grid"):
        load_eeg_to_mne(path, dejitter_timestamps=False, timing_tolerance=0.1)
    for tolerance in (-1, 0.5, np.nan, True):
        with pytest.raises(ValueError, match="timing_tolerance"):
            load_eeg_to_mne(path, timing_tolerance=tolerance)


def test_xdf_preserves_callback_and_surfaces_callback_failure(xdf_file):
    """Wrap callbacks once and prevent PyXDF from hiding an exception as recovery."""
    path, _, _ = xdf_file(marker_times=None, chunk_size=50)
    calls = []

    def callback(values, stamps, header, stream_id):
        calls.append(stream_id)
        return values, stamps, header

    load_eeg_to_mne(path, on_chunk=callback)
    assert calls == [1, 1]

    failed_calls = 0

    def broken(values, stamps, header, stream_id):
        nonlocal failed_calls
        failed_calls += 1
        # A late failure used to leave a readable but silently shortened stream.
        if failed_calls == 2:
            raise RuntimeError("callback failed")
        return values, stamps, header

    with pytest.raises(ValueError, match="on_chunk"):
        load_eeg_to_mne(path, on_chunk=broken)


def test_xdf_preloaded_bids_export(xdf_file, tmp_path):
    """Exercise genuine XDF to EDF/BIDS with explicit preload/export decisions."""
    pytest.importorskip("edfio")
    path, _, _ = xdf_file(times=10 + np.arange(128) / 128, rate=128)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = load_eeg_to_mne(path)
    destination = BIDSPath(root=tmp_path / "bids", subject="01", task="xdf")
    with pytest.raises(ValueError, match="allow_preload"):
        convert_eeg_to_bids(raw, destination, format="EDF")
    written = convert_eeg_to_bids(
        raw,
        destination,
        format="EDF",
        allow_preload=True,
        event_id={"event-0": 1, "event-1": 2},
    )
    restored = read_raw_bids(written)
    for actual, expected in zip(restored.get_data(), raw.get_data()):
        tolerance = np.ptp(expected) / 65534 + 1e-12
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    np.testing.assert_allclose(
        restored.annotations.onset, raw.annotations.onset, atol=1e-10
    )
    assert restored.ch_names == raw.ch_names
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
