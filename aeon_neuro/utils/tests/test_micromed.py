"""Exercise Micromed intake through the native reader and a generated TRC."""

import builtins
import struct

import mne
import numpy as np
import pytest

from aeon_neuro.utils.data_io import load_eeg_to_mne

_ORDER = 512
_ELECTRODES = 640
_SEGMENTS = 944
_TRIGGERS = 1000
_DATA = 1056


@pytest.fixture
def trc(tmp_path):
    """Write a type-4 header and two voltage channels in swapped header order."""
    pytest.importorskip("neo")
    path = tmp_path / "recording.TRC"
    header = bytearray(_DATA)
    header[64:86] = b"SYNTHETIC".ljust(22)
    struct.pack_into("<6b", header, 128, 1, 1, 124, 0, 0, 0)
    struct.pack_into("<IHHHH", header, 138, _DATA, 2, 2, 256, 2)
    header[175] = 4
    zones = [
        ("ORDER", _ORDER, 4),
        ("LABCOD", _ELECTRODES, 256),
        ("NOTE", 896, 44),
        ("FLAGS", 0, 0),
        ("TRONCA", _SEGMENTS, 24),
        ("IMPED_B", 0, 0),
        ("IMPED_E", 0, 0),
        ("MONTAGE", 0, 0),
        ("COMPRESS", 0, 0),
        ("AVERAGE", 0, 0),
        ("HISTORY", 0, 0),
        ("DVIDEO", 0, 0),
        ("EVENT A", 968, 12),
        ("EVENT B", 980, 12),
        ("TRIGGER", _TRIGGERS, 24),
    ]
    for index, (name, start, size) in enumerate(zones):
        struct.pack_into(
            "<8sII", header, 176 + index * 16, name.encode().ljust(8), start, size
        )
    struct.pack_into("<HH", header, _ORDER, 1, 0)
    for index, (name, unit, ground, span) in enumerate(
        [(b"C3", 0, 200, 512), (b"C4", 1, 100, 256)]
    ):
        electrode = _ELECTRODES + 128 * index
        struct.pack_into(
            "<6s6siiiiih",
            header,
            electrode + 2,
            name,
            b"REF",
            0,
            255,
            ground,
            -span // 2,
            span // 2,
            unit,
        )
        struct.pack_into("<H", header, electrode + 44, 1)
    struct.pack_into("<I40s", header, 896, 3, b"synthetic operator note")
    for index, (sample, code) in enumerate([(0, 21), (3, 7), (7, 0)]):
        struct.pack_into("<IH", header, _TRIGGERS + index * 6, sample, code)
    samples = np.column_stack([100 + np.arange(8), 200 + 2 * np.arange(8)])
    path.write_bytes(header + samples.astype("<u2").tobytes())
    return path


@pytest.mark.parametrize("preload", [False, True])
def test_read_trc_preserves_samples_and_triggers(trc, preload):
    """Load voltages and codes on the exact source sample grid."""
    raw = load_eeg_to_mne(trc, preload=preload, verbose=False)
    assert raw.preload and raw.n_times == 8
    assert raw.info["sfreq"] == 256
    assert raw.ch_names == ["C4", "C3"]
    assert raw.get_channel_types() == ["eeg", "eeg"]
    expected = np.arange(8) * np.array([[1e-3], [4e-6]])
    np.testing.assert_allclose(raw.get_data(), expected, rtol=1e-12, atol=0)
    assert raw.annotations.description.tolist() == ["21", "7", "0"]
    np.testing.assert_array_equal(
        mne.events_from_annotations(raw, verbose=False)[0][:, 0], [0, 3, 7]
    )
    assert raw.info["subject_info"] is None and raw.info["meas_date"] is None


@pytest.mark.parametrize("unit, scale", [(-1, 1e-9), (2, 1.0)])
def test_trc_uses_declared_voltage_unit(trc, unit, scale):
    """Use the TRC unit code even when Neo represents it generically."""
    with trc.open("r+b") as stream:
        stream.seek(_ELECTRODES + 34)
        stream.write(struct.pack("<h", unit))
    raw = load_eeg_to_mne(trc, verbose=False)
    np.testing.assert_allclose(raw.get_data(picks=["C3"])[0], 4 * scale * np.arange(8))


@pytest.mark.parametrize(
    "offset, layout, values, message",
    [
        (175, "<B", (5,), "type-4"),
        (_ELECTRODES + 34, "<h", (99,), "voltage units"),
        (_ELECTRODES + 44, "<H", (2,), "sampling rates"),
        (_SEGMENTS, "<II", (9, 2), "continuous, complete segment"),
        (_SEGMENTS, "<IIII", (1, 0, 9, 4), "continuous, complete segment"),
        (_TRIGGERS, "<IH", (8, 21), "outside the recording"),
    ],
)
def test_read_trc_rejects_unsupported_layout(trc, offset, layout, values, message):
    """Reject formats that would require a guessed channel or time grid."""
    with trc.open("r+b") as stream:
        stream.seek(offset)
        stream.write(struct.pack(layout, *values))
    with pytest.raises(ValueError, match=message):
        load_eeg_to_mne(trc)


def test_truncated_trc_rejected(trc):
    """A partial final sample frame cannot be silently discarded."""
    trc.write_bytes(trc.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incomplete samples"):
        load_eeg_to_mne(trc)


def test_trc_without_triggers(trc):
    """Zeroed entries leave the recording available without annotations."""
    with trc.open("r+b") as stream:
        stream.seek(_TRIGGERS)
        stream.write(bytes(24))
    assert len(load_eeg_to_mne(trc, verbose=False).annotations) == 0


def test_trc_preload_filename_rejected(tmp_path):
    """The in-memory reader cannot honour MNE's memory-map filename."""
    with pytest.raises(ValueError, match="preload must be a bool"):
        load_eeg_to_mne(tmp_path / "recording.trc", preload=str(tmp_path / "map"))


def test_trc_missing_neo_explains_extra(tmp_path, monkeypatch):
    """Advertise the optional dependency without importing it for other formats."""
    original_import = builtins.__import__

    def without_neo(name, *args, **kwargs):
        if name.startswith("neo."):
            raise ModuleNotFoundError(name="neo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_neo)
    with pytest.raises(ImportError, match=r"aeon-neuro\[eeg_io\]"):
        load_eeg_to_mne(tmp_path / "recording.trc")
