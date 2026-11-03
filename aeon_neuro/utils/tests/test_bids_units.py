"""Test dimensionless native BDF Status through actual BIDS read-back."""

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_array_equal

from aeon_neuro.utils.bids import convert_eeg_to_bids


def _make_bdf(path):
    """Generate one second of EEG and unitless Status without export backends."""

    def field(value, width):
        return str(value).encode("ascii").ljust(width)

    header = b"\xffBIOSEMI" + b" " * 160 + b"01.01.24" + b"00.00.00"
    header += field(768, 8) + field("24BIT", 44)
    header += field(1, 8) + field(1, 8) + field(2, 4)
    for width, values in (
        (16, ["1-Cz", "Status"]),
        (80, ["", ""]),
        (8, ["uV", "Boolean"]),
        (8, [-262144, -8388608]),
        (8, [262143, 8388607]),
        (8, [-8388608, -8388608]),
        (8, [8388607, 8388607]),
        (80, ["", ""]),
        (8, [100, 100]),
        (32, ["", ""]),
    ):
        header += b"".join(field(v, width) for v in values)
    # BDF stores consecutive channel blocks as signed little-endian 24-bit words.
    data = np.concatenate([np.arange(-50, 50), np.tile([0, 1, 0, 3], 25)]).astype("<i4")
    path.write_bytes(header + data.view("u1").reshape(-1, 4)[:, :3].tobytes())


@pytest.mark.parametrize("source_kind", ["filename", "raw"])
def test_convert_keeps_dimensionless_status(tmp_path, source_kind):
    """Retain signal values, native payload and Status unit code in both routes."""
    path = tmp_path / "recording.bdf"
    _make_bdf(path)
    original = mne.io.read_raw_bdf(path)
    source = path if source_kind == "filename" else original
    destination = BIDSPath(root=tmp_path / "output", subject="01", task="rest")

    written = convert_eeg_to_bids(source, destination)
    restored = read_raw_bids(written)

    assert (
        restored.get_channel_types() == original.get_channel_types() == ["eeg", "stim"]
    )
    assert original.info["chs"][-1]["unit"] == -1
    assert restored.info["chs"][-1]["unit"] == -1
    assert restored.info["chs"][0]["unit"] == original.info["chs"][0]["unit"]
    assert_array_equal(original.get_data(), restored.get_data())
    assert written.fpath.read_bytes()[768:] == path.read_bytes()[768:]
