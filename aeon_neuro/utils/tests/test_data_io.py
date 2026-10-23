"""Test BrainVision metadata retention and the existing array interface."""

from pathlib import Path

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon_neuro.utils.data_io import (
    load_auxiliary_info,
    load_brainvision_to_mne,
    load_brainvision_to_numpy,
    load_eeg_to_mne,
)


@pytest.mark.parametrize("extension,width", [("edf", 2), ("bdf", 3)])
@pytest.mark.parametrize(
    "variant",
    ["truncated", "discontinuous", "unknown_count", "count_too_low", "count_too_high"],
)
def test_edf_bdf_record_layout(tmp_path, extension, width, variant):
    """Reject lost records and time gaps while accepting legal unknown counts."""
    # One 100 Hz channel and two complete seconds, using the shared fixed header.
    fields = [
        ("0" if extension == "edf" else "\xffBIOSEMI", 8),
        ("X", 80),
        ("X", 80),
        ("01.01.00", 8),
        ("00.00.00", 8),
        ("512", 8),
        ("", 44),
        ("2", 8),
        ("1", 8),
        ("1", 4),
        ("Cz", 16),
        ("", 80),
        ("uV", 8),
        ("-1", 8),
        ("1", 8),
        ("-32768" if width == 2 else "-8388608", 8),
        ("32767" if width == 2 else "8388607", 8),
        ("", 80),
        ("100", 8),
        ("", 32),
    ]
    content = bytearray(
        b"".join(value.encode("latin-1").ljust(size) for value, size in fields)
    )
    content.extend(bytes(200 * width))
    source = tmp_path / f"recording.{extension}"
    if variant == "truncated":
        content = content[:-1]
    elif variant == "discontinuous":
        content[192:236] = ("EDF+D" if width == 2 else "BDF+D").encode().ljust(44)
    elif variant in ("count_too_low", "count_too_high"):
        # Keep both payload records complete so only the declared count is wrong.
        content[236:244] = b"1       " if variant == "count_too_low" else b"3       "
    else:
        content[236:244] = b"-1      "
    source.write_bytes(content)

    if variant == "unknown_count":
        raw = load_eeg_to_mne(source)
        assert raw.n_times == 200
        assert raw.info["sfreq"] == 100
    else:
        with pytest.raises(ValueError, match="records|Discontinuous"):
            load_eeg_to_mne(source)


def test_auxiliary_info(tmp_path):
    """Return decoded auxiliary data from a readable JSON file."""
    (tmp_path / "dataset.json").write_text('{"sfreq": 1000}')
    assert load_auxiliary_info(str(tmp_path) + "/", "dataset") == {"sfreq": 1000}


@pytest.mark.parametrize("contents", [None, "invalid JSON"])
def test_auxiliary_info_unreadable(tmp_path, contents):
    """Warn and retain the None fallback for missing or malformed metadata."""
    if contents is not None:
        (tmp_path / "dataset.json").write_text(contents)
    with pytest.warns(UserWarning, match="Unable to load auxiliary information"):
        assert load_auxiliary_info(str(tmp_path) + "/", "dataset") is None


@pytest.mark.parametrize("preload", [False, True])
def test_brainvision_metadata(preload):
    """Keep the bundled recording's channel, sampling and marker information."""
    header = (
        Path(__file__).resolve().parents[3]
        / "example_raw_eeg/basic_classification_task/sub-01/ses-01/eeg"
        / "sub-01_ses-01_task-task_run-01_eeg.vhdr"
    )
    expected = mne.io.read_raw_brainvision(header, preload=False)
    raw = load_brainvision_to_mne(header, preload=preload)

    assert raw.preload == preload
    assert raw.ch_names == expected.ch_names
    assert raw.get_channel_types() == expected.get_channel_types()
    assert raw.info["sfreq"] == 1000
    assert raw.n_times == 82491
    assert_array_equal(raw.annotations.onset, expected.annotations.onset)
    assert_array_equal(raw.annotations.duration, expected.annotations.duration)
    assert_array_equal(raw.annotations.description, expected.annotations.description)
    assert_array_equal(raw.get_data(), expected.get_data())


@pytest.mark.parametrize("remove_non_eeg", [False, True])
def test_brainvision_numpy_channel_selection(monkeypatch, remove_non_eeg):
    """Preserve array values and the legacy choice to exclude non-EEG channels."""
    data = np.array([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]])
    raw = mne.io.RawArray(data, mne.create_info(["Fp1", "EOG"], 1000, ["eeg", "eog"]))
    # The bundled file has only EEG types; this fixture exercises actual removal.
    monkeypatch.setattr(
        mne.io, "read_raw_brainvision", lambda *args, **kwargs: raw.copy()
    )

    result = load_brainvision_to_numpy("recording.vhdr", remove_non_eeg)

    assert_array_equal(result, data[:1] if remove_non_eeg else data)
