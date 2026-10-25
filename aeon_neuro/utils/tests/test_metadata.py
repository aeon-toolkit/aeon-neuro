"""Test explicit metadata preparation and preservation of the input recording."""

import json

import mne
import numpy as np
import pytest
from mne._fiff.constants import FIFF
from numpy.testing import assert_array_equal

from aeon_neuro.utils.metadata import prepare_eeg_metadata


@pytest.mark.parametrize("frequency", [None, -50, float("nan")])
def test_metadata_report(frequency):
    """Report unknown frequency without guessing acquisition facts."""
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["Cz"], 100, "eeg"))
    raw.info["line_freq"] = frequency
    prepared, report = prepare_eeg_metadata(raw)

    assert prepared is not raw
    assert report["issues"][0]["code"] == (
        "unknown" if frequency is None else "invalid"
    )
    assert report["channel_types"] == {"Cz": "eeg"}
    assert report["corrections"] == []
    assert prepared.info["line_freq"] == raw.info["line_freq"] or np.isnan(frequency)
    json.dumps(report, allow_nan=False)


def test_verified_metadata_corrections():
    """Correct roles/frequency on a copy with serializable correction provenance."""
    data = np.arange(20).reshape(2, 10) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(["Cz", "Eye"], 100, "eeg"))
    raw.set_annotations(mne.Annotations([0.01], [0.02], ["marker"]))
    prepared, report = prepare_eeg_metadata(
        raw,
        channel_types={"Eye": "eog"},
        line_freq=50,
        correction_source="acquisition record, revision 1",
    )

    assert prepared.get_channel_types() == ["eeg", "eog"]
    assert prepared.info["line_freq"] == 50
    assert report["issues"] == []
    assert len(report["corrections"]) == 2
    assert all(
        change["source"] == "acquisition record, revision 1"
        for change in report["corrections"]
    )
    assert_array_equal(prepared.get_data(), data)
    assert_array_equal(prepared.annotations.onset, raw.annotations.onset)
    assert_array_equal(prepared.annotations.description, raw.annotations.description)
    assert raw.get_channel_types() == ["eeg", "eeg"]
    assert raw.info["line_freq"] is None
    json.dumps(report, allow_nan=False)


def test_channel_type_unit_change_preserves_samples():
    """Report a declared-unit change without rescaling or modifying the input."""
    data = np.arange(20).reshape(2, 10) * 1e-6
    raw = mne.io.RawArray(data.copy(), mne.create_info(["Cz", "Aux"], 100, "eeg"))
    # MISC changes the unit tag, whereas EEG and EOG both use volts.
    with pytest.warns(RuntimeWarning, match="unit.*changed"):
        prepared, report = prepare_eeg_metadata(
            raw,
            channel_types={"Aux": "misc"},
            correction_source="acquisition record, revision 1",
        )

    assert prepared.get_channel_types() == ["eeg", "misc"]
    assert prepared.info["chs"][0]["unit"] == FIFF.FIFF_UNIT_V
    assert prepared.info["chs"][1]["unit"] == FIFF.FIFF_UNIT_NONE
    assert report["corrections"] == [
        {
            "field": "channel_type",
            "channel": "Aux",
            "before": "eeg",
            "after": "misc",
            "source": "acquisition record, revision 1",
            "unit_before": int(FIFF.FIFF_UNIT_V),
            "unit_after": int(FIFF.FIFF_UNIT_NONE),
        }
    ]
    assert_array_equal(prepared.get_data(), data)
    assert_array_equal(raw.get_data(), data)
    assert raw.get_channel_types() == ["eeg", "eeg"]
    assert all(channel["unit"] == FIFF.FIFF_UNIT_V for channel in raw.info["chs"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("frequency", [-50, True, float("nan")])
def test_invalid_frequency_correction(frequency):
    """Reject invalid corrections rather than writing invalid acquisition fields."""
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["Cz"], 100, "eeg"))
    with pytest.raises(ValueError, match="finite positive"):
        prepare_eeg_metadata(raw, line_freq=frequency, correction_source="record")
    assert raw.info["line_freq"] is None


def test_correction_requires_source():
    """Do not silently convert a proposed value into verified metadata."""
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["Cz"], 100, "eeg"))
    with pytest.raises(ValueError, match="correction_source"):
        prepare_eeg_metadata(raw, line_freq=50)


def test_metadata_requires_raw():
    """Reject inputs without the recording metadata and preservation contract."""
    with pytest.raises(TypeError, match="MNE Raw"):
        prepare_eeg_metadata(np.zeros((1, 10)))
