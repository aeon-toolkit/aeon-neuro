"""Test BrainVision metadata retention and the existing array interface."""

from pathlib import Path

import mne
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon_neuro.utils.data_io import (
    load_brainvision_to_mne,
    load_brainvision_to_numpy,
)


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
    raw = mne.io.RawArray(
        data, mne.create_info(["Fp1", "EOG"], 1000, ["eeg", "eog"])
    )
    # The bundled file has only EEG types; this fixture exercises actual removal.
    monkeypatch.setattr(
        mne.io, "read_raw_brainvision", lambda *args, **kwargs: raw.copy()
    )

    result = load_brainvision_to_numpy("recording.vhdr", remove_non_eeg)

    assert_array_equal(result, data[:1] if remove_non_eeg else data)
