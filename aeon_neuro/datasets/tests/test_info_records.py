"""Protect the portable contract while metadata uses named internal records."""

from copy import deepcopy
from dataclasses import is_dataclass

import numpy as np
import pytest

from aeon_neuro.datasets import EEGDatasetInfo


def saved_state():
    """Version-2 source context with a version-1 current case view."""
    return {
        "version": 2,
        "ready": True,
        "dataset": {
            "description": {"Name": "study"},
            "readme": None,
            "participants": None,
            "participants_json": None,
            "selected_participants": [1, "1"],
            "custom": {"values": [1]},
        },
        "source_ids": ["a", "b"],
        "recordings": [
            {
                "recording_id": identity,
                "participant_id": participant,
                "channel_names": ["C3"],
                "sampling_frequency": 8.0,
                "n_samples": 16,
                "channel_metadata": [{"name": "C3", "units": "uV"}],
                "custom": {"values": [2]},
            }
            for identity, participant in (("a", 1), ("b", "1"))
        ],
        "events": [
            {
                "recording_id": "a",
                "bids_event_id": "event-0",
                "onset": 0.0,
                "bids_onset_sample": 0,
                "custom": {"values": [3]},
            }
        ],
        "view": {
            "version": 1,
            "kind": "cases",
            "rows": [
                {
                    "case_id": "trial-a",
                    "recording_id": "a",
                    "participant_id": 1,
                    "group_id": None,
                    "source_support": [0, 8],
                    "custom": {"values": [4]},
                }
            ],
            "layouts": [
                {
                    "channel_names": ["C3"],
                    "sampling_frequency": 8.0,
                    "units": [None],
                    "n_samples": 8,
                    "source_origin_sample": 0.0,
                    "source_sample_step": 1.0,
                    "custom": {"values": [5]},
                }
            ],
            "custom": {"values": [6]},
        },
        "custom": {"values": [7]},
    }


@pytest.mark.parametrize("current", [True, False])
def test_saved_dictionary_round_trip_preserves_extensions_and_omissions(current):
    """Restoration retains absent fields, custom facts and identifier types."""
    state = saved_state()
    if not current:
        state.pop("view")
    info = EEGDatasetInfo(state)
    assert info.to_dict() == state
    assert is_dataclass(info._dataset)
    assert is_dataclass(info.recordings[0]._source)
    if current:
        assert is_dataclass(info._view)
        assert is_dataclass(info._view.rows[0])
        assert is_dataclass(info._view.layouts[0])
        assert info._view.rows[0].case_id == "trial-a"
        assert info._view.layouts[0].n_samples == 8
        assert info.units == (None,)
    else:
        assert info.participant_ids == (1, "1")
    assert info.recordings[0].eeg_json == {}
    assert info.recordings[0].electrodes is None


def test_nested_metadata_remains_detached_at_every_public_boundary():
    """Named records must not expose mutable source or selection aliases."""
    state = saved_state()
    expected = deepcopy(state)
    info = EEGDatasetInfo(state)
    state["view"]["layouts"][0]["channel_names"].append("C4")
    state["recordings"][0]["channel_metadata"][0]["units"] = "changed"
    exported = info.to_dict()
    exported["dataset"]["custom"]["values"].append(8)
    exported["view"]["layouts"][0]["units"][0] = "V"
    info.cases[0]["source_support"][0] = 4
    info.layouts[0]["custom"]["values"].append(9)
    info.recordings[0].channels[0]["units"] = "changed"
    info.recordings[0].events[0]["custom"]["values"].append(10)
    selected = info[[0]]
    selected._view.layouts[0].channel_names[0] = "changed"
    selected._dataset.description["Name"] = "changed"
    assert info.to_dict() == expected


def test_current_selection_retains_source_scope_with_named_records():
    """Current rows and retained original recordings keep distinct scopes."""
    info = EEGDatasetInfo(saved_state())
    selected_X, selected = info.select(np.zeros((1, 1, 8)), [])
    assert selected_X.shape == (0, 1, 8)
    assert selected.cases == []
    assert selected.participant_ids == ()
    assert selected.dataset["selected_participants"] == [1, "1"]
    assert len(selected.recordings) == 2
    assert EEGDatasetInfo(selected.to_dict()).to_dict() == selected.to_dict()


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("indices", [[1], [1, 0], []])
@pytest.mark.parametrize("diagnostics", [None, [], [None, "a", "b"]])
def test_source_selection_preserves_optional_issues(paired, indices, diagnostics):
    """Source selection filters supplied diagnostics without inventing the key."""
    state = saved_state()
    state.pop("view")
    if diagnostics is not None:
        state["issues"] = [
            dict(recording_id=identity, severity="warning", message="source warning")
            for identity in diagnostics
        ]
    before = deepcopy(state)
    info = EEGDatasetInfo(state)
    if paired:
        X = np.arange(32.0).reshape(2, 1, 16)
        selected_X, selected = info.select(X, indices)
        np.testing.assert_array_equal(selected_X, X[indices])
    else:
        selected = info[indices]
    result = selected.to_dict()
    ids = [state["source_ids"][i] for i in indices]
    assert result["source_ids"] == ids
    assert result["events"] == [
        event
        for identity in ids
        for event in state["events"]
        if event["recording_id"] == identity
    ]
    if diagnostics is None:
        assert "issues" not in result
    else:
        assert result["issues"] == [
            issue
            for issue in state["issues"]
            if issue["recording_id"] is None or issue["recording_id"] in ids
        ]
        result["issues"].append({"message": "changed"})
    assert info.to_dict() == before
    assert state == before
    assert EEGDatasetInfo(selected.to_dict()).to_dict() == selected.to_dict()


@pytest.mark.parametrize(
    "start, origin, step, length, valid",
    [
        (0, 0.0, 1.0, 8, True),
        (0, 0.0, 1.0, 9, False),
        (2, 2.0, 1.0, 6, True),
        (2, 2.0, 1.0, 7, False),
        (2, np.nextafter(2.0, 0.0), 1.0, 1, False),
        (0, 0.0, 0.5, 16, True),
        (0, 0.0, 0.5, 17, False),
        (0, np.nextafter(8.0, 0.0), 1.0, 1, True),
        (0, 8.0, 1.0, 1, False),
        (0, np.nextafter(8.0, np.inf), 1.0, 1, False),
        (0, 8.0 + 5e-9, 1.0, 1, False),
    ],
)
def test_restored_grid_respects_half_open_source_support(
    start, origin, step, length, valid
):
    """Every current sample belongs to [start, stop), including fractional grids."""
    state = saved_state()
    state["view"]["rows"][0]["source_support"] = [start, 8]
    state["view"]["layouts"][0].update(
        n_samples=length,
        source_origin_sample=origin,
        source_sample_step=step,
        sampling_frequency=8.0 / step,
    )
    if valid:
        info = EEGDatasetInfo(state)
        info.validate_layout(np.zeros((1, 1, length)))
        assert info.to_dict() == state
    else:
        with pytest.raises(ValueError, match="sample grid exceeds"):
            EEGDatasetInfo(state)
