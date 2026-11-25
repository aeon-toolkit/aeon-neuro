"""Validate detached EEG information without readers or estimators."""

import json
from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import EEGDatasetInfo


def example_info():
    """Explicit identities avoid inferring subjects from arbitrary case order."""
    X = np.arange(4 * 2 * 32, dtype=float).reshape(4, 2, 32)
    info = EEGDatasetInfo.from_arrays(
        X,
        sfreq=32,
        channel_names=["C3", "C4"],
        units=["V", "V"],
        subjects=["01", "01", "02", "02"],
        case_ids=["a", "b", "c", "d"],
    )
    return X, info


def test_case_view_selection_round_trip_and_ownership():
    """CV subsets retain exact row identities, metadata and detached ownership."""
    X, info = example_info()
    selected, view = info.select(X, [3, 0])
    assert view.case_ids == ("d", "a")
    assert view.participant_ids == ("02", "01")
    assert view.groups.tolist() == ["02", "01"]
    assert view.sfreq == 32 and view.channel_names == ("C3", "C4")
    assert view.units == ("V", "V")
    assert info[[False, True, False, True]].case_ids == ("b", "d")
    assert view.subject_of(0) == "02"
    restored = EEGDatasetInfo(json.loads(json.dumps(view.to_dict())))
    restored.validate_layout(selected)
    assert restored.to_dict() == view.to_dict()
    selected[:] = -1
    assert np.all(X >= 0)
    report = view.to_dict()
    report["view"]["rows"][0]["participant_id"] = "changed"
    assert view.participant_ids == ("02", "01")
    assert len(info[[]]) == 0


@pytest.mark.parametrize("fault", ["count", "channels", "samples", "dtype", "nan"])
def test_case_view_rejects_wrong_arrays(fault):
    """A metadata object cannot quietly describe an incompatible collection."""
    X, info = example_info()
    if fault == "count":
        X = X[:-1]
    elif fault == "channels":
        X = X[:, :1]
    elif fault == "samples":
        X = X[:, :, :-1]
    elif fault == "dtype":
        X = X.astype(complex)
    else:
        X[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        info.validate_layout(X)


@pytest.mark.parametrize("fault", ["sfreq", "names", "subjects", "units", "ids"])
def test_array_metadata_requires_explicit_valid_alignment(fault):
    """Unknown facts stay explicit; mismatched or duplicated identities fail."""
    X, _ = example_info()
    options = dict(
        sfreq=32,
        channel_names=["C3", "C4"],
        subjects=["01"] * 4,
        units=["V", "V"],
        case_ids=["a", "b", "c", "d"],
    )
    options[
        {
            "sfreq": "sfreq",
            "names": "channel_names",
            "subjects": "subjects",
            "units": "units",
            "ids": "case_ids",
        }[fault]
    ] = {
        "sfreq": True,
        "names": ["C3", "C3"],
        "subjects": ["01"],
        "units": ["V"],
        "ids": ["a"] * 4,
    }[
        fault
    ]
    with pytest.raises(ValueError):
        EEGDatasetInfo.from_arrays(X, **options)


@pytest.mark.parametrize(
    "fault", ["version", "row_count", "rate", "support", "duplicate"]
)
def test_saved_current_layout_is_validated(fault):
    """Serialized derived state is checked rather than trusted during restoration."""
    _, info = example_info()
    report = deepcopy(info.to_dict())
    view = report["view"]
    if fault == "version":
        view["version"] = True
    elif fault == "row_count":
        view["layouts"].pop()
    elif fault == "rate":
        view["layouts"][0]["sampling_frequency"] = -1
    elif fault == "support":
        view["rows"][0]["source_support"] = [5, 2]
    else:
        view["rows"][1]["case_id"] = view["rows"][0]["case_id"]
    with pytest.raises(ValueError):
        EEGDatasetInfo(report)


def test_array_subject_and_group_identifiers_remain_portable():
    """Ordinary NumPy group arrays must serialize without losing identifier types."""
    X, _ = example_info()
    info = EEGDatasetInfo.from_arrays(
        X,
        sfreq=32,
        channel_names=np.array(["C3", "C4"]),
        subjects=np.array([1, 1, 2, 2]),
        groups=np.array([5, 6, 7, 8]),
    )
    restored = EEGDatasetInfo(json.loads(json.dumps(info.to_dict())))
    assert restored.participant_ids == (1, 1, 2, 2)
    assert restored.groups.tolist() == [5, 6, 7, 8]


def test_subject_lookup_does_not_scan_the_complete_case_collection():
    """A per-case lookup should stay constant-time as a catalogue grows."""
    X = np.zeros((1000, 1, 4))
    info = EEGDatasetInfo.from_arrays(
        X, sfreq=4, channel_names=["sensor"], subjects=list(range(1000))
    )

    class CountedRows(list):
        visits = 0

        def __iter__(self):
            self.visits += len(self)
            return super().__iter__()

        def __getitem__(self, index):
            self.visits += 1
            return super().__getitem__(index)

    rows = CountedRows(info._view.rows)
    info._view.rows = rows
    assert info.subject_of(999) == 999
    # Count accesses instead of timing, which would depend on the test machine.
    assert rows.visits <= 1


def test_explicit_source_metadata_needs_no_inspector_and_stays_detached():
    """Original source facts and selected rows have different lifetimes."""
    records = [
        dict(
            recording_id="a",
            participant_id="01",
            channel_names=["C3"],
            sampling_frequency=32,
            n_samples=8,
        ),
        dict(
            recording_id="b",
            participant_id="02",
            channel_names=["C3"],
            sampling_frequency=32,
            n_samples=8,
        ),
    ]
    dataset = dict(
        description={"Name": "study"},
        participants={
            "columns": ["participant_id"],
            "rows": [{"participant_id": "01"}, {"participant_id": "02"}],
        },
    )
    info = EEGDatasetInfo(recordings=records, dataset=dataset)
    X = np.arange(16.0).reshape(2, 1, 8)
    selected, view = info.select(X, [1])
    records[1]["channel_names"][0] = "changed"
    dataset["description"]["Name"] = "changed"
    assert info.channel_names == ("C3",)
    assert view.description == {"Name": "study"}
    assert view.participant_ids == ("02",)
    assert len(view.participants["rows"]) == 2
    assert view.recordings[0].channels is None
    assert view.recordings[0].electrodes is None
    np.testing.assert_array_equal(selected, X[[1]])
    restored = EEGDatasetInfo(json.loads(json.dumps(view.to_dict())))
    restored.validate_layout(selected)


@pytest.mark.parametrize("selector", [[0, 0], [-1], [4], [True], [True, 1]])
def test_invalid_selection_does_not_modify_parent(selector):
    """Bad selectors cannot duplicate identities or silently change alignment."""
    X, info = example_info()
    before = info.to_dict()
    with pytest.raises(ValueError):
        info.select(X, selector)
    assert info.to_dict() == before


def test_unequal_lengths_and_unknown_facts_survive_selection():
    """Variable-length arrays keep their layouts without fabricated identities."""
    X = [np.zeros((1, 8)), np.ones((1, 12))]
    info = EEGDatasetInfo.from_arrays(X, sfreq=32, channel_names=["C3"])
    selected, view = info.select(X, [1, 0])
    assert [v.shape for v in selected] == [(1, 12), (1, 8)]
    assert view.participant_ids == (None, None)
    assert view.units == (None,)
    assert view.recordings == ()
    selected[0][:] = 5
    assert np.all(X[1] == 1)


@pytest.mark.parametrize(
    "subjects", [["01", None], [None, "01"], [None, None], [1, "1"]]
)
def test_source_subject_scope_preserves_unknowns_and_identifier_types(subjects):
    """Known participant scope must not reorder or invent aligned source facts."""
    records = [
        dict(
            recording_id=f"r{i}",
            participant_id=subject,
            channel_names=["C3"],
            sampling_frequency=32,
            n_samples=8,
        )
        for i, subject in enumerate(subjects)
    ]
    info = EEGDatasetInfo(recordings=records)
    X = np.zeros((2, 1, 8))
    assert info.participant_ids == tuple(subjects)
    assert None not in info.to_dict()["dataset"]["selected_participants"]
    assert set(info.to_dict()["dataset"]["selected_participants"]) == set(subjects) - {
        None
    }
    restored = EEGDatasetInfo(json.loads(json.dumps(info.to_dict())))
    assert restored.participant_ids == tuple(subjects)
    for indices in ([1, 0], [0], [1], []):
        selected, subset = restored.select(X, indices)
        assert subset.participant_ids == tuple(subjects[i] for i in indices)
        assert set(subset.to_dict()["dataset"]["selected_participants"]) == {
            subjects[i] for i in indices if subjects[i] is not None
        }
        subset.validate_layout(selected)
