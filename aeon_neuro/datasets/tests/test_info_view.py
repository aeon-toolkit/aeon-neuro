"""Keep estimator-facing information aligned without rewriting acquisition facts."""

import json
from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import (
    EEGDatasetInfo,
    build_classification_dataset,
    load_bids_dataset,
    make_event_cases,
)
from aeon_neuro.datasets.tests import test_dataset_info as _fixtures

source = _fixtures.source


def _event_dataset(X, info):
    events = info.events_for_cases()
    cases = make_event_cases(X, info, events, tmin=0.5, tmax=1.5)
    for row in cases:
        row["channel_names"] = ["C4", "C3"]
    return build_classification_dataset(X, info, cases)


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


def test_object_builder_exposes_cases_and_preserves_original_metadata(source):
    """Epoch information describes the selected layout and retains source facts."""
    X, original = load_bids_dataset(source[0])
    dataset = _event_dataset(X, original)
    info = dataset.info
    assert isinstance(info, EEGDatasetInfo) and len(info) == len(dataset.X) == 4
    assert info.channel_names == ("C4", "C3")
    assert info.participant_ids == ("sub-01", "sub-01", "sub-02", "sub-02")
    assert info.groups.tolist() == dataset.groups.tolist()
    assert info.recordings[0].channel_names == ("C3", "C4")
    assert info.cases[0]["source_support"] == [48, 80]
    assert info.units == ("V", "V")  # decoded MNE arrays, not BDF sidecar microvolts
    info.validate_layout(dataset.X)
    assert info.case_table["case_id"].tolist() == [r["case_id"] for r in dataset.cases]
    assert len(info.participants_frame) == 3


def test_event_convenience_reuses_literal_source_values(source):
    """Event rows are usable without rereading files or inferring a target mapping."""
    _, info = load_bids_dataset(source[0])
    rows = info.events_for_cases()
    assert [r["label"] for r in rows] == ["left", "right", "left", "right"]
    assert [r["group_id"] for r in rows] == ["sub-01"] * 2 + ["sub-02"] * 2
    assert rows[0]["onset"] == 1
    with pytest.raises(ValueError, match="label"):
        info.events_for_cases(label_column="missing")
    assert len(info.recordings[0].events_frame) == 2
    assert info.recordings[0].channels_frame["name"].tolist() == ["C3", "C4"]


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


def test_case_view_preserves_flat_source_session_and_run_fields(source):
    """BIDS descriptor identities remain available when a binding omits them."""
    X, original = load_bids_dataset(source[0])
    dataset = _event_dataset(X, original)
    recorded = dataset.metadata["provenance"]["source"]["recordings"]
    recorded[0]["session"], recorded[0]["run"] = "visit-1", "run-2"
    assert dataset.info.cases[0]["session"] == "visit-1"
    assert dataset.info.cases[0]["run"] == "run-2"


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


def test_explicit_case_participant_is_retained(source):
    """A case's declared person can differ from the recording's BIDS subject."""
    root, paths = source
    for i, path in enumerate(paths):
        events = path.with_name(path.name.replace("_eeg.bdf", "_events.tsv"))
        lines = events.read_text().splitlines()
        events.write_text(
            lines[0]
            + "\tactor_id\n"
            + "".join(row + f"\tperson-{i+1}\n" for row in lines[1:]),
            encoding="utf-8",
        )
    X, original = load_bids_dataset(root)
    events = original.events_for_cases()
    for row in events:
        row["participant_id"] = row["source_event"]["actor_id"]
    cases = make_event_cases(X, original, events, tmin=0.5, tmax=1.5)
    dataset = build_classification_dataset(X, original, cases)
    assert dataset.info.participant_ids == (
        "person-1",
        "person-1",
        "person-2",
        "person-2",
    )
    assert dataset.info.recordings[0].metadata["participant_id"] == "sub-01"
