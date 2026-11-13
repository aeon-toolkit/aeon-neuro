"""Keep estimator-facing information aligned without rewriting acquisition facts."""

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


def test_case_view_preserves_flat_source_session_and_run_fields(source):
    """BIDS descriptor identities remain available when a binding omits them."""
    X, original = load_bids_dataset(source[0])
    dataset = _event_dataset(X, original)
    recorded = dataset.metadata["provenance"]["source"]["recordings"]
    recorded[0]["session"], recorded[0]["run"] = "visit-1", "run-2"
    assert dataset.info.cases[0]["session"] == "visit-1"
    assert dataset.info.cases[0]["run"] == "run-2"


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
