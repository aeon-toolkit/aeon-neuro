"""Exercise authored problems against independently sliced loaded recordings."""

import json
from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import (
    load_classification_problem,
    validate_classification_split,
)


@pytest.fixture
def recordings():
    """Make every source channel and sample distinguishable."""
    X = np.arange(96, dtype=float).reshape(2, 3, 16)
    metadata = [
        {
            "recording_id": f"R{i}",
            "subject": f"P{i}",
            "session": "S1",
            "cohort": "control" if i == 0 else "patient",
            "channel_names": ["C4", "Cz", "C3"],
            "sampling_frequency": 4,
            "units": "V",
            "processing": [{"name": "acquisition"}],
        }
        for i in range(2)
    ]
    return X, metadata


def definition(mode="fixed_windows"):
    """Keep scientific choices independent of physical source column names."""
    return {
        "format_version": 1,
        "id": "cohort" if mode == "fixed_windows" else "imagery",
        "version": "1",
        "question": "Distinguish the declared classes from one EEG case",
        "class_labels": ["left", "right"],
        "channels": ["C3", "C4"],
        "segmentation": (
            {"mode": mode, "window_size": 1, "start": 0.5, "stop": 3.5}
            if mode == "fixed_windows"
            else {"mode": mode, "tmin": -0.25, "tmax": 0.75}
        ),
    }


def binding(mode="fixed_windows"):
    """Supply exact source scopes, class meanings and split identity fields."""
    problem = definition(mode)
    result = {
        "format_version": 1,
        "problem": {key: problem[key] for key in ("id", "version")},
        "target": {
            "source": "recording" if mode == "fixed_windows" else "event",
            "column": "cohort" if mode == "fixed_windows" else "condition",
            "mapping": (
                {"control": "left", "patient": "right"}
                if mode == "fixed_windows"
                else {"11": "left", "22": "right"}
            ),
        },
        "group": {"source": "recording", "column": "subject"},
        "case_fields": {
            "participant_id": {"source": "recording", "column": "subject"},
            "session": {"source": "recording", "column": "session"},
        },
    }
    if mode == "event_epochs":
        result["events"] = {"onset": "cue", "id": "trial"}
    return result


def events():
    """Retain source ordering even when event IDs are not lexicographically sorted."""
    return [
        {
            "recording_id": recording,
            "cue": str(onset),
            "trial": trial,
            "condition": code,
            "response": "yes" if code == 11 else "no",
        }
        for recording in ("R0", "R1")
        for onset, trial, code in ((0.5, "T2", 11), (2.5, "T1", 22))
    ]


def test_fixed_windows_exact_samples_provenance_and_independent_outputs(recordings):
    """Define a cohort problem without manually constructing each case row."""
    X, metadata = recordings
    problem, fields = definition(), binding()
    original = deepcopy((X, metadata, problem, fields))
    data = load_classification_problem(X, metadata, problem, fields)
    expected = [
        X[i, [2, 0], start : start + 4] for i in range(2) for start in (2, 6, 10)
    ]
    np.testing.assert_array_equal(data.X, expected)
    assert data.y.tolist() == ["left"] * 3 + ["right"] * 3
    assert data.groups.tolist() == ["P0"] * 3 + ["P1"] * 3
    assert data.metadata["channel_names"] == ["C3", "C4"]
    assert data.cases[3]["participant_id"] == "P1"
    assert data.cases[0]["source_sampling_frequency"] == 4
    assert data.metadata["recordings"][0]["units"] == "V"
    resolved = data.metadata["problem"]
    assert resolved["definition"]["segmentation"]["step_size"] == 1
    assert resolved["binding"]["target"]["on_unmapped"] == "raise"
    data.X[0, 0, 0] = -1
    data.metadata["recordings"][0]["processing"][0]["name"] = "changed"
    resolved["definition"]["question"] = "changed"
    resolved["binding"]["target"]["mapping"]["control"] = "changed"
    np.testing.assert_array_equal(X, original[0])
    assert (metadata, problem, fields) == original[1:]


def test_event_epochs_exact_timing_source_rows_and_explicit_split(recordings):
    """Offsets are half-open on the source grid, with stable original event IDs."""
    X, metadata = recordings
    source_events = events()
    before = deepcopy(source_events)
    data = load_classification_problem(
        X,
        metadata,
        definition("event_epochs"),
        binding("event_epochs"),
        events=source_events,
    )
    np.testing.assert_array_equal(
        data.X, [X[i, [2, 0], start : start + 4] for i in range(2) for start in (1, 9)]
    )
    assert data.y.tolist() == ["left", "right", "left", "right"]
    assert [c["event_id"] for c in data.cases] == ["T2", "T1", "T2", "T1"]
    assert [c["source_event_row"] for c in data.cases] == [0, 1, 2, 3]
    assert data.cases[0]["source_event"] == source_events[0]
    split = validate_classification_split(
        data,
        {"train": [0, 1], "test": [2, 3]},
        disjoint_by=[("participant_id",)],
        source_fields=("recording_id", "start_sample", "stop_sample"),
        require_all_classes=("train", "test"),
    )
    assert split["partitions"]["test"]["case_ids"] == [
        c["case_id"] for c in data.cases[2:]
    ]
    data.cases[0]["source_event"]["condition"] = "changed"
    assert source_events == before


def test_one_dataset_supports_two_different_targets(recordings):
    """Changing the declared target can reuse exactly the same observations."""
    X, metadata = recordings
    problem, fields = definition("event_epochs"), binding("event_epochs")
    first = load_classification_problem(X, metadata, problem, fields, events=events())
    problem.update(id="response", question="Classify the recorded response")
    problem["class_labels"] = ["yes", "no"]
    fields["problem"]["id"] = "response"
    fields["target"].update(column="response", mapping={"yes": "yes", "no": "no"})
    second = load_classification_problem(X, metadata, problem, fields, events=events())
    np.testing.assert_array_equal(first.X, second.X)
    assert [c["case_id"] for c in first.cases] == [c["case_id"] for c in second.cases]
    assert first.y.tolist() == ["left", "right", "left", "right"]
    assert second.y.tolist() == ["yes", "no", "yes", "no"]
    assert (
        first.metadata["problem"]["definition_sha256"]
        != second.metadata["problem"]["definition_sha256"]
    )


def test_one_definition_supports_two_source_bindings_and_json_files(
    recordings, tmp_path
):
    """The Python and JSON routes resolve identically across physical columns/codes."""
    X, metadata = recordings
    problem, first_binding = definition("event_epochs"), binding("event_epochs")
    original = load_classification_problem(
        X, metadata, problem, first_binding, events=events()
    )
    changed_events = events()
    for row in changed_events:
        row["code"] = "L" if row.pop("condition") == 11 else "R"
    second_binding = deepcopy(first_binding)
    second_binding["target"].update(column="code", mapping={"L": "left", "R": "right"})
    paths = [tmp_path / "problem.json", tmp_path / "binding.json"]
    for path, value in zip(paths, (problem, second_binding)):
        path.write_text(json.dumps(value))
    changed = load_classification_problem(X, metadata, *paths, events=changed_events)
    np.testing.assert_array_equal(changed.X, original.X)
    np.testing.assert_array_equal(changed.y, original.y)
    assert (
        changed.metadata["problem"]["definition_sha256"]
        == original.metadata["problem"]["definition_sha256"]
    )
    assert (
        changed.metadata["problem"]["binding_sha256"]
        != original.metadata["problem"]["binding_sha256"]
    )
    same = load_classification_problem(
        X, metadata, problem, second_binding, events=changed_events
    )
    assert same.metadata["problem"] == changed.metadata["problem"]


def test_eligibility_unmapped_events_and_outside_recording_are_auditable(recordings):
    """Every excluded candidate retains source identity and a distinct reason."""
    X, metadata = recordings
    problem, fields = definition("event_epochs"), binding("event_epochs")
    rows = events()
    rows[0].update(included=False, exclusion_reason="source_quality")
    rows[0].pop("condition")
    rows[1]["condition"] = "unrelated"
    rows[2]["cue"] = "0"
    with pytest.raises(ValueError, match="Unmapped"):
        load_classification_problem(X, metadata, problem, fields, events=rows)
    fields["target"]["on_unmapped"] = "exclude"
    with pytest.raises(ValueError, match="outside recording"):
        load_classification_problem(X, metadata, problem, fields, events=rows)
    problem["segmentation"]["on_out_of_bounds"] = "exclude"
    data = load_classification_problem(X, metadata, problem, fields, events=rows)
    assert [c["source_event_row"] for c in data.cases] == [3]
    assert [c["exclusion_reason"] for c in data.exclusions] == [
        "source_quality",
        "unmapped_target",
        "outside_recording",
    ]
    assert [c["source_event"] for c in data.exclusions] == rows[:3]
    assert data.exclusions[0]["label"] is None


def test_unmapped_recordings_keep_excluded_window_inventory(recordings):
    """A deliberate cohort exclusion cannot silently drop its candidate windows."""
    X, metadata = recordings
    metadata[0]["cohort"] = "other"
    fields = binding()
    fields["target"]["on_unmapped"] = "exclude"
    data = load_classification_problem(X, metadata, definition(), fields)
    assert len(data.exclusions) == 3
    assert all(
        c["label"] is None and c["exclusion_reason"] == "unmapped_target"
        for c in data.exclusions
    )
    assert data.y.tolist() == ["right"] * 3


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "version",
        "binding_version",
        "identity",
        "mapping",
        "classes",
        "mode",
        "source",
        "reserved",
        "finite",
        "time",
        "channels",
    ],
)
def test_invalid_definitions_and_bindings_fail_explicitly(recordings, change):
    """Unsupported fields and scientific choices never fall back to guesses."""
    X, metadata = recordings
    problem, fields = definition(), binding()
    if change == "unknown":
        problem["preprocessing"] = {"filter": "automatic"}
    elif change == "version":
        problem["format_version"] = True
    elif change == "binding_version":
        fields["format_version"] = 2
    elif change == "identity":
        fields["problem"]["version"] = "2"
    elif change == "mapping":
        fields["target"]["mapping"].pop("patient")
    elif change == "classes":
        problem["class_labels"] = ["left", "left"]
    elif change == "mode":
        problem["segmentation"]["mode"] = "guess"
    elif change == "source":
        fields["group"]["source"] = "event"
    elif change == "reserved":
        fields["case_fields"]["label"] = fields["group"]
    elif change == "finite":
        problem["segmentation"]["window_size"] = np.inf
    elif change == "time":
        problem["segmentation"]["window_size"] = 0.1
    else:
        problem["channels"] = ["missing"]
    with pytest.raises(ValueError):
        load_classification_problem(X, metadata, problem, fields)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "identity",
        "unknown_recording",
        "duplicate",
        "onset",
        "off_grid",
        "eligibility",
        "all_excluded",
    ],
)
def test_bad_event_sources_fail_without_reinterpreting_them(recordings, change):
    """Catch incomplete identities, timing and eligibility before returning cases."""
    X, metadata = recordings
    rows = events()
    if change == "missing":
        rows[0].pop("condition")
    elif change == "identity":
        metadata[0]["subject"] = "n/a"
    elif change == "unknown_recording":
        rows[0]["recording_id"] = "R9"
    elif change == "duplicate":
        rows[1]["trial"] = rows[0]["trial"]
    elif change == "onset":
        rows[0]["cue"] = True
    elif change == "off_grid":
        rows[0]["cue"] = 0.1
    elif change == "eligibility":
        rows[0]["included"] = False
    else:
        for row in rows:
            row.update(included=False, exclusion_reason="source_quality")
    with pytest.raises(ValueError):
        load_classification_problem(
            X,
            metadata,
            definition("event_epochs"),
            binding("event_epochs"),
            events=rows,
        )


def test_duplicate_json_keys_and_unused_events_are_rejected(recordings, tmp_path):
    """Malformed persisted choices cannot silently replace each other."""
    X, metadata = recordings
    path = tmp_path / "problem.json"
    path.write_text('{"format_version":1,"format_version":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_classification_problem(X, metadata, path, binding())
    with pytest.raises(ValueError, match="only used"):
        load_classification_problem(
            X, metadata, definition(), binding(), events=events()
        )


def test_event_id_types_and_reordering_do_not_change_source_identity(recordings):
    """Numeric and textual event IDs remain distinct after JSON-safe encoding."""
    X, metadata = recordings
    rows = events()[:2]
    rows[0]["trial"], rows[1]["trial"] = 1, "1"
    problem, fields = definition("event_epochs"), binding("event_epochs")
    first = load_classification_problem(X, metadata, problem, fields, events=rows)
    second = load_classification_problem(
        X, metadata, problem, fields, events=list(reversed(rows))
    )
    ids = [case["case_id"] for case in first.cases]
    assert len(set(ids)) == 2
    assert [case["case_id"] for case in second.cases] == list(reversed(ids))
    np.testing.assert_array_equal(second.X, first.X[::-1])


def test_source_scope_is_explicit_when_columns_overlap(recordings):
    """An event column cannot shadow a recording-level target or participant."""
    X, metadata = recordings
    rows = events()
    for row in rows:
        row.update(subject="wrong-person", cohort="unmapped-event-cohort")
    fields = binding("event_epochs")
    fields["target"] = binding()["target"]
    data = load_classification_problem(
        X, metadata, definition("event_epochs"), fields, events=rows
    )
    assert data.y.tolist() == ["left", "left", "right", "right"]
    assert data.groups.tolist() == ["P0", "P0", "P1", "P1"]


def test_python_mapping_keys_cannot_collide_during_json_normalisation(recordings):
    """The dictionary route must not silently overwrite code meanings."""
    X, metadata = recordings
    fields = binding("event_epochs")
    fields["target"]["mapping"] = {11: "left", "11": "right", "22": "right"}
    with pytest.raises(ValueError, match="JSON"):
        load_classification_problem(
            X, metadata, definition("event_epochs"), fields, events=events()
        )


@pytest.mark.parametrize(
    "scope,missing,unsupported",
    [
        ("problem", ["id", "question"], ["name", "title"]),
        ("binding", ["group", "target"], ["groups", "labels"]),
        ("problem.segmentation", ["tmin", "tmax"], ["start", "stop"]),
        ("binding.target", ["mapping", "column"], ["labels", "field"]),
        ("binding.group", ["column"], ["field"]),
        ("binding.events", ["onset", "id"], ["sample", "identity"]),
        ("binding.case_fields.session", ["column"], ["field"]),
    ],
)
def test_schema_errors_name_each_missing_and_unsupported_field(
    recordings, scope, missing, unsupported
):
    """A typo should identify all affected keys in the offending object."""
    X, metadata = recordings
    problem, fields = definition("event_epochs"), binding("event_epochs")
    documents = {"problem": problem, "binding": fields}
    value = documents
    for name in scope.split("."):
        value = value[name]
    for name in missing:
        value.pop(name)
    for name in unsupported:
        value[name] = None
    with pytest.raises(ValueError) as caught:
        load_classification_problem(X, metadata, problem, fields, events=events())
    assert str(caught.value) == (
        f"Invalid fields in {scope}: missing {sorted(missing)}; "
        f"unsupported {sorted(unsupported)}"
    )


@pytest.mark.parametrize("scope", ["binding.target", "binding.group", "binding.events"])
def test_schema_errors_identify_non_dictionary_objects(recordings, scope):
    """Malformed nested values need a type error rather than a list of key errors."""
    X, metadata = recordings
    problem, fields = definition("event_epochs"), binding("event_epochs")
    fields[scope.split(".")[1]] = None
    with pytest.raises(ValueError, match=rf"{scope} must be a dictionary"):
        load_classification_problem(X, metadata, problem, fields, events=events())
