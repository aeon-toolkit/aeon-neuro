"""Check evaluation policy and original-source support on small shared cases."""

import json
from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import (
    build_classification_dataset,
    validate_classification_split,
    validate_group_split,
)


@pytest.fixture
def sessions():
    """Return T/E sessions for one person, with both labels in each session."""
    metadata = [
        {
            "recording_id": session,
            "channel_names": ["C3", "C4"],
            "sampling_frequency": 4,
        }
        for session in ("T", "E")
    ]
    cases = [
        {
            "case_id": f"{session}-{label}",
            "recording_id": session,
            "start_sample": label * 4,
            "stop_sample": label * 4 + 4,
            "label": label,
            "group_id": session,
            "participant_id": "P1",
            "session": session,
            "source_identity": session,
            "original_start": label * 4,
            "original_stop": label * 4 + 4,
        }
        for session in ("T", "E")
        for label in (0, 1)
    ]
    return build_classification_dataset(np.arange(32).reshape(2, 2, 8), metadata, cases)


def validate(dataset, partitions=None, **kwargs):
    """Apply the explicitly chosen session protocol, allowing focused overrides."""
    options = {
        "disjoint_by": [("participant_id", "session")],
        "source_fields": ("source_identity", "original_start", "original_stop"),
        "partition_values": {"train": {"session": ["T"]}, "test": {"session": ["E"]}},
        "require_all_classes": ("train", "test"),
    }
    options.update(kwargs)
    return validate_classification_split(
        dataset,
        {"train": [0, 1], "test": [2, 3]} if partitions is None else partitions,
        **options,
    )


def test_explicit_session_policy_and_independent_snapshot(sessions):
    """Accept deliberate within-person sessions without weakening old validation."""
    before = deepcopy(sessions)
    with pytest.raises(ValueError, match="participant_id"):
        validate_group_split(sessions, [0, 1], [2, 3])
    report = validate(sessions, {"train": [1, 0], "test": [3, 2]})
    assert report["partitions"]["train"] == {
        "indices": [1, 0],
        "case_ids": ["T-1", "T-0"],
    }
    json.dumps(report, allow_nan=False)
    report["partitions"]["train"]["indices"][0] = 99
    np.testing.assert_array_equal(sessions.X, before.X)
    assert sessions.cases == before.cases
    with pytest.raises(ValueError, match="crosses"):
        validate(sessions, disjoint_by=[("participant_id",)])


def test_reversed_roles_and_coverage(sessions):
    """Grouping alone cannot enforce session roles or class availability."""
    with pytest.raises(ValueError, match="forbids"):
        validate(sessions, {"train": [2, 3], "test": [0, 1]})
    with pytest.raises(ValueError, match="all eligible classes"):
        validate(sessions, {"train": [0], "test": [1, 2, 3]})
    with pytest.raises(ValueError, match="crosses"):
        validate(
            sessions,
            {"train": [0], "test": [1, 2, 3]},
            partition_values=None,
            require_all_classes=(),
        )


def test_original_source_alias_overlap_and_touching(sessions):
    """Logical recording IDs cannot hide shared original physical support."""
    for row in sessions.cases:
        row["source_identity"] = "same-original"
    with pytest.raises(ValueError, match="Source sample intervals overlap"):
        validate(sessions)
    for row in sessions.cases[2:]:
        row["original_start"] += 8
        row["original_stop"] += 8
    assert validate(sessions)["partitions"]["test"]["indices"] == [2, 3]
    # Two groups can overlap within a partition; only cross-partition reuse fails.
    sessions.cases[0]["original_stop"] = 8
    validate(sessions)
    sessions.cases[0]["original_stop"] = 9
    with pytest.raises(ValueError, match="overlap"):
        validate(sessions)


def test_nested_interval_is_not_hidden_by_shorter_predecessor(sessions):
    """A long interval must remain active after a shorter same-partition interval."""
    for row in sessions.cases:
        row["source_identity"] = "same-original"
    sessions.cases[0].update(original_start=0, original_stop=20)
    sessions.cases[1].update(original_start=4, original_stop=8)
    sessions.cases[2].update(original_start=10, original_stop=12)
    sessions.cases[3].update(original_start=12, original_stop=14)
    with pytest.raises(ValueError, match="overlap"):
        validate(sessions)


def test_scope_prevents_outer_test_assignment(sessions):
    """Inner validation covers precisely the supplied parent training scope."""
    for index, row in enumerate(sessions.cases):
        row["trial"] = index
    options = {
        "eligible_indices": [0, 1],
        "disjoint_by": [("trial",)],
        "partition_values": None,
        "require_all_classes": (),
    }
    report = validate(sessions, {"train": [0], "validation": [1]}, **options)
    assert report["eligible_indices"] == [0, 1]
    with pytest.raises(ValueError, match="eligible cases exactly once"):
        validate(sessions, {"train": [0], "validation": [2]}, **options)


@pytest.mark.parametrize(
    "bad",
    [
        [0, 0],
        [0, 4],
        [-1, 0],
        [0.0, 1.0],
        [False, 1],
        [np.bool_(False), 1],
        [[0, 1]],
        [],
    ],
)
def test_invalid_indices_and_scope(sessions, bad):
    """Reject invalid indices, including booleans hidden in mixed integer lists."""
    with pytest.raises(ValueError, match="indices"):
        validate(sessions, {"train": bad, "test": [2, 3]})
    with pytest.raises(ValueError, match="indices"):
        validate(sessions, eligible_indices=bad)


@pytest.mark.parametrize(
    "partitions",
    [
        {"train": [0, 1], "test": [1, 2, 3]},
        {"train": [0, 1], "test": [2]},
        {"train": [0, 1, 2, 3]},
    ],
)
def test_invalid_partition_assignment(sessions, partitions):
    """No eligible case may be missing, duplicated or assigned to only one role."""
    with pytest.raises(ValueError, match="partitions|eligible cases"):
        validate(sessions, partitions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("participant_id", None),
        ("session", "n/a"),
        ("source_identity", ""),
        ("original_start", -1),
        ("original_start", False),
        ("original_stop", 0),
        ("original_stop", 4.5),
    ],
)
def test_missing_identity_and_invalid_support(sessions, field, value):
    """Missing policy metadata and invalid original bounds cannot pass validation."""
    sessions.cases[0][field] = value
    with pytest.raises(ValueError):
        validate(sessions)


@pytest.mark.parametrize(
    "update",
    [
        {"disjoint_by": []},
        {"disjoint_by": ["participant_id"]},
        {"disjoint_by": [("session", "session")]},
        {"disjoint_by": [("session",), ("session",)]},
        {"source_fields": ["source_identity", "original_start"]},
        {"require_all_classes": "train"},
        {"require_all_classes": ["validation"]},
        {"partition_values": {"validation": {"session": ["T"]}}},
        {"partition_values": {"train": {"session": "T"}}},
    ],
)
def test_invalid_policy(sessions, update):
    """Malformed policy must fail instead of silently weakening the checks."""
    with pytest.raises(ValueError):
        validate(sessions, **update)


@pytest.mark.parametrize("mutation", ["labels", "groups", "case_id", "alignment"])
def test_mutated_shared_dataset(sessions, mutation):
    """Direct dataclass mutation must not bypass case/array alignment checks."""
    if mutation == "labels":
        sessions.cases[0]["label"] = 99
    elif mutation == "groups":
        sessions.groups[0] = "other"
    elif mutation == "case_id":
        sessions.cases[1]["case_id"] = sessions.cases[0]["case_id"]
    else:
        sessions.X = sessions.X[:2]
    with pytest.raises(ValueError, match="agree|unique|aligned"):
        validate(sessions)


def test_three_partitions_and_multiple_identity_rules(sessions):
    """Each declared grouping rule applies independently across every partition."""
    for index, row in enumerate(sessions.cases):
        row["trial"] = index
        row["block"] = index
    options = {
        "disjoint_by": [("trial",), ("block",)],
        "partition_values": None,
        "require_all_classes": (),
    }
    validate(sessions, {"train": [0, 1], "validation": [2], "test": [3]}, **options)
    sessions.cases[3]["block"] = 0
    with pytest.raises(ValueError, match="crosses"):
        validate(sessions, {"train": [0, 1], "validation": [2], "test": [3]}, **options)
