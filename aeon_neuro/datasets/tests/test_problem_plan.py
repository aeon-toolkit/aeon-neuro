"""Verify metadata-only planning and deferred selected source access."""

import json
from copy import deepcopy

import mne
import numpy as np
import pytest

from aeon_neuro.datasets import (
    load_classification_problem,
    materialize_classification_problem,
    plan_classification_problem,
    resolve_classification_problem,
)
from aeon_neuro.datasets.tests.test_problem import binding, definition, events


@pytest.fixture
def sources():
    """Pair small physical signals with descriptors that need no array access."""
    X = np.arange(96, dtype=float).reshape(2, 3, 16)
    metadata = [
        {
            "recording_id": f"R{i}",
            "subject": f"P{i}",
            "session": "S1",
            "cohort": "control" if i == 0 else "patient",
            "channel_names": ["C4", "Cz", "C3"],
            "sampling_frequency": 4,
            "n_samples": 16,
            "units": "V",
            "release": "pinned-release",
        }
        for i in range(2)
    ]
    return X, metadata


def make_opener(X, trace, monkeypatch, *, change=None):
    """Observe real public Raw reads and closure, including exceptional paths."""

    def open_recording(descriptor):
        trace.append(("open", descriptor["recording_id"]))
        index = int(descriptor["recording_id"][1:])
        names = list(descriptor["channel_names"])
        sfreq = descriptor["sampling_frequency"]
        signal = X[index].copy()
        if change == "channels":
            names = names[::-1]
        elif change == "rate":
            sfreq += 1
        elif change == "length":
            signal = signal[:, :-1]
        raw = mne.io.RawArray(
            signal, mne.create_info(names, sfreq, "eeg"), verbose=False
        )
        original = raw.get_data

        def get_data(*, picks, start, stop):
            trace.append(("read", descriptor["recording_id"], start, stop, list(picks)))
            result = original(picks=picks, start=start, stop=stop)
            if change == "nonfinite":
                result[0, 0] = np.nan
            elif change == "shape":
                result = result[:, :-1]
            return result

        monkeypatch.setattr(raw, "get_data", get_data)
        monkeypatch.setattr(
            raw, "close", lambda: trace.append(("close", descriptor["recording_id"]))
        )
        return raw

    return open_recording


def test_definition_and_128gib_catalogue_plan_need_no_signal_files(
    sources, monkeypatch
):
    """Large sizes are ordinary metadata, never a reason to allocate source arrays."""
    _, metadata = sources
    for i, item in enumerate(metadata):
        item.update(
            n_samples=2**30,
            source_size_bytes=64 * 2**30,
            url=f"https://example.invalid/unavailable-{i}.bdf",
        )
    resolved = resolve_classification_problem(definition(), binding())
    assert resolved["definition"]["id"] == "cohort"
    problem = definition()
    # Small previews can be authored even when the source spans years of samples.
    plan = plan_classification_problem(metadata, problem, binding())
    assert sum(row["source_size_bytes"] for row in plan["recordings"]) == 128 * 2**30
    assert plan["dense_shape"] == [6, 2, 4]
    assert plan["estimated_case_bytes_float64"] == 6 * 2 * 4 * 8
    assert "not verified" in plan["source_verification"]
    assert "X" not in plan


@pytest.mark.parametrize("mode", ["fixed_windows", "event_epochs"])
def test_planned_materialization_matches_eager_cases_exactly(
    sources, monkeypatch, mode
):
    """Sharing case rules preserves every original sample, target and identity."""
    X, metadata = sources
    event_rows = events() if mode == "event_epochs" else None
    plan = plan_classification_problem(
        metadata, definition(mode), binding(mode), events=event_rows
    )
    trace = []
    dataset = materialize_classification_problem(
        json.loads(json.dumps(plan)), make_opener(X, trace, monkeypatch)
    )
    eager = load_classification_problem(
        X, metadata, definition(mode), binding(mode), events=event_rows
    )
    np.testing.assert_array_equal(dataset.X, eager.X)
    np.testing.assert_array_equal(dataset.y, eager.y)
    np.testing.assert_array_equal(dataset.groups, eager.groups)
    assert dataset.cases == eager.cases
    assert dataset.exclusions == eager.exclusions
    assert [entry for entry in trace if entry[0] == "open"] == [
        ("open", "R0"),
        ("open", "R1"),
    ]
    assert len([entry for entry in trace if entry[0] == "read"]) == len(eager.cases)
    assert [entry for entry in trace if entry[0] == "close"] == [
        ("close", "R0"),
        ("close", "R1"),
    ]


def test_only_selected_source_is_opened_and_only_selected_interval_read(
    sources, monkeypatch
):
    """Unselected recordings can be absent from both disk and the local cache."""
    X, metadata = sources
    plan = plan_classification_problem(metadata, definition(), binding())
    before = deepcopy(plan)
    selected = plan["cases"][-1]
    trace = []
    result = materialize_classification_problem(
        plan, make_opener(X, trace, monkeypatch), case_ids=[selected["case_id"]]
    )
    assert trace == [
        ("open", "R1"),
        ("read", "R1", 10, 14, ["C3", "C4"]),
        ("close", "R1"),
    ]
    np.testing.assert_array_equal(result.X[0], X[1, [2, 0], 10:14])
    assert result.cases[0]["row"] == 0
    assert result.metadata["materialization"]["unselected_cases"] == 5
    assert plan == before


def test_selected_membership_preserves_canonical_plan_order(sources, monkeypatch):
    """Requested membership must not silently reorder an experiment's rows."""
    X, metadata = sources
    plan = plan_classification_problem(metadata, definition(), binding())
    wanted = [plan["cases"][4]["case_id"], plan["cases"][1]["case_id"]]
    result = materialize_classification_problem(
        plan, make_opener(X, [], monkeypatch), case_ids=wanted
    )
    assert [row["case_id"] for row in result.cases] == list(reversed(wanted))


@pytest.mark.parametrize(
    "change", ["case", "metadata", "definition", "hash", "code", "extra"]
)
def test_altered_plan_is_rejected_before_source_access(sources, change):
    """Metadata and assignments are regenerated before invoking a fetch adapter."""
    _, metadata = sources
    plan = plan_classification_problem(metadata, definition(), binding())
    if change == "case":
        plan["cases"][0]["stop_sample"] += 1
    elif change == "metadata":
        plan["recordings"][0]["release"] = "another-release"
    elif change == "definition":
        plan["problem"]["definition"]["question"] = "different"
    elif change == "hash":
        plan["plan_id"] = "changed"
    elif change == "code":
        plan["implementation_sha256"]["case_rules"] = "changed"
    else:
        plan["ignored_field"] = True
    with pytest.raises(ValueError):
        materialize_classification_problem(
            plan, lambda descriptor: pytest.fail("Source opened")
        )


@pytest.mark.parametrize("change", ["channels", "rate", "length", "nonfinite", "shape"])
def test_reader_mismatches_and_bad_samples_close_the_source(
    sources, monkeypatch, change
):
    """Header declarations are checked after fetching, samples when extracted."""
    X, metadata = sources
    plan = plan_classification_problem(metadata, definition(), binding())
    trace = []
    with pytest.raises(ValueError):
        materialize_classification_problem(
            plan, make_opener(X, trace, monkeypatch, change=change)
        )
    assert trace[-1] == ("close", "R0")


@pytest.mark.parametrize("selection", [[], ["unknown"], ["same", "same"]])
def test_invalid_selections_fail_before_source_access(sources, selection):
    """No implicit fallback to fetching all recordings is permitted."""
    _, metadata = sources
    plan = plan_classification_problem(metadata, definition(), binding())
    with pytest.raises(ValueError):
        materialize_classification_problem(
            plan, lambda descriptor: pytest.fail("Source opened"), case_ids=selection
        )


def test_heterogeneous_inventory_can_be_planned_and_compatible_subset_selected(
    sources, monkeypatch
):
    """Inspection can precede the choice of which channel layouts to combine."""
    X, metadata = sources
    metadata[1]["sampling_frequency"] = 8
    plan = plan_classification_problem(metadata, definition(), binding())
    assert plan["dense_shape"] is None
    with pytest.raises(ValueError, match="sampling frequency"):
        materialize_classification_problem(
            plan, lambda descriptor: pytest.fail("Source opened")
        )
    wanted = [row["case_id"] for row in plan["cases"] if row["recording_id"] == "R1"]
    result = materialize_classification_problem(
        plan, make_opener(X, [], monkeypatch), case_ids=wanted
    )
    assert result.metadata["sampling_frequency"] == 8


def test_incomplete_catalogue_does_not_invent_sample_counts(sources):
    """Definitions can be saved before header facts needed for exact planning exist."""
    _, metadata = sources
    metadata[0].pop("n_samples")
    assert resolve_classification_problem(definition(), binding())["definition"]
    with pytest.raises(ValueError, match="n_samples"):
        plan_classification_problem(metadata, definition(), binding())


def test_event_exclusions_survive_plan_roundtrip_and_materialization(
    sources, monkeypatch
):
    """A rejected source trial cannot be selected or disappear during replay."""
    X, metadata = sources
    rows = events()
    rows[0].update(included=False, exclusion_reason="source_quality")
    plan = plan_classification_problem(
        metadata, definition("event_epochs"), binding("event_epochs"), events=rows
    )
    dataset = materialize_classification_problem(plan, make_opener(X, [], monkeypatch))
    assert dataset.exclusions == plan["exclusions"]
    with pytest.raises(ValueError, match="excluded"):
        materialize_classification_problem(
            plan,
            lambda descriptor: pytest.fail("Source opened"),
            case_ids=[plan["exclusions"][0]["case_id"]],
        )
