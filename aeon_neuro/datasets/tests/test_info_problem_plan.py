"""Compare shared-info eager cases with metadata-only plans and selected reads."""

import json
from copy import deepcopy

import mne
import numpy as np
import pytest

from aeon_neuro import __version__
from aeon_neuro.datasets import (
    EEGDatasetInfo,
    inspect_bids_eeg,
    load_bids_dataset,
    make_classification_problem,
    materialize_classification_problem,
    open_bids_eeg_recording,
    plan_classification_problem,
)
from aeon_neuro.datasets import _bids_input as adapter
from aeon_neuro.datasets.tests import test_info_problem as fixtures

source = fixtures.source


def _opener(root, trace):
    def open_recording(descriptor):
        identity = descriptor["recording_id"]
        trace.append(("open", identity))
        raw = open_bids_eeg_recording(root, descriptor)
        read, close = raw.get_data, raw.close

        def get_data(**kwargs):
            trace.append(("read", identity, kwargs["start"], kwargs["stop"]))
            return read(**kwargs)

        def closed():
            trace.append(("close", identity))
            close()

        raw.get_data, raw.close = get_data, closed
        return raw

    return open_recording


def _assert_matching(eager, deferred):
    np.testing.assert_array_equal(deferred.X, eager.X)
    np.testing.assert_array_equal(deferred.y, eager.y)
    np.testing.assert_array_equal(deferred.groups, eager.groups)
    assert deferred.cases == eager.cases
    assert deferred.exclusions == eager.exclusions
    left, right = eager.metadata["provenance"], deferred.metadata["provenance"]
    assert left["version"] == right["version"] == 1
    for section in ("source", "history", "problem"):
        assert right[section] == left[section]
    assert left["construction"]["implementation"].endswith(
        "make_classification_problem"
    )
    assert right["construction"] == {
        "implementation": "aeon_neuro.datasets.materialize_classification_problem",
        "version": __version__,
    }
    assert "problem" not in deferred.metadata


@pytest.mark.parametrize("mode", ["event_epochs", "fixed_windows"])
def test_info_planning_and_materialization_match_eager_after_reordering(
    source, monkeypatch, mode
):
    """Plans reuse the same source facts, labels, grouping and reported history."""
    root, _ = source
    X, info = load_bids_dataset(root)
    info = info.with_history(
        "metadata review",
        {"columns": ["trial_type", "participant.group"]},
        implementation="example.review",
        version="1",
        fit_scope="none",
    )
    X, info = info.select(X, [1, 0])
    problem, binding = fixtures._info_problem_documents(mode)
    eager = make_classification_problem(X, info, problem, binding)

    def unexpected(*args, **kwargs):
        raise AssertionError("Source access while planning from shared info")

    with monkeypatch.context() as context:
        context.setattr(mne.io.BaseRaw, "get_data", unexpected)
        context.setattr(adapter, "load_eeg_to_mne", unexpected)
        context.setattr(adapter, "_hash", unexpected)
        plan = plan_classification_problem(info, problem, binding)
    assert plan["format_version"] == 2
    assert plan["input_info"] == info.to_dict()
    assert "metadata" in plan["implementation_sha256"]
    assert "package" in plan["implementation_sha256"]
    assert "X" not in plan
    assert plan["input_info"]["history"][0]["source_ids"] != list(info.source_ids)
    trace = []
    result = materialize_classification_problem(
        json.loads(json.dumps(plan)), _opener(root, trace)
    )
    _assert_matching(eager, result)
    assert [row[1] for row in trace if row[0] == "open"] == list(info.source_ids)
    assert len([row for row in trace if row[0] == "read"]) == len(result.cases)
    assert len([row for row in trace if row[0] == "close"]) == len(info)


def test_info_plan_survives_unavailable_signals_without_sample_access(
    source, monkeypatch
):
    """Inspection metadata can define exact cases while all source files are absent."""
    root, paths = source
    info = EEGDatasetInfo(inspect_bids_eeg(root))
    for path in paths:
        path.unlink()

    def unexpected(*args, **kwargs):
        raise AssertionError("Signal access during metadata-only planning")

    monkeypatch.setattr(adapter, "load_eeg_to_mne", unexpected)
    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected)
    monkeypatch.setattr(adapter, "_hash", unexpected)
    plan = plan_classification_problem(
        info, *fixtures._info_problem_documents("event_epochs")
    )
    assert plan["dense_shape"] == [4, 2, 32]
    assert json.loads(json.dumps(plan))["input_info"] == info.to_dict()


@pytest.mark.parametrize(
    "fault",
    [
        "history",
        "event",
        "source",
        "recordings",
        "metadata_code",
        "bool_version",
        "unknown_version",
        "missing_info",
        "extra",
    ],
)
def test_changed_info_plan_is_rejected_before_calling_reader(source, fault):
    """Replay must validate the saved object and its join with planned case rows."""
    info = EEGDatasetInfo(inspect_bids_eeg(source[0])).with_history(
        "review", {}, implementation="example", version="1"
    )
    plan = plan_classification_problem(
        info, *fixtures._info_problem_documents("event_epochs")
    )
    if fault == "history":
        plan["input_info"]["history"][0]["fit_scope"] = "changed"
    elif fault == "event":
        plan["input_info"]["events"][0]["trial_type"] = "changed"
    elif fault == "source":
        plan["input_info"]["recordings"][0]["channel_names"].reverse()
    elif fault == "recordings":
        plan["recordings"].reverse()
    elif fault == "metadata_code":
        plan["implementation_sha256"]["metadata"] = "changed"
    elif fault == "bool_version":
        plan["format_version"] = True
    elif fault == "unknown_version":
        plan["format_version"] = 3
    elif fault == "missing_info":
        plan.pop("input_info")
    else:
        plan["ignored"] = True
    with pytest.raises(ValueError):
        materialize_classification_problem(plan, lambda _: pytest.fail("Reader called"))


@pytest.mark.parametrize("events", [[], [{"recording_id": "other"}]])
def test_info_events_cannot_be_overridden_separately(source, events):
    """There must be one authoritative event source for the shared object route."""
    info = EEGDatasetInfo(inspect_bids_eeg(source[0]))
    with pytest.raises(ValueError, match="events"):
        plan_classification_problem(
            info, *fixtures._info_problem_documents("event_epochs"), events=events
        )


def test_selected_cases_open_only_needed_source_and_preserve_planning_context(source):
    """Unselected signal files may be missing while earlier history remains intact."""
    root, paths = source
    info = EEGDatasetInfo(inspect_bids_eeg(root)).with_history(
        "review", {"columns": ["trial_type"]}, implementation="example", version="1"
    )
    plan = plan_classification_problem(
        info, *fixtures._info_problem_documents("event_epochs")
    )
    before = deepcopy(plan)
    chosen = plan["cases"][-1]
    paths[0].unlink()
    trace = []
    result = materialize_classification_problem(
        plan, _opener(root, trace), case_ids=[chosen["case_id"]]
    )
    assert trace == [
        ("open", info.source_ids[1]),
        ("read", info.source_ids[1], 80, 112),
        ("close", info.source_ids[1]),
    ]
    assert result.cases[0]["row"] == 0 and len(result.cases) == 1
    assert result.metadata["materialization"]["unselected_cases"] == 3
    assert result.metadata["provenance"]["source"]["source_ids"] == list(
        info.source_ids
    )
    assert result.metadata["provenance"]["history"] == info.history
    assert len(result.metadata["recordings"]) == 1
    assert plan == before
    provenance = result.metadata["provenance"]
    provenance["history"][0]["parameters"]["columns"].append("changed")
    provenance["source"]["events"][0]["trial_type"] = "changed"
    provenance["problem"]["definition"]["channels"].reverse()
    assert plan == before and info.history == before["input_info"]["history"]


def test_info_plan_retains_out_of_bounds_exclusions_exactly(source):
    """The shared binding rules keep excluded identities and reasons identical."""
    X, info = load_bids_dataset(source[0])
    problem, binding = fixtures._info_problem_documents("event_epochs")
    problem["segmentation"].update(tmin=0.0, tmax=7.0, on_out_of_bounds="exclude")
    eager = make_classification_problem(X, info, problem, binding)
    plan = plan_classification_problem(info, problem, binding)
    result = materialize_classification_problem(plan, _opener(source[0], []))
    _assert_matching(eager, result)
    assert len(result.exclusions) == 2


def test_stale_bids_metadata_is_rejected_by_explicit_opener(source, monkeypatch):
    """Saving a shared-info plan must not adopt changed sidecars on later loading."""
    root, _ = source
    info = EEGDatasetInfo(inspect_bids_eeg(root))
    plan = plan_classification_problem(
        info, *fixtures._info_problem_documents("event_epochs")
    )
    (root / "participants.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        adapter, "load_eeg_to_mne", lambda *a, **k: pytest.fail("Decoded")
    )
    with pytest.raises(ValueError, match="Inspected source changed"):
        materialize_classification_problem(plan, _opener(root, []))
