"""Shared-information integration with authored problem definitions."""

import json
from copy import deepcopy

import mne
import numpy as np
import pytest

from aeon_neuro import __version__
from aeon_neuro.datasets import _dataset_info as module
from aeon_neuro.datasets import (
    load_bids_dataset,
    load_classification_problem,
    make_classification_problem,
)
from aeon_neuro.datasets.tests import test_dataset_info as fixtures

source = fixtures.source


def test_selected_info_feeds_the_existing_problem_loader(source):
    """Reuse explicit task bindings while retaining a reordered source join."""
    X, info = load_bids_dataset(source[0])
    X, info = info.select(X, [1, 0])
    problem = {
        "format_version": 1,
        "id": "imagery",
        "version": "1",
        "question": "Classify declared imagery",
        "class_labels": ["left", "right"],
        "channels": ["C4", "C3"],
        "segmentation": {"mode": "event_epochs", "tmin": 0.5, "tmax": 1.5},
    }
    binding = {
        "format_version": 1,
        "problem": {"id": "imagery", "version": "1"},
        "target": {
            "source": "event",
            "column": "trial_type",
            "mapping": {"left": "left", "right": "right"},
        },
        "group": {"source": "recording", "column": "participant_id"},
        "events": {"onset": "onset", "id": "bids_event_id"},
    }
    cases = load_classification_problem(
        X, info.metadata, problem, binding, events=info.events
    )
    assert cases.X.shape == (4, 2, 32)
    assert cases.groups.tolist() == ["sub-02", "sub-02", "sub-01", "sub-01"]
    assert cases.y.tolist() == ["left", "right", "left", "right"]
    np.testing.assert_array_equal(cases.X[0], X[0, [1, 0], 48:80])


def _info_problem_documents(mode):
    problem = {
        "format_version": 1,
        "id": "info-problem",
        "version": "1",
        "question": "Classify the explicitly bound source label",
        "class_labels": ["control", "case"],
        "channels": ["C4", "C3"],
        "segmentation": (
            {"mode": mode, "tmin": 0.5, "tmax": 1.5}
            if mode == "event_epochs"
            else {"mode": mode, "window_size": 1.0}
        ),
    }
    binding = {
        "format_version": 1,
        "problem": {"id": "info-problem", "version": "1"},
        "target": (
            {
                "source": "event",
                "column": "trial_type",
                "mapping": {"left": "control", "right": "case"},
            }
            if mode == "event_epochs"
            else {
                "source": "recording",
                "column": "participant.group",
                "mapping": {"control": "control", "case": "case"},
            }
        ),
        "group": {"source": "recording", "column": "participant_id"},
        "case_fields": {
            "participant_id": {"source": "recording", "column": "participant_id"},
            "source_task": {"source": "recording", "column": "task"},
        },
    }
    if mode == "event_epochs":
        binding["events"] = {"onset": "onset", "id": "bids_event_id"}
    return problem, binding


@pytest.mark.parametrize("mode", ["event_epochs", "fixed_windows"])
def test_object_builder_matches_existing_cases_and_keeps_unified_provenance(
    source, mode
):
    """The object route must preserve case semantics, source joins and history."""
    X, info = load_bids_dataset(source[0])
    info = info.with_history(
        "metadata review",
        {"columns": ["trial_type", "participant.group"]},
        implementation="example.review",
        version="1",
        fit_scope="none",
    )
    X, info = info.select(X, [1, 0])
    problem, binding = _info_problem_documents(mode)
    expected = load_classification_problem(
        X,
        info.metadata,
        problem,
        binding,
        events=info.events if mode == "event_epochs" else None,
    )
    result = make_classification_problem(X, info, problem, binding)
    np.testing.assert_array_equal(result.X, expected.X)
    np.testing.assert_array_equal(result.y, expected.y)
    np.testing.assert_array_equal(result.groups, expected.groups)
    assert result.cases == expected.cases and result.exclusions == expected.exclusions
    assert result.cases[0]["recording_id"] == info.source_ids[0]
    assert result.cases[0]["participant_id"] == "sub-02"
    start, stop = (48, 80) if mode == "event_epochs" else (0, 32)
    np.testing.assert_array_equal(result.X[0], X[0, [1, 0], start:stop])
    assert result.y[0] == ("control" if mode == "event_epochs" else "case")
    provenance = result.metadata["provenance"]
    assert provenance["version"] == 1
    assert provenance["problem"] == expected.metadata["problem"]
    assert "problem" not in result.metadata
    source_report = info.to_dict()
    assert provenance["history"] == source_report.pop("history")
    assert provenance["source"] == source_report
    assert provenance["history"][0]["source_ids"] != list(info.source_ids)
    assert provenance["construction"] == {
        "implementation": "aeon_neuro.datasets.make_classification_problem",
        "version": __version__,
    }
    assert json.loads(json.dumps(result.metadata, allow_nan=False)) == result.metadata
    other_metadata = deepcopy(result.metadata)
    other_metadata.pop("provenance")
    expected.metadata.pop("problem")
    assert other_metadata == expected.metadata


def test_object_builder_detaches_signals_source_history_and_documents(source):
    """Changing an output must not change another problem or its original inputs."""
    X, info = load_bids_dataset(source[0])
    info = info.with_history(
        "metadata review",
        {"columns": ["trial_type"]},
        implementation="example.review",
        version="1",
    )
    problem, binding = _info_problem_documents("event_epochs")
    before = deepcopy((X, info.to_dict(), problem, binding))
    result = make_classification_problem(X, info, problem, binding)
    result.X[0] = 0
    provenance = result.metadata["provenance"]
    provenance["source"]["events"][0]["trial_type"] = "changed"
    provenance["source"]["dataset"]["participants"]["rows"][0]["group"] = "changed"
    provenance["history"][0]["parameters"]["columns"].append("changed")
    provenance["problem"]["definition"]["channels"].reverse()
    np.testing.assert_array_equal(X, before[0])
    assert (info.to_dict(), problem, binding) == before[1:]
    again = make_classification_problem(X, info, problem, binding)
    expected_source = info.to_dict()
    expected_source.pop("history")
    assert again.metadata["provenance"]["source"] == expected_source
    assert again.metadata["provenance"]["history"] == info.history


@pytest.mark.parametrize("fault", ["count", "channels", "samples"])
def test_object_builder_rejects_mismatched_source_layouts(source, fault):
    """The source view cannot silently describe changed axes or sample grids."""
    X, info = load_bids_dataset(source[0])
    changed = {"count": X[:1], "channels": X[:, :1], "samples": X[:, :, :128]}[fault]
    with pytest.raises(ValueError):
        make_classification_problem(
            changed, info, *_info_problem_documents("event_epochs")
        )


@pytest.mark.parametrize("wrong", [None, {}, [], "metadata"])
def test_object_builder_requires_the_shared_metadata_object(source, wrong):
    """Reject alternate implicit metadata/event input conventions."""
    X, _ = load_bids_dataset(source[0])
    with pytest.raises(ValueError, match="EEGDatasetInfo"):
        make_classification_problem(X, wrong, *_info_problem_documents("event_epochs"))


def test_object_builder_reads_no_bids_files_after_loading(source, monkeypatch):
    """Already-loaded sources need no second reader or event-table parser."""
    X, info = load_bids_dataset(source[0])

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "BIDS signal or metadata access during problem construction"
        )

    monkeypatch.setattr(module, "inspect_bids_eeg", unexpected)
    monkeypatch.setattr(module, "open_bids_eeg_recording", unexpected)
    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected)
    result = make_classification_problem(
        X, info, *_info_problem_documents("event_epochs")
    )
    assert result.X.shape == (4, 2, 32)
    assert result.metadata["provenance"]["history"] == []


def test_object_builder_preserves_exclusion_and_source_event_identity(source):
    """Out-of-bounds exclusions stay governed by the existing problem rules."""
    X, info = load_bids_dataset(source[0])
    problem, binding = _info_problem_documents("event_epochs")
    problem["segmentation"].update(tmin=0.0, tmax=7.0, on_out_of_bounds="exclude")
    result = make_classification_problem(X, info, problem, binding)
    expected = load_classification_problem(
        X, info.metadata, problem, binding, events=info.events
    )
    assert result.cases == expected.cases and result.exclusions == expected.exclusions
    assert len(result.cases) == len(result.exclusions) == 2
    assert result.exclusions[0]["exclusion_reason"] == "outside_recording"
    assert result.cases[0]["event_id"] == info.events[0]["bids_event_id"]


def test_object_builder_rejects_empty_views_and_missing_bound_participant_columns(
    source,
):
    """Metadata-only validity cannot supply absent labels or training observations."""
    X, info = load_bids_dataset(source[0])
    empty, empty_info = info.select(X, [])
    problem, binding = _info_problem_documents("fixed_windows")
    with pytest.raises(ValueError, match="nonempty"):
        make_classification_problem(empty, empty_info, problem, binding)
    binding["target"]["column"] = "participant.missing"
    with pytest.raises(ValueError, match="participant.missing"):
        make_classification_problem(X, info, problem, binding)
