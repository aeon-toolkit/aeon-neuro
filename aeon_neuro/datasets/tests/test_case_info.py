"""Carry shared information through ordinary case builders without a schema."""

import builtins
import json

import numpy as np
import pytest

from aeon_neuro.datasets import (
    EEGDatasetInfo,
    build_classification_dataset,
    load_bids_dataset,
    make_event_cases,
    make_fixed_length_cases,
    make_recording_cases,
)
from aeon_neuro.datasets.tests import test_dataset_info as fixtures

source = fixtures.source


def test_explicit_case_channel_aliases_keep_source_units_and_channel_mapping(source):
    """Logical case names retain their declared acquisition-channel mapping."""
    X, info = load_bids_dataset(source[0])
    cases = make_event_cases(X, info, info.events_for_cases(), tmin=0, tmax=1)
    for row in cases:
        row["channel_names"] = ["C4", "C3"]
        row["output_channel_names"] = ["right", "left"]
    result = build_classification_dataset(X, info, cases)
    assert result.info.channel_names == ("right", "left")
    assert result.info.layouts[0]["source_channel_names"] == ["C4", "C3"]
    assert result.info.units == ("V", "V")
    EEGDatasetInfo(json.loads(json.dumps(result.info.to_dict()))).validate_layout(
        result.X
    )


@pytest.mark.parametrize("kind", ["events", "windows", "recordings"])
def test_shared_info_case_builders_preserve_source(source, kind, monkeypatch):
    """Source information must not depend on authored problem documents."""
    X, info = load_bids_dataset(source[0])
    before = info.to_dict()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {
            "aeon_neuro.datasets._problem",
            "aeon_neuro.datasets._problem_plan",
        }:
            raise AssertionError("Problem schema imported by an ordinary case builder")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    if kind == "events":
        cases = make_event_cases(X, info, info.events_for_cases(), tmin=0, tmax=1)
        first = X[0, :, 32:64]
    elif kind == "windows":
        cases = make_fixed_length_cases(
            X, info, labels=[0, 1], groups=["a", "b"], window_size=1
        )
        first = X[0, :, :32]
    else:
        cases = make_recording_cases(X, info, labels=[0, 1], groups=["a", "b"])
        first = X[0]
    result = build_classification_dataset(X, info, cases)
    np.testing.assert_array_equal(result.X[0], first)
    assert "problem" not in result.metadata["provenance"]
    assert result.info.history == info.history
    assert result.info.participant_ids[0] == "sub-01"
    assert result.info.units == ("V", "V")
    restored = EEGDatasetInfo(json.loads(json.dumps(result.info.to_dict())))
    restored.validate_layout(result.X)
    result.metadata["provenance"]["source"]["events"].clear()
    result.X[0] = 0
    assert info.to_dict() == before
    np.testing.assert_array_equal(X[0], load_bids_dataset(source[0])[0][0])


@pytest.mark.parametrize("fault", ["count", "channels", "samples", "cases"])
def test_shared_info_builders_reject_mismatched_or_segmented_inputs(source, fault):
    """A shared object cannot silently describe another collection or grid."""
    X, info = load_bids_dataset(source[0])
    if fault == "cases":
        cases = make_event_cases(X, info, info.events_for_cases(), tmin=0, tmax=1)
        result = build_classification_dataset(X, info, cases)
        X, info = result.X, result.info
    else:
        X = {"count": X[:1], "channels": X[:, :1], "samples": X[:, :, :-1]}[fault]
    with pytest.raises(ValueError):
        make_recording_cases(X, info, labels=[0] * len(X), groups=["a"] * len(X))


def test_restored_current_recordings_keep_dictionary_descriptor_boundary():
    """Restored current layouts feed public recording and event helpers."""
    from aeon_neuro.datasets.tests.test_info_records import saved_state

    state = saved_state()
    state["view"]["kind"] = "recordings"
    state["events"][0]["trial_type"] = "left"
    info = EEGDatasetInfo(state)
    assert info.current_metadata[0]["n_samples"] == 8
    assert info.metadata[0]["n_samples"] == 16
    assert info.events_for_cases()[0]["label"] == "left"
    assert info.events_for_cases()[0]["group_id"] == 1
