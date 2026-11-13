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
    selected, current = result.info.process(result.X, "channels", names=["left"])
    assert current.layouts[0]["source_channel_names"] == ["C3"]
    np.testing.assert_array_equal(selected, result.X[:, [1]])
    EEGDatasetInfo(json.loads(json.dumps(current.to_dict()))).validate_layout(selected)


@pytest.mark.parametrize("kind", ["events", "windows", "recordings"])
def test_shared_info_case_builders_preserve_source_and_history(
    source, kind, monkeypatch
):
    """Source information must not depend on authored problem documents."""
    X, info = load_bids_dataset(source[0])
    info = info.with_history("review", {}, implementation="example", version="1")
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


def test_processed_recordings_use_current_grid_and_original_source_support(source):
    """Case metadata preserves physical units and the acquisition coordinate map."""
    X, original = load_bids_dataset(source[0])
    X, info = original.process(X, "crop", tmin=0.5, tmax=4)
    X, info = info.process(X, "resample", sfreq=16)
    X, info = info.process(X, "scale", factor=1e6, units="uV")
    cases = make_event_cases(X, info, info.events_for_cases(), tmin=0.5, tmax=1.5)
    dataset = build_classification_dataset(X, info, cases)
    np.testing.assert_array_equal(dataset.X[0], X[0, :, 16:32])
    assert dataset.info.sfreq == 16 and dataset.info.units == ("uV", "uV")
    assert dataset.info.layouts[0]["source_origin_sample"] == 48
    assert dataset.info.layouts[0]["source_sample_step"] == 2
    assert dataset.info.cases[0]["source_support"] == [0, 256]
    assert dataset.info.recordings[0].sfreq == 32
    assert len(dataset.info.history) == 3


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


def _window_dataset(source, operation):
    X, info = load_bids_dataset(source[0], subjects=["01"])
    if operation == "filter":
        X, info = info.process(X, "filter", l_freq=2, h_freq=8, method="iir")
    elif operation == "crop":
        X, info = info.process(X, "crop", tmin=1, tmax=7)
    else:
        X, info = info.process(X, "channels", names=["C3"])
    cases = make_fixed_length_cases(
        X, info, labels=[0], groups=["recording"], window_size=1
    )
    for i, row in enumerate(cases):
        row.update(
            label=i % 2,
            group_id=f"trial-{i}",
            trial_id=f"trial-{i}",
            participant_id=f"actor-{i // 2}",
        )
    return build_classification_dataset(X, info, cases)


def test_split_validation_enforces_shared_processing_support(source):
    """Disjoint displayed windows can still depend on one filtered source row."""
    from aeon_neuro.datasets import validate_classification_split

    dataset = _window_dataset(source, "filter")
    with pytest.raises(ValueError, match="Original source support"):
        validate_classification_split(
            dataset,
            {"train": [0, 1, 2, 3], "test": [4, 5, 6, 7]},
            disjoint_by=[("trial_id",)],
            source_fields=("recording_id", "start_sample", "stop_sample"),
        )


@pytest.mark.parametrize("strategy", ["auto", "stratified"])
def test_inner_validation_rejects_known_shared_source_support(source, strategy):
    """Subject identities alone cannot make a shared filtered input independent."""
    from aeon_neuro.classification.tests.test_base import _InfoClassifier

    dataset = _window_dataset(source, "filter")
    classifier = _InfoClassifier().fit(dataset.X, dataset.y, info=dataset.info)
    with pytest.raises(ValueError, match="Original source support"):
        classifier.validation_split(
            dataset.y, strategy=strategy, random_state=0, validation_size=0.5
        )


@pytest.mark.parametrize("operation,origin", [("channels", 0), ("crop", 32)])
def test_pointwise_recording_operations_keep_exact_independent_case_support(
    source, operation, origin
):
    """Channel selection/cropping must not claim whole-recording time mixing."""
    from aeon_neuro.datasets import validate_classification_split

    dataset = _window_dataset(source, operation)
    info = dataset.info
    assert info.cases[0]["source_support"] == [origin, origin + 32]
    middle = len(dataset.y) // 2
    validate_classification_split(
        dataset,
        {"train": list(range(middle)), "test": list(range(middle, len(dataset.y)))},
        disjoint_by=[("trial_id",)],
        source_fields=("recording_id", "start_sample", "stop_sample"),
    )
