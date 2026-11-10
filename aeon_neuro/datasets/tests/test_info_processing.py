"""Check executed operations against independent arrays and source coordinates."""

import json
from math import ceil

import numpy as np
import pytest

from aeon_neuro.datasets import (
    EEGDatasetInfo,
    load_bids_dataset,
)
from aeon_neuro.datasets.tests import test_dataset_info as _fixtures
from aeon_neuro.datasets.tests.test_info_view import _event_dataset, example_info

source = _fixtures.source


@pytest.mark.parametrize("operation", ["reference", "resample"])
def test_processing_serializes_new_optional_layout_facts(operation):
    """Executing an operation records facts omitted from restored metadata."""
    X = np.arange(32, dtype=float).reshape(1, 2, 16)
    info = EEGDatasetInfo.from_arrays(
        X, sfreq=8, channel_names=["C3", "C4"], units=["V", "V"]
    )
    state = info.to_dict()
    state["view"]["layouts"][0].pop("reference")
    state["view"]["layouts"][0].pop("temporal_context")
    info = EEGDatasetInfo(state)
    parameters = {"channels": "average"} if operation == "reference" else {"sfreq": 4}
    Y, processed = info.process(X, operation, **parameters)
    layout = processed.to_dict()["view"]["layouts"][0]
    if operation == "reference":
        assert layout["reference"] == {"method": "mean", "channels": ["C3", "C4"]}
        np.testing.assert_array_equal(Y, X - X.mean(axis=1, keepdims=True))
    else:
        assert layout["temporal_context"] == "whole_row"
        assert Y.shape == (1, 2, 8)
    EEGDatasetInfo(processed.to_dict()).validate_layout(Y)
    assert info.to_dict() == state


@pytest.mark.parametrize("processed", [False, True])
def test_current_descriptors_match_plain_event_cases_and_remain_detached(
    source, processed
):
    """Plain case construction uses the same current grid as the object builder."""
    from aeon_neuro.datasets import build_classification_dataset, make_event_cases

    X, info = load_bids_dataset(source[0])
    original = info.metadata
    if processed:
        X, info = info.process(X, "channels", names=["C4", "C3"])
        X, info = info.process(X, "crop", tmin=0.5, tmax=4.0)
        X, info = info.process(X, "resample", sfreq=16)
    X, info = info.select(X, [1, 0])
    metadata = info.current_metadata
    events = info.events_for_cases()
    candidates = make_event_cases(X, metadata, events, tmin=0.5, tmax=1.5)
    for row in candidates:
        row["channel_names"] = ["C4", "C3"]
    plain = build_classification_dataset(X, metadata, candidates)
    offset, length = (16, 16) if processed else (48, 32)
    np.testing.assert_array_equal(
        plain.X[0],
        X[
            0,
            [metadata[0]["channel_names"].index(n) for n in ("C4", "C3")],
            offset : offset + length,
        ],
    )
    assert plain.groups.tolist() == ["sub-02"] * 2 + ["sub-01"] * 2
    assert [row["onset"] for row in events] == (
        [0.5, 1.5] * 2 if processed else [1, 2] * 2
    )
    # Derived views retain full source context independently of current order.
    assert info.metadata == (original if processed else original[::-1])
    metadata[0]["channel_names"].clear()
    assert info.current_metadata[0]["channel_names"] == list(info.channel_names)


def test_current_recording_descriptors_reject_segmented_cases():
    """Already built cases cannot be presented as continuous recording inputs."""
    _, info = example_info()
    with pytest.raises(ValueError, match="already built cases"):
        _ = info.current_metadata


def test_processing_chain_updates_current_layout_without_rewriting_source(source):
    """Current channels/rate/units follow values; acquisition facts stay intact."""
    X, source_info = load_bids_dataset(source[0])
    dataset = _event_dataset(X, source_info)
    original = dataset.info
    Y, selected = original.process(dataset.X, "channels", names=["C3"])
    np.testing.assert_array_equal(Y, dataset.X[:, [1]])
    Y, scaled = selected.process(Y, "scale", factor=1e6, units="uV")
    np.testing.assert_array_equal(Y, dataset.X[:, [1]] * 1e6)
    Y, cropped = scaled.process(Y, "crop", tmin=0.25, tmax=0.75)
    np.testing.assert_array_equal(Y, dataset.X[:, [1], 8:24] * 1e6)
    Y, resampled = cropped.process(Y, "resample", sfreq=16)
    assert Y.shape == (4, 1, 8) and resampled.sfreq == 16
    assert resampled.channel_names == ("C3",) and resampled.units == ("uV",)
    assert resampled.layouts[0]["source_origin_sample"] == 56
    assert resampled.layouts[0]["source_sample_step"] == 2
    assert resampled.cases[0]["source_support"] == [48, 80]
    assert resampled.recordings[0].sfreq == 32
    assert resampled.recordings[0].channel_names == ("C3", "C4")
    assert resampled.recordings[0].channels[0]["units"] == "uV"
    assert original.history == []
    assert [r["step"] for r in resampled.history] == [
        "channels",
        "scale",
        "crop",
        "resample",
    ]
    assert all(r["parameters"]["execution"] == "performed" for r in resampled.history)
    EEGDatasetInfo(json.loads(json.dumps(resampled.to_dict()))).validate_layout(Y)
    _, subset = resampled.select(Y, [0])
    assert (
        subset.history == resampled.history
    )  # do not narrow earlier processing scopes


def test_reference_and_filter_match_independent_operations():
    """Helpers execute existing numerical implementations, not descriptive notes."""
    from mne.filter import filter_data

    X, info = example_info()
    Y, referenced = info.process(X, "reference", channels=["C3"])
    np.testing.assert_array_equal(Y, X - X[:, [0]])
    assert referenced.layouts[0]["reference"] == {"method": "mean", "channels": ["C3"]}
    Y, filtered = info.process(X, "filter", l_freq=2.0, h_freq=8.0, method="iir")
    np.testing.assert_array_equal(
        Y, filter_data(X, 32, 2.0, 8.0, method="iir", verbose=False)
    )
    assert filtered.history[-1]["parameters"]["support_policy"].startswith(
        "conservative"
    )
    np.testing.assert_array_equal(X, example_info()[0])


@pytest.mark.parametrize(
    "operation,parameters",
    [
        ("unknown", {}),
        ("channels", {"names": ["missing"]}),
        ("channels", {"names": ["C3", "C3"]}),
        ("scale", {"factor": True, "units": "V"}),
        ("scale", {"factor": float("inf"), "units": "V"}),
        ("scale", {"factor": 2.0, "units": ["V"]}),
        ("resample", {"sfreq": 0}),
        ("crop", {"tmin": 0.01}),
        ("crop", {"tmax": 2.0}),
        ("filter", {"l_freq": 2.0, "h_freq": 20.0}),
        ("reference", {"channels": []}),
        ("crop", {"unknown": 1}),
    ],
)
def test_failed_processing_leaves_arrays_and_information_unchanged(
    operation, parameters
):
    """Validation and numerical failures do not mutate the supplied pair."""
    X, info = example_info()
    before = info.to_dict()
    with pytest.raises(ValueError):
        info.process(X, operation, **parameters)
    np.testing.assert_array_equal(X, example_info()[0])
    assert info.to_dict() == before


def test_recording_processing_preserves_original_events_and_layout(source):
    """Recording-level output keeps its current grid separate from acquisition."""
    X, info = load_bids_dataset(source[0])
    Y, changed = info.process(X, "crop", tmin=0.5, tmax=2.5)
    np.testing.assert_array_equal(Y, X[:, :, 16:80])
    assert changed.layouts[0]["source_origin_sample"] == 16
    assert changed.events == info.events and changed.metadata == info.metadata
    assert changed.recordings[0].sfreq == 32 and Y.shape[-1] == 64


def test_already_segmented_cases_are_not_silently_reinterpreted_as_recordings(source):
    """The object builder requires an actual recording grid for event extraction."""
    X, info = load_bids_dataset(source[0])
    dataset = _event_dataset(X, info)
    with pytest.raises(ValueError, match="already built cases"):
        _event_dataset(dataset.X, dataset.info)


def test_derived_event_column_collision_is_rejected_without_rewriting_source(source):
    """A derived annotation must not silently replace a published event field."""
    root, paths = source
    for path in paths:
        events = path.with_name(path.name.replace("_eeg.bdf", "_events.tsv"))
        lines = events.read_text().splitlines()
        events.write_text(
            lines[0]
            + "\tacquisition_event\n"
            + "".join(row + "\toriginal-source-value\n" for row in lines[1:]),
            encoding="utf-8",
        )
    X, info = load_bids_dataset(root)
    Y, changed = info.process(X, "crop", tmin=0.5, tmax=4.0)
    with pytest.raises(ValueError, match="acquisition_event"):
        _event_dataset(Y, changed)
    assert info.events[0]["acquisition_event"] == "original-source-value"


@pytest.mark.parametrize(
    "length,rate,target,up,down",
    [
        (101, 100, 64, 16, 25),
        (102, 100, 64, 16, 25),
        (101, 64, 100, 25, 16),
        (101, 250.5, 128, 256, 501),
    ],
)
def test_resampling_preserves_requested_sample_grid(length, rate, target, up, down):
    """Rounding the output length must not change the executed sampling ratio."""
    from scipy.signal import resample_poly

    X = np.arange(length, dtype=float)[None, None, :]
    info = EEGDatasetInfo.from_arrays(X, sfreq=rate, channel_names=["C3"])
    output, changed = info.process(X, "resample", sfreq=target)
    expected = resample_poly(X, up, down, axis=-1, padtype="reflect")
    np.testing.assert_array_equal(output, expected)
    assert output.shape[-1] == ceil(length * up / down)
    assert changed.sfreq == target
    assert changed.layouts[0]["source_sample_step"] == rate / target
    assert changed.cases[0]["source_support"] == [0, length]
    restored = EEGDatasetInfo(json.loads(json.dumps(changed.to_dict())))
    restored.validate_layout(output)


def test_resampled_recording_event_cases_and_crops_follow_original_grid(source):
    """Later event extraction must consume the same samples as the stored map."""
    from scipy.signal import resample_poly

    from aeon_neuro.datasets import build_classification_dataset, make_event_cases

    X, info = load_bids_dataset(source[0])
    X, info = info.process(X, "crop", tmin=0.5, tmax=7.53125)
    assert X.shape[-1] == 225
    original = resample_poly(X, 5, 8, axis=-1, padtype="reflect")
    X, info = info.process(X, "resample", sfreq=20)
    cases = make_event_cases(X, info, info.events_for_cases(), tmin=0.1, tmax=0.5)
    dataset = build_classification_dataset(X, info, cases)
    np.testing.assert_array_equal(dataset.X[0], original[0, :, 12:20])
    current = dataset.info
    assert current.layouts[0]["source_origin_sample"] == pytest.approx(35.2)
    assert current.layouts[0]["source_sample_step"] == 1.6
    output, cropped = current.process(dataset.X, "crop", tmin=0.1, tmax=0.3)
    np.testing.assert_array_equal(output[0], original[0, :, 14:18])
    assert cropped.layouts[0]["source_origin_sample"] == pytest.approx(38.4)
    assert cropped.cases[0]["source_support"] == [0, 256]
    assert cropped.history[-2]["parameters"]["backend_versions"]["scipy"]


def test_large_resampling_ratio_fails_before_allocating_a_filter(monkeypatch):
    """Unusual rates must not cause unbounded polyphase filter construction."""
    import scipy.signal

    X = np.zeros((1, 1, 101))
    info = EEGDatasetInfo.from_arrays(X, sfreq=100, channel_names=["C3"])
    before = info.to_dict()

    def unexpected(*args, **kwargs):
        raise AssertionError("Filter allocated before checking the rate ratio")

    monkeypatch.setattr(scipy.signal, "resample_poly", unexpected)
    with pytest.raises(ValueError, match="rate ratio"):
        info.process(X, "resample", sfreq=64.000001)
    np.testing.assert_array_equal(X, np.zeros_like(X))
    assert info.to_dict() == before


@pytest.mark.parametrize(
    "fault",
    [
        "not_list",
        "record",
        "missing",
        "extra",
        "version",
        "boolean_version",
        "step",
        "implementation",
        "implementation_version",
        "fit_scope",
        "parameters",
        "sources",
        "duplicate_source",
        "empty_source",
    ],
)
def test_saved_history_records_are_validated(source, fault):
    """Restoring a metadata view must not silently trust malformed provenance."""
    X, info = load_bids_dataset(source[0])
    _, info = info.process(X, "scale", factor=1.0, units="V")
    report = info.to_dict()
    record = report["history"][0]
    if fault == "not_list":
        report["history"] = None
    elif fault == "record":
        report["history"] = ["step"]
    elif fault == "missing":
        record.pop("parameters")
    elif fault == "extra":
        record["unknown"] = "value"
    elif fault == "version":
        record["version"] = 2
    elif fault == "boolean_version":
        record["version"] = True
    elif fault in ("step", "implementation", "implementation_version", "fit_scope"):
        record[fault] = ""
    elif fault == "parameters":
        record["parameters"] = {"value": float("nan")}
    elif fault == "sources":
        record["source_ids"] = "not a list"
    elif fault == "duplicate_source":
        record["source_ids"] *= 2
    else:
        record["source_ids"] = []
    with pytest.raises(ValueError, match="history"):
        EEGDatasetInfo(report)


def test_executed_history_keeps_original_scope_after_selection(source):
    """A later subset must not hide the population used by an earlier operation."""
    X, info = load_bids_dataset(source[0])
    X, processed = info.process(X, "scale", factor=1.0, units="V")
    selected, subset = processed.select(X, [1])
    assert subset.history == processed.history
    assert subset.history[0]["source_ids"] == list(info.source_ids)
    _, following = subset.process(selected, "channels", names=["C3"])
    assert following.history[:-1] == processed.history
    assert following.history[-1]["source_ids"] == [info.source_ids[1]]
    restored = EEGDatasetInfo(json.loads(json.dumps(following.to_dict())))
    assert restored.history == following.history


@pytest.mark.parametrize("target", [0.5, 2.0])
def test_singleton_resampling_raises_without_native_crash(target):
    """A native boundary failure must not terminate the calling Python process."""
    import os
    import subprocess
    import sys

    # Isolate the regression so an unsafe SciPy call cannot kill the test runner.
    code = f"""
try:
    import resource
except ImportError:
    pass
else:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
from aeon_neuro.datasets import EEGDatasetInfo
X = np.ones((2, 1, 1))
info = EEGDatasetInfo.from_arrays(X, sfreq=1, channel_names=['C3'])
try:
    info.process(X, 'resample', sfreq={target})
except ValueError as exc:
    assert 'at least two samples' in str(exc)
else:
    raise AssertionError('Expected a singleton input error')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)


def test_singleton_resampling_preserves_inputs_after_a_later_row_fails():
    """An unequal-length collection stays untouched when its second row fails."""
    X = [np.ones((1, 4)), np.ones((1, 1))]
    info = EEGDatasetInfo.from_arrays(X, sfreq=1, channel_names=["C3"])
    before = info.to_dict()
    arrays = [row.copy() for row in X]
    with pytest.raises(ValueError, match="at least two samples"):
        info.process(X, "resample", sfreq=2)
    assert info.to_dict() == before
    for expected, actual in zip(arrays, X, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_singleton_resampling_at_the_original_rate_is_safe():
    """An unchanged sampling grid needs no reflected filter boundaries."""
    X = np.ones((2, 1, 1))
    info = EEGDatasetInfo.from_arrays(X, sfreq=1, channel_names=["C3"])
    result, updated = info.process(X, "resample", sfreq=1)
    np.testing.assert_array_equal(result, X)
    assert updated.sfreq == 1
    assert not np.shares_memory(result, X)
