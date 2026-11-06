"""Check executed operations against independent arrays and source coordinates."""

import json

import numpy as np
import pytest

from aeon_neuro.datasets import (
    EEGDatasetInfo,
    load_bids_dataset,
)
from aeon_neuro.datasets.tests import test_dataset_info as _fixtures
from aeon_neuro.datasets.tests.test_info_view import _event_dataset, example_info

source = _fixtures.source


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


def test_ts_comment_is_portable_and_accepted_by_existing_aeon_writer(tmp_path):
    """Provenance comments preserve values and do not replace the .ts structure."""
    from aeon.datasets import load_from_ts_file, save_to_ts_file

    X, info = example_info()
    Y, changed = info.process(X, "scale", factor=1e6, units="uV")
    header = changed.to_ts_header()
    assert header.startswith("# aeon-neuro-info ")
    report = json.loads(header.removeprefix("# aeon-neuro-info "))
    assert report == changed.to_dict()
    save_to_ts_file(
        Y,
        np.array([0, 1, 0, 1]),
        label_type="classification",
        path=str(tmp_path),
        problem_name="generated",
        header=header,
    )
    values, labels = load_from_ts_file(tmp_path / "generated.ts")
    np.testing.assert_array_equal(values, Y)
    assert labels.tolist() == ["0", "1", "0", "1"]


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
