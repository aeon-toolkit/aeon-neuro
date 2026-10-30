"""Regression tests for signal/target alignment and evaluation boundaries.

Small arrays make off-by-one errors and incorrect channel picks observable exactly.
These tests cover extraction, eligibility, provenance, input validation and group
isolation without external datasets or downloads.
"""

from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import (
    build_classification_dataset,
    make_event_cases,
    make_fixed_length_cases,
    make_recording_cases,
    validate_group_split,
)


@pytest.fixture
def recordings():
    """Return distinguishable signals and loader-compatible metadata."""
    X = np.arange(2 * 3 * 12, dtype=np.float32).reshape(2, 3, 12)
    metadata = [
        {
            "filepath": f"sub-{i}/eeg/recording.edf",
            "subject": str(i),
            "channel_names": ["Fz", "Cz", "Pz"],
            "sampling_frequency": 4.0,
            "processing": [{"name": "acquisition", "scope": "recording"}],
        }
        for i in range(2)
    ]
    return X, metadata


def test_whole_recordings_and_independent_outputs(recordings):
    """Preserve alignment and metadata without aliasing any input."""
    X, metadata = recordings
    cases = make_recording_cases(X, metadata, labels=[0, 1], groups=[10, 11])
    original_cases = deepcopy(cases)
    dataset = build_classification_dataset(X, metadata, cases)
    np.testing.assert_array_equal(dataset.X, X)
    np.testing.assert_array_equal(dataset.y, [0, 1])
    np.testing.assert_array_equal(dataset.groups, [10, 11])
    assert dataset.cases[1]["row"] == 1
    assert dataset.cases[1]["source_file"] == metadata[1]["filepath"]
    assert dataset.cases[1]["source_sampling_frequency"] == 4
    assert dataset.exclusions == []
    dataset.X[0, 0, 0] = -10
    dataset.metadata["recordings"][0]["processing"][0]["scope"] = "changed"
    dataset.cases[0]["label"] = "changed"
    assert X[0, 0, 0] == 0
    assert metadata[0]["processing"][0]["scope"] == "recording"
    assert cases == original_cases


def test_fixed_windows_exclusions_and_exact_boundaries(recordings):
    """Keep source intervals and labels aligned after an eligibility rejection."""
    X, metadata = recordings
    cases = make_fixed_length_cases(
        X,
        metadata,
        labels=["covid", "close_contact"],
        groups=["p0", "p1"],
        window_size=1,
        start=0.5,
        stop=3,
    )
    assert [(c["start_sample"], c["stop_sample"]) for c in cases] == [
        (2, 6),
        (6, 10),
        (2, 6),
        (6, 10),
    ]
    cases[1].update(included=False, exclusion_reason="crosses_discontinuity")
    dataset = build_classification_dataset(X, metadata, cases)
    np.testing.assert_array_equal(
        dataset.X, [X[0, :, 2:6], X[1, :, 2:6], X[1, :, 6:10]]
    )
    assert dataset.y.tolist() == ["covid", "close_contact", "close_contact"]
    assert dataset.groups.tolist() == ["p0", "p1", "p1"]
    assert [r["row"] for r in dataset.cases] == [0, 1, 2]
    assert dataset.exclusions[0]["exclusion_reason"] == "crosses_discontinuity"
    assert "row" not in dataset.exclusions[0]


def test_recipe_order_is_preserved(recordings):
    """Prevent implicit sorting from misaligning external case-indexed metadata."""
    X, metadata = recordings
    cases = make_fixed_length_cases(
        X, metadata, labels=["a", "b"], groups=[10, 20], window_size=1
    )
    requested = [cases[5], cases[0], cases[3]]
    dataset = build_classification_dataset(X, metadata, requested)
    np.testing.assert_array_equal(dataset.X, [X[1, :, 8:12], X[0, :, :4], X[1, :, :4]])
    assert dataset.y.tolist() == ["b", "a", "b"]
    assert dataset.groups.tolist() == [20, 10, 20]
    assert [case["case_id"] for case in dataset.cases] == [
        case["case_id"] for case in requested
    ]


def test_window_stride_rate_and_partial_tail(recordings):
    """Derive lengths from each recording's rate and omit partial windows."""
    X, metadata = recordings
    metadata[1]["sampling_frequency"] = 8
    options = dict(labels=[0, 1], groups=[0, 1], window_size=1)
    cases = make_fixed_length_cases(X, metadata, step_size=0.5, **options)
    assert [c["start_sample"] for c in cases] == [0, 2, 4, 6, 8, 0, 4]
    assert [c["stop_sample"] for c in cases] == [4, 6, 8, 10, 12, 8, 12]
    # stop is capped at the recording, but start outside it yields no candidates.
    assert len(make_fixed_length_cases(X, metadata, stop=100, **options)) == 4
    assert make_fixed_length_cases(X, metadata, start=100, **options) == []


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_size": 0}, "at least one sample"),
        ({"window_size": 0.1}, "sample grid"),
        ({"step_size": 0}, "at least one sample"),
        ({"start": -1}, "nonnegative"),
        ({"start": 2, "stop": 1}, "greater than start"),
        ({"window_size": np.inf}, "finite seconds"),
    ],
)
def test_invalid_window_specification(recordings, kwargs, match):
    """Reject ambiguous or impossible timing specifications."""
    X, metadata = recordings
    options = dict(labels=[0, 1], groups=[0, 1], window_size=1)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_fixed_length_cases(X, metadata, **options)


def test_event_channel_banks_share_a_split_group():
    """One event can yield separate player cases while preserving the pair group."""
    X = np.arange(4 * 16, dtype=float).reshape(1, 4, 16)
    metadata = [
        {
            "filepath": "pair-1.bdf",
            "sampling_frequency": 4,
            "channel_names": ["p1-Fz", "p1-Cz", "p2-Fz", "p2-Cz"],
        }
    ]
    events = [
        {
            "recording_id": "pair-1.bdf",
            "onset": 1,
            "event_id": "trial-1",
            "label": label,
            "group_id": "pair-1",
            "participant_id": f"player-{player}",
            "channel_names": [f"p{player}-Fz", f"p{player}-Cz"],
            "output_channel_names": ["Fz", "Cz"],
        }
        for player, label in [(1, "rock"), (2, "paper")]
    ]
    # A second trial has no response from player 1; retain the rejection.
    events.append(
        dict(
            events[0],
            onset=2,
            event_id="trial-2",
            label=None,
            included=False,
            exclusion_reason="no_response",
        )
    )
    cases = make_event_cases(X, metadata, events, tmin=0, tmax=2)
    dataset = build_classification_dataset(X, metadata, cases)
    np.testing.assert_array_equal(dataset.X[0], X[0, :2, 4:12])
    np.testing.assert_array_equal(dataset.X[1], X[0, 2:, 4:12])
    assert dataset.y.tolist() == ["rock", "paper"]
    assert dataset.groups.tolist() == ["pair-1", "pair-1"]
    assert dataset.cases[0]["event_id"] == dataset.cases[1]["event_id"] == "trial-1"
    assert dataset.cases[0]["case_id"] != dataset.cases[1]["case_id"]
    assert dataset.metadata["channel_names"] == ["Fz", "Cz"]
    assert dataset.exclusions[0]["label"] is None
    with pytest.raises(ValueError, match="share group_id"):
        validate_group_split(dataset, [0], [1])


@pytest.mark.parametrize("sfreq", [4, 8])
def test_event_boundaries_and_exclusion_policy(recordings, sfreq):
    """Never silently clip event intervals or hide an unknown recording."""
    X, metadata = recordings
    metadata[0]["sampling_frequency"] = sfreq
    events = [
        dict(
            recording_id=metadata[0]["filepath"], onset=onset, label="a", group_id="p0"
        )
        for onset in [0, 1, 3]
    ]
    cases = make_event_cases(X, metadata, events, tmin=-0.5, tmax=0.5)
    with pytest.raises(ValueError, match="outside recording bounds"):
        build_classification_dataset(X, metadata, cases)
    dataset = build_classification_dataset(
        X, metadata, cases, on_out_of_bounds="exclude"
    )
    np.testing.assert_array_equal(dataset.X, X[:1, :, sfreq // 2 : 3 * sfreq // 2])
    assert [r["exclusion_reason"] for r in dataset.exclusions] == [
        "outside_recording"
    ] * 2
    assert dataset.exclusions[0]["start_sample"] == -sfreq // 2
    with pytest.raises(ValueError, match="Unknown recording_id"):
        make_event_cases(
            X, metadata, [dict(events[0], recording_id="missing")], tmin=0, tmax=1
        )
    with pytest.raises(ValueError, match="sample grid"):
        make_event_cases(X, metadata, [dict(events[0], onset=0.1)], tmin=0, tmax=1)
    with pytest.raises(ValueError, match="greater than tmin"):
        make_event_cases(X, metadata, [], tmin=1, tmax=0)


def test_unequal_lengths_and_channel_reordering(recordings):
    """Keep unequal-length arrays while aligning their semantic channel order."""
    X, metadata = recordings
    X = [X[0], X[1, [2, 0, 1], :8]]
    metadata[1]["channel_names"] = ["Pz", "Fz", "Cz"]
    cases = make_recording_cases(X, metadata, labels=[0, 1], groups=[0, 1])
    with pytest.raises(ValueError, match="same output channel"):
        build_classification_dataset(X, metadata, cases)
    cases[1]["channel_names"] = ["Fz", "Cz", "Pz"]
    dataset = build_classification_dataset(X, metadata, cases)
    assert isinstance(dataset.X, list)
    assert [x.shape for x in dataset.X] == [(3, 12), (3, 8)]
    np.testing.assert_array_equal(dataset.X[1], X[1][[1, 2, 0]])
    assert not any(np.shares_memory(case, source) for case, source in zip(dataset.X, X))
    metadata[1]["sampling_frequency"] = 8
    with pytest.raises(ValueError, match="sampling frequency"):
        build_classification_dataset(X, metadata, cases)


@pytest.mark.parametrize(
    "update, match",
    [
        ({"start_sample": 0.5}, "must be an integer"),
        ({"start_sample": True}, "must be an integer"),
        ({"stop_sample": 0}, "greater than start_sample"),
        ({"label": None}, "label must be"),
        ({"group_id": ""}, "group_id must be"),
        ({"case_id": ""}, "case_id must be"),
        ({"recording_id": "missing"}, "Unknown recording_id"),
        ({"included": "false"}, "must be a boolean"),
        ({"included": False}, "need an exclusion_reason"),
        ({"exclusion_reason": "bad"}, "cannot have an exclusion_reason"),
        ({"channel_names": ["missing"]}, "unknown source channels"),
        ({"channel_names": ["Fz", "Fz"]}, "unique strings"),
        ({"output_channel_names": ["Fz"]}, "match channel count"),
    ],
)
def test_invalid_case_definitions(recordings, update, match):
    """Reject malformed definitions rather than silently repairing targets or bounds."""
    X, metadata = recordings
    cases = make_recording_cases(X, metadata, labels=[0, 1], groups=[0, 1])
    cases[0].update(update)
    with pytest.raises(ValueError, match=match):
        build_classification_dataset(X, metadata, cases)


def test_duplicates_missing_labels_empty_cases_and_nonfinite_data(recordings):
    """Fail closed for missing identities, targets and usable signals."""
    X, metadata = recordings
    cases = make_recording_cases(X, metadata, labels=[0, 1], groups=[0, 1])
    with pytest.raises(ValueError, match="unique nonempty"):
        build_classification_dataset(X, metadata, cases + cases[:1])
    with pytest.raises(ValueError, match="No included cases"):
        build_classification_dataset(X, metadata, [])
    with pytest.raises(ValueError, match="No included cases"):
        build_classification_dataset(
            X,
            metadata,
            [dict(c, included=False, exclusion_reason="bad") for c in cases],
        )
    with pytest.raises(ValueError, match="Missing required fields"):
        build_classification_dataset(
            X, metadata, [{k: v for k, v in cases[0].items() if k != "label"}]
        )
    with pytest.raises(ValueError, match="uniformly strings or integers"):
        build_classification_dataset(X, metadata, [cases[0], dict(cases[1], label="1")])
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_classification_dataset(X, metadata, cases)


@pytest.mark.parametrize(
    "update, match",
    [
        ({"sampling_frequency": 0}, "finite and positive"),
        ({"sampling_frequency": np.nan}, "finite and positive"),
        ({"n_samples": 100}, "n_samples does not match"),
        ({"channel_names": ["Fz"]}, "channel metadata does not match"),
        ({"filepath": "sub-1/eeg/recording.edf"}, "unique nonempty"),
    ],
)
def test_invalid_recording_metadata(recordings, update, match):
    """Validate metadata before interpreting time or channel coordinates."""
    X, metadata = recordings
    metadata[0].update(update)
    with pytest.raises(ValueError, match=match):
        make_recording_cases(X, metadata, labels=[0, 1], groups=[0, 1])


def test_target_and_recording_alignment(recordings):
    """Require explicitly aligned targets, grouping identifiers and metadata."""
    X, metadata = recordings
    with pytest.raises(ValueError, match="same length"):
        make_recording_cases(X, metadata[:1], labels=[0, 1], groups=[0, 1])
    with pytest.raises(ValueError, match="one value per recording"):
        make_recording_cases(X, metadata, labels=[0], groups=[0, 1])
    with pytest.raises(ValueError, match="one value per recording"):
        make_recording_cases(X, metadata, labels=[0, 1], groups="ab")
    with pytest.raises(ValueError, match="2D array"):
        make_recording_cases(
            X[0], metadata * 1 + metadata[:1], labels=[0, 1, 2], groups=[0, 1, 2]
        )


def test_grouped_split_and_subset_folds(recordings):
    """Validate folds without requiring that every case is assigned."""
    X, metadata = recordings
    cases = make_fixed_length_cases(
        X, metadata, labels=[0, 1], groups=[0, 1], window_size=1
    )
    dataset = build_classification_dataset(X, metadata, cases)
    assert validate_group_split(dataset, [0, 1, 2], [3, 4, 5]) is None
    assert validate_group_split(dataset, [0, 2], [4]) is None
    with pytest.raises(ValueError, match="share group_id"):
        validate_group_split(dataset, [0], [1])
    # Incorrect grouping by player/window still cannot split a source recording.
    dataset.cases[1]["group_id"] = 99
    dataset.groups[1] = 99
    with pytest.raises(ValueError, match="share recording_id"):
        validate_group_split(dataset, [0], [1])
    dataset.cases[0]["participant_id"] = "same-person"
    dataset.cases[3]["participant_id"] = "same-person"
    with pytest.raises(ValueError, match="share participant_id"):
        validate_group_split(dataset, [0], [3])


@pytest.mark.parametrize(
    "train, test",
    [
        ([0], [0]),
        ([0, 0], [1]),
        ([], [1]),
        ([0], [2]),
        ([-1], [1]),
        ([0.0], [1]),
        ([True], [1]),
        ([[0]], [1]),
    ],
)
def test_invalid_split_indices(recordings, train, test):
    """Reject overlapping, duplicated, empty and out-of-range split indices."""
    X, metadata = recordings
    dataset = build_classification_dataset(
        X,
        metadata,
        make_recording_cases(X, metadata, labels=[0, 1], groups=[0, 1]),
    )
    with pytest.raises(ValueError):
        validate_group_split(dataset, train, test)


def test_generated_edf_loader_integration(tmp_path):
    """Catch loader/builder contract regressions using small EDF files in CI."""
    from aeon_neuro.datasets.load_bids import load_bids_data
    from aeon_neuro.datasets.tests.test_load_bids_data import create_recording

    create_recording(tmp_path, subject="01", n_times=100, sfreq=100)
    create_recording(tmp_path, subject="02", n_times=200, sfreq=100)
    X, metadata = load_bids_data(tmp_path)
    cases = make_fixed_length_cases(
        X,
        metadata,
        labels=["a", "b"],
        groups=["01", "02"],
        window_size=0.5,
    )
    dataset = build_classification_dataset(X, metadata, cases)
    assert dataset.X.shape == (6, 2, 50)
    assert dataset.y.tolist() == ["a", "a", "b", "b", "b", "b"]
    validate_group_split(dataset, [0, 1], [2, 3, 4, 5])


def test_numpy_target_scalars_and_explicit_recording_ids(recordings):
    """Accept ordinary NumPy scalar targets without coercing mixed label types."""
    X, metadata = recordings
    for i, item in enumerate(metadata):
        item["recording_id"] = f"recording-{i}"
        del item["filepath"]
    cases = make_recording_cases(
        X, metadata, labels=[np.str_("a"), "b"], groups=[np.int64(1), 2]
    )
    dataset = build_classification_dataset(X, metadata, cases)
    assert dataset.y.tolist() == ["a", "b"]
    assert dataset.cases[0]["recording_id"] == "recording-0"
    assert dataset.groups.tolist() == [1, 2]
    with pytest.raises(ValueError, match="unique strings"):
        build_classification_dataset(X, metadata, [dict(cases[0], channel_names=None)])
    with pytest.raises(ValueError, match="Unknown recording_id"):
        build_classification_dataset(X, metadata, [dict(cases[0], recording_id=[])])
