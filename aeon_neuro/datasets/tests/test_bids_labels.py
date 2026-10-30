"""Tests for BIDS EEG classification labels."""

import os

import numpy as np
import pandas as pd
import pytest

from aeon_neuro.datasets.bids_labels import label_bids_recordings


def create_record(
    root,
    subject="01",
    session="01",
    task="rest",
    run="01",
    condition=None,
    events=None,
):
    """Create a synthetic BIDS recording and optional TSV files."""
    eeg_dir = root / f"sub-{subject}"

    if session is not None:
        eeg_dir /= f"ses-{session}"

    eeg_dir /= "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    entities = [f"sub-{subject}"]

    if session is not None:
        entities.append(f"ses-{session}")
    if task is not None:
        entities.append(f"task-{task}")
    if run is not None:
        entities.append(f"run-{run}")

    filename = "_".join(entities) + "_eeg.edf"
    filepath = eeg_dir / filename

    record = {
        "filepath": str(filepath.relative_to(root)),
        "filename": filename,
        "subject": subject,
        "session": session,
        "task": task,
        "acq": None,
        "run": run,
        "sfreq": 100,
    }

    if condition is not None:
        pd.DataFrame({"condition": [condition]}).to_csv(
            eeg_dir / filename.replace("_eeg.edf", "_eeg.tsv"),
            sep="\t",
            index=False,
        )

    if events is not None:
        pd.DataFrame(events).to_csv(
            eeg_dir / filename.replace("_eeg.edf", "_events.tsv"),
            sep="\t",
            index=False,
        )

    return record


def recording_tsv_path(root, record):
    """Return the recording-specific TSV path."""
    return root / record["filepath"].replace("_eeg.edf", "_eeg.tsv")


def call_labels(X, metadata, tmp_path, **kwargs):
    """Call the labeling function with common test settings."""
    return label_bids_recordings(
        X,
        metadata,
        class_names=["LeftHand", "RightHand", "Rest"],
        bids_path=tmp_path,
        **kwargs,
    )


def test_recording_labels_preserve_order(tmp_path):
    """Test that recording labels preserve input order."""
    metadata = [
        create_record(tmp_path, task="RightHand"),
        create_record(tmp_path, task="Rest"),
        create_record(tmp_path, task="LeftHand"),
    ]
    X = np.arange(3 * 2 * 10).reshape(3, 2, 10)
    original = X.copy()

    X_out, y, counts, unmatched, details = call_labels(X, metadata, tmp_path)

    assert X_out is X
    np.testing.assert_array_equal(X_out, original)
    assert y.tolist() == ["RightHand", "Rest", "LeftHand"]
    assert counts == {"LeftHand": 1, "RightHand": 1, "Rest": 1}
    assert unmatched == 0
    assert len(details) == 3


def test_recording_labels_from_tsv_column(tmp_path):
    """Test recording labels sourced from a selected TSV column."""
    metadata = [
        create_record(tmp_path, task="recording_a", condition="LeftHand"),
        create_record(tmp_path, task="recording_b", condition="RightHand"),
    ]

    for record in metadata:
        tsv = recording_tsv_path(tmp_path, record)
        assert tsv.is_file(), f"Expected TSV file not found: {tsv}"

    _, y, counts, unmatched, details = call_labels(
        np.zeros((2, 2, 100)),
        metadata,
        tmp_path,
        label_column="condition",
    )

    assert y.tolist() == ["LeftHand", "RightHand"]
    assert counts == {"LeftHand": 1, "RightHand": 1, "Rest": 0}
    assert unmatched == 0
    assert len(details) == 2


def test_recording_labels_from_explicit_tsv(tmp_path):
    """Test that an explicit TSV labels only matching recordings."""
    metadata = [
        create_record(tmp_path, task="recording_a", condition="LeftHand"),
        create_record(tmp_path, task="recording_b", condition="RightHand"),
    ]

    selected_tsv = recording_tsv_path(tmp_path, metadata[0])

    _, y, counts, unmatched, details = call_labels(
        np.zeros((2, 2, 100)),
        metadata,
        tmp_path,
        label_source=str(selected_tsv),
        label_column="condition",
    )

    assert y.tolist() == ["LeftHand", None]
    assert counts == {"LeftHand": 1, "RightHand": 0, "Rest": 0}
    assert unmatched == 1
    assert details[0]["label_source_tsv"] is not None
    assert details[1]["label"] is None


def test_participant_tsv_labels(tmp_path):
    """Test recording labels sourced from participants.tsv."""
    metadata = [
        create_record(tmp_path, subject="01", task="recording_a"),
        create_record(tmp_path, subject="02", task="recording_b"),
    ]

    pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-02"],
            "group": ["LeftHand", "RightHand"],
        }
    ).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    _, y, counts, unmatched, details = call_labels(
        np.zeros((2, 2, 100)),
        metadata,
        tmp_path,
        label_source="participants",
        label_column="group",
    )

    assert y.tolist() == ["LeftHand", "RightHand"]
    assert counts == {"LeftHand": 1, "RightHand": 1, "Rest": 0}
    assert unmatched == 0
    assert details[0]["label_column"] == "group"
    assert details[0]["label_source_tsv"] == "participants.tsv"


def test_unmatched_recordings(tmp_path):
    """Test that recordings without matching classes remain unlabeled."""
    metadata = [
        create_record(tmp_path, task="unknown_a"),
        create_record(tmp_path, task="unknown_b"),
    ]

    _, y, counts, unmatched, _ = call_labels(
        np.zeros((2, 2, 100)),
        metadata,
        tmp_path,
    )

    assert y.tolist() == [None, None]
    assert counts == {"LeftHand": 0, "RightHand": 0, "Rest": 0}
    assert unmatched == 2


def test_ambiguous_recording(tmp_path):
    """Test that multiple matching classes produce no recording label."""
    metadata = [create_record(tmp_path, task="LeftHand")]

    _, y, counts, unmatched, details = label_bids_recordings(
        np.zeros((1, 2, 100)),
        metadata,
        class_names=["LeftHand", "Hand"],
        bids_path=tmp_path,
    )

    assert y.tolist() == [None]
    assert counts == {"LeftHand": 0, "Hand": 0}
    assert unmatched == 1
    assert details[0]["candidate_matches"] == ["Hand", "LeftHand"]


def test_tsv_column_controls_the_label(tmp_path):
    """Test that the selected TSV column alone determines the label."""
    metadata = [create_record(tmp_path, task="recording", condition="LeftHand")]
    tsv = recording_tsv_path(tmp_path, metadata[0])

    pd.DataFrame(
        {
            "condition": ["RightHand"],
            "description": ["LeftHand"],
        }
    ).to_csv(tsv, sep="\t", index=False)

    _, y, counts, unmatched, _ = label_bids_recordings(
        np.zeros((1, 2, 100)),
        metadata,
        class_names=["LeftHand", "RightHand"],
        bids_path=tmp_path,
        label_source=str(tsv),
        label_column="condition",
    )

    assert y.tolist() == ["RightHand"]
    assert counts == {"LeftHand": 0, "RightHand": 1}
    assert unmatched == 0


def test_event_labels_and_tsv_provenance(tmp_path):
    """Test event labels, counts, and source TSV provenance."""
    metadata = [
        create_record(
            tmp_path,
            task="motor",
            events={
                "onset": [0.5, 1.5, 2.5],
                "duration": [0.2, 0.2, 0.2],
                "trial_type": ["LeftHand", "RightHand", "Unknown"],
            },
        )
    ]

    _, events, counts, unmatched, details = call_labels(
        [np.zeros((2, 1000))],
        metadata,
        tmp_path,
        label_level="event",
    )

    assert events["label"].iloc[:2].tolist() == [
        "LeftHand",
        "RightHand",
    ]
    assert pd.isna(events["label"].iloc[2])
    assert events["onset_seconds"].tolist() == [0.5, 1.5, 2.5]
    assert counts == {"LeftHand": 1, "RightHand": 1, "Rest": 0}
    assert unmatched == 1

    expected = os.path.join(
        "sub-01",
        "ses-01",
        "eeg",
        "sub-01_ses-01_task-motor_run-01_events.tsv",
    )
    assert events["source_tsv"].tolist() == [expected] * 3
    assert details[0]["label_source_tsv"] == expected


def test_event_segments_support_variable_durations(tmp_path):
    """Test segmentation of events with different durations."""
    metadata = [
        create_record(
            tmp_path,
            task="motor",
            events={
                "onset": [0.1, 0.5],
                "duration": [0.2, 0.4],
                "trial_type": ["LeftHand", "RightHand"],
            },
        )
    ]

    segments, events, _, _, _ = call_labels(
        [np.zeros((2, 200))],
        metadata,
        tmp_path,
        label_level="event",
    )

    assert [segment.shape[1] for segment in segments] == [20, 40]
    assert events["start_sample"].tolist() == [10, 50]
    assert events["end_sample"].tolist() == [30, 90]


def test_event_without_events_tsv(tmp_path):
    """Test event labeling when a recording has no events TSV."""
    metadata = [create_record(tmp_path, task="rest")]

    _, events, counts, unmatched, details = call_labels(
        [np.zeros((2, 100))],
        metadata,
        tmp_path,
        label_level="event",
    )

    assert events.empty
    assert counts == {"LeftHand": 0, "RightHand": 0, "Rest": 0}
    assert unmatched == 0
    assert details[0]["label_source_tsv"] is None


def test_event_missing_label_column(tmp_path):
    """Test that a missing event label column raises an error."""
    metadata = [
        create_record(
            tmp_path,
            task="motor",
            events={"onset": [1.0], "condition": ["LeftHand"]},
        )
    ]

    with pytest.raises(ValueError, match="trial_type"):
        call_labels(
            [np.zeros((2, 200))],
            metadata,
            tmp_path,
            label_level="event",
        )


def test_event_invalid_onset(tmp_path):
    """Test that invalid event onsets raise an error."""
    metadata = [
        create_record(
            tmp_path,
            task="motor",
            events={
                "onset": ["not-a-number"],
                "trial_type": ["LeftHand"],
            },
        )
    ]

    with pytest.raises(ValueError, match="onset"):
        call_labels(
            [np.zeros((2, 200))],
            metadata,
            tmp_path,
            label_level="event",
        )


def test_event_explicit_tsv_path(tmp_path):
    """Test event labeling from an explicitly supplied TSV path."""
    metadata = [
        create_record(
            tmp_path,
            task="motor",
            events={"onset": [1.0], "trial_type": ["LeftHand"]},
        )
    ]

    custom_tsv = tmp_path / "custom_events.tsv"
    pd.DataFrame(
        {
            "onset": [2.0],
            "trial_type": ["RightHand"],
        }
    ).to_csv(custom_tsv, sep="\t", index=False)

    _, events, counts, unmatched, details = call_labels(
        [np.zeros((2, 300))],
        metadata,
        tmp_path,
        label_level="event",
        label_source=str(custom_tsv),
    )

    assert events["label"].tolist() == ["RightHand"]
    assert events["onset_seconds"].tolist() == [2.0]
    assert counts == {"LeftHand": 0, "RightHand": 1, "Rest": 0}
    assert unmatched == 0
    assert details[0]["label_source_tsv"] == "custom_events.tsv"


def test_missing_tsv_column_raises(tmp_path):
    """Test that a missing recording TSV column raises an error."""
    metadata = [create_record(tmp_path, task="recording", condition="LeftHand")]
    tsv = recording_tsv_path(tmp_path, metadata[0])

    with pytest.raises(ValueError, match="not found"):
        label_bids_recordings(
            np.zeros((1, 2, 100)),
            metadata,
            class_names=["LeftHand", "RightHand"],
            bids_path=tmp_path,
            label_source=str(tsv),
            label_column="missing_column",
        )


def test_missing_explicit_tsv_gives_unmatched_label(tmp_path):
    """Test that a missing explicit TSV leaves the label unmatched."""
    metadata = [create_record(tmp_path, task="recording")]

    _, y, counts, unmatched, details = call_labels(
        np.zeros((1, 2, 100)),
        metadata,
        tmp_path,
        label_source="missing.tsv",
    )

    assert y.tolist() == [None]
    assert counts == {"LeftHand": 0, "RightHand": 0, "Rest": 0}
    assert unmatched == 1
    assert details[0]["label"] is None


def test_class_counts_include_all_requested_classes(tmp_path):
    """Test that class counts include classes with zero examples."""
    metadata = [create_record(tmp_path, task="Rest")]

    _, y, counts, unmatched, _ = call_labels(
        np.zeros((1, 2, 100)),
        metadata,
        tmp_path,
    )

    assert y.tolist() == ["Rest"]
    assert list(counts) == ["LeftHand", "RightHand", "Rest"]
    assert counts == {"LeftHand": 0, "RightHand": 0, "Rest": 1}
    assert unmatched == 0


def test_empty_class_names_raise(tmp_path):
    """Test that empty class names are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        label_bids_recordings(
            np.zeros((1, 2, 100)),
            [create_record(tmp_path)],
            class_names=[],
            bids_path=tmp_path,
        )


def test_duplicate_class_names_raise(tmp_path):
    """Test that duplicate class names are rejected."""
    with pytest.raises(ValueError, match="duplicates"):
        label_bids_recordings(
            np.zeros((1, 2, 100)),
            [create_record(tmp_path)],
            class_names=["Rest", "Rest"],
            bids_path=tmp_path,
        )


def test_empty_class_name_raises(tmp_path):
    """Test that empty individual class names are rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        label_bids_recordings(
            np.zeros((1, 2, 100)),
            [create_record(tmp_path)],
            class_names=["Rest", ""],
            bids_path=tmp_path,
        )


def test_mismatched_input_lengths_raise(tmp_path):
    """Test that recordings and metadata must have equal lengths."""
    with pytest.raises(ValueError, match="same length"):
        label_bids_recordings(
            np.zeros((2, 2, 100)),
            [create_record(tmp_path)],
            class_names=["Rest"],
            bids_path=tmp_path,
        )


def test_invalid_label_level_raises(tmp_path):
    """Test that unsupported label levels are rejected."""
    with pytest.raises(ValueError, match="label_level"):
        label_bids_recordings(
            np.zeros((1, 2, 100)),
            [create_record(tmp_path)],
            class_names=["Rest"],
            bids_path=tmp_path,
            label_level="epoch",
        )


def test_temp_labels_with_extra_tsv_columns(tmp_path):
    """Additional TSV columns do not interfere with label extraction."""
    metadata = [
        create_record(tmp_path, task="recording_a", condition="LeftHand"),
    ]

    tsv = recording_tsv_path(tmp_path, metadata[0])
    df = pd.read_csv(tsv, sep="\t")
    df["extra_column"] = "irrelevant"
    df["another_column"] = 123
    df.to_csv(tsv, sep="\t", index=False)

    _, y, _, unmatched, _ = call_labels(
        np.zeros((1, 2, 100)),
        metadata,
        tmp_path,
        label_column="condition",
    )

    assert y.tolist() == ["LeftHand"]
    assert unmatched == 0


def test_temp_labels_do_not_leak_between_recordings(tmp_path):
    """A missing label in one TSV does not inherit another recording's label."""
    metadata = [
        create_record(tmp_path, task="recording_a", condition="LeftHand"),
        create_record(tmp_path, task="recording_b", condition="RightHand"),
    ]

    second_tsv = recording_tsv_path(tmp_path, metadata[1])
    pd.DataFrame({"condition": [np.nan]}).to_csv(second_tsv, sep="\t", index=False)

    _, y, _, unmatched, details = call_labels(
        np.zeros((2, 2, 100)),
        metadata,
        tmp_path,
        label_column="condition",
    )

    assert y[0] == "LeftHand"
    assert y[1] is None
    assert unmatched == 1
    assert details[0]["label"] == "LeftHand"
    assert details[1]["label"] is None


def test_temp_label_counts_exclude_unmatched_recordings(tmp_path):
    """Counts reflect assigned labels and exclude missing TSVs."""
    metadata = [
        create_record(tmp_path, task="recording_a", condition="LeftHand"),
        create_record(tmp_path, task="recording_b", condition="RightHand"),
        create_record(tmp_path, task="recording_c", condition="Rest"),
    ]

    # Remove the third TSV to create an unmatched recording.
    third_tsv = recording_tsv_path(tmp_path, metadata[2])
    third_tsv.unlink()

    _, y, counts, unmatched, _ = call_labels(
        np.zeros((3, 2, 100)),
        metadata,
        tmp_path,
        label_column="condition",
    )

    assert y.tolist() == ["LeftHand", "RightHand", None]
    assert unmatched == 1
    assert counts["LeftHand"] == 1
    assert counts["RightHand"] == 1
    assert counts["Rest"] == 0
