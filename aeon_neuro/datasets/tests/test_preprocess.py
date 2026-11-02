"""Tests for EEG preprocessing, including filtering, channel selection, and epoching."""

import numpy as np
import pytest

from aeon_neuro.datasets.preprocess import preprocess_eeg


@pytest.fixture
def sample_data():
    """Create reproducible synthetic EEG recordings and metadata."""
    sfreq = 250
    rng = np.random.default_rng(42)

    X = rng.normal(size=(2, 4, sfreq * 10)) * 1e-5
    y = np.array(["LeftHand", "RightHand"], dtype=object)

    metadata = [
        {
            "filepath": f"sub-01_task-test_run-{i:02d}_eeg.edf",
            "filename": f"run-{i:02d}_eeg.edf",
            "subject": "01",
            "task": "test",
            "run": f"{i:02d}",
            "channel_names": ["C3", "C4", "Cz", "EOG"],
            "sampling_frequency": sfreq,
        }
        for i in (1, 2)
    ]

    return X, y, metadata


def test_preprocess_preserves_shape_without_epoching(sample_data):
    """Continuous preprocessing preserves recording dimensions and labels."""
    X, y, metadata = sample_data

    X_out, y_out, meta_out = preprocess_eeg(X, y, metadata, l_freq=1.0, h_freq=40.0)

    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == X.shape
    np.testing.assert_array_equal(y_out, y)
    assert len(meta_out) == len(metadata)


def test_preprocess_can_disable_filtering(sample_data):
    """Disabling both filters preserves the input signal."""
    X, y, metadata = sample_data

    X_out, y_out, _ = preprocess_eeg(X, y, metadata, l_freq=None, h_freq=None)

    np.testing.assert_allclose(X_out, X)
    np.testing.assert_array_equal(y_out, y)


def test_preprocess_applies_filtering(sample_data):
    """Filtering changes the signal without changing its dimensions."""
    X, y, metadata = sample_data

    X_out, _, _ = preprocess_eeg(X, y, metadata, l_freq=1.0, h_freq=40.0)

    assert X_out.shape == X.shape
    assert not np.allclose(X_out, X)


def test_preprocess_includes_selected_channels(sample_data):
    """Only requested channels remain, in their original order."""
    X, y, metadata = sample_data

    X_out, _, meta_out = preprocess_eeg(
        X,
        y,
        metadata,
        include_channels=["C3", "Cz"],
        l_freq=None,
        h_freq=None,
    )

    assert X_out.shape == (2, 2, 2500)
    assert meta_out[0]["channel_names"] == ["C3", "Cz"]


def test_preprocess_excludes_selected_channels(sample_data):
    """Excluded channels are removed from the output."""
    X, y, metadata = sample_data

    X_out, _, meta_out = preprocess_eeg(
        X,
        y,
        metadata,
        exclude_channels=["EOG", "C4"],
        l_freq=None,
        h_freq=None,
    )

    assert X_out.shape == (2, 2, 2500)
    assert meta_out[0]["channel_names"] == ["C3", "Cz"]


def test_preprocess_rejects_missing_included_channel(sample_data):
    """Requesting a channel absent from a recording raises ValueError."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="missing requested channels"):
        preprocess_eeg(
            X,
            y,
            metadata,
            include_channels=["C3", "UNKNOWN"],
        )


def test_preprocess_rejects_selection_that_removes_all_channels(sample_data):
    """Removing every available channel raises ValueError."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="No channels remain"):
        preprocess_eeg(
            X,
            y,
            metadata,
            exclude_channels=["C3", "C4", "Cz", "EOG"],
        )


def test_preprocess_rejects_mismatched_input_lengths(sample_data):
    """X, y and metadata must contain the same number of recordings."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="equal lengths"):
        preprocess_eeg(X, y[:1], metadata)


def test_preprocess_rejects_non_three_dimensional_array(sample_data):
    """A NumPy input must have recording, channel and time dimensions."""
    _, y, metadata = sample_data

    with pytest.raises(
        ValueError,
        match=r"X must have shape \(n_recordings, n_channels, n_times\)",
    ):
        preprocess_eeg(np.zeros((2, 100)), y, metadata)


def test_preprocess_rejects_invalid_recording_dimensions(sample_data):
    """Each individual recording must be two-dimensional."""
    _, y, metadata = sample_data

    with pytest.raises(ValueError, match="must have shape"):
        preprocess_eeg(
            [np.zeros(100), np.zeros((4, 100))],
            y,
            metadata,
            l_freq=None,
            h_freq=None,
        )


def test_preprocess_rejects_channel_count_mismatch(sample_data):
    """Channel metadata must match the number of data channels."""
    X, y, metadata = sample_data
    metadata[0]["channel_names"] = ["C3", "C4"]

    with pytest.raises(ValueError, match="channel count"):
        preprocess_eeg(X, y, metadata)


def test_preprocess_rejects_missing_channel_metadata(sample_data):
    """Missing channel names raise ValueError."""
    X, y, metadata = sample_data
    del metadata[0]["channel_names"]

    with pytest.raises(ValueError, match="no channel_names"):
        preprocess_eeg(X, y, metadata)


def test_preprocess_rejects_invalid_sampling_frequency(sample_data):
    """A missing or non-positive sampling frequency raises ValueError."""
    X, y, metadata = sample_data
    metadata[0]["sampling_frequency"] = 0

    with pytest.raises(ValueError, match="sampling frequency"):
        preprocess_eeg(X, y, metadata)


def test_preprocess_rejects_duplicate_channel_names(sample_data):
    """Duplicate channel names raise ValueError."""
    X, y, metadata = sample_data
    metadata[0]["channel_names"] = ["C3", "C3", "Cz", "EOG"]

    with pytest.raises(ValueError, match="duplicate channel names"):
        preprocess_eeg(X, y, metadata)


def test_preprocess_rejects_invalid_filter_range(sample_data):
    """The high-pass cutoff must be lower than the low-pass cutoff."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="l_freq must be lower"):
        preprocess_eeg(X, y, metadata, l_freq=40.0, h_freq=1.0)


@pytest.mark.parametrize("cutoff", [0, -1])
def test_preprocess_rejects_nonpositive_filter_cutoffs(sample_data, cutoff):
    """Filter cutoffs must be positive or None."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError):
        preprocess_eeg(X, y, metadata, l_freq=cutoff)


def test_preprocess_rejects_cutoff_at_nyquist_frequency(sample_data):
    """Filter cutoffs must be below the recording's Nyquist frequency."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="Nyquist"):
        preprocess_eeg(X, y, metadata, h_freq=125.0)


def test_preprocess_epochs_recordings_and_repeats_labels(sample_data):
    """Each epoch inherits the label of its source recording."""
    X, y, metadata = sample_data

    X_out, y_out, meta_out = preprocess_eeg(
        X,
        y,
        metadata,
        l_freq=None,
        h_freq=None,
        epoch_duration=2.0,
    )

    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == (10, 4, 500)
    assert y_out.tolist() == (["LeftHand"] * 5 + ["RightHand"] * 5)
    assert len(meta_out) == 10


def test_preprocess_records_epoch_metadata(sample_data):
    """Epoch metadata records source recording, index and time boundaries."""
    X, y, metadata = sample_data

    _, _, meta_out = preprocess_eeg(
        X,
        y,
        metadata,
        l_freq=None,
        h_freq=None,
        epoch_duration=2.0,
    )

    assert meta_out[0]["source_recording"] == metadata[0]["filepath"]
    assert meta_out[0]["epoch_index"] == 0
    assert meta_out[0]["start_time"] == 0.0
    assert meta_out[0]["end_time"] == 2.0

    assert meta_out[1]["epoch_index"] == 1
    assert meta_out[1]["start_time"] == 2.0
    assert meta_out[1]["end_time"] == 4.0


def test_preprocess_supports_overlapping_epochs(sample_data):
    """Overlapping epochs use the configured step size."""
    X, y, metadata = sample_data

    X_out, y_out, _ = preprocess_eeg(
        X,
        y,
        metadata,
        l_freq=None,
        h_freq=None,
        epoch_duration=2.0,
        epoch_overlap=1.0,
    )

    assert X_out.shape == (18, 4, 500)
    assert len(y_out) == 18


def test_preprocess_rejects_nonpositive_epoch_duration(sample_data):
    """Epoch duration must be positive when specified."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="epoch_duration must be positive"):
        preprocess_eeg(
            X,
            y,
            metadata,
            l_freq=None,
            h_freq=None,
            epoch_duration=0,
        )


def test_preprocess_rejects_negative_epoch_overlap(sample_data):
    """Epoch overlap cannot be negative."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="cannot be negative"):
        preprocess_eeg(
            X,
            y,
            metadata,
            l_freq=None,
            h_freq=None,
            epoch_overlap=-0.1,
        )


def test_preprocess_rejects_overlap_equal_to_epoch_duration(sample_data):
    """Overlap must be shorter than the epoch duration."""
    X, y, metadata = sample_data

    with pytest.raises(ValueError, match="smaller than epoch_duration"):
        preprocess_eeg(
            X,
            y,
            metadata,
            l_freq=None,
            h_freq=None,
            epoch_duration=2.0,
            epoch_overlap=2.0,
        )


def test_preprocess_rejects_recordings_too_short_for_epochs(sample_data):
    """Preprocessing fails if no complete epoch can be produced."""
    _, y, metadata = sample_data
    short_recordings = np.zeros((2, 4, 100))

    with pytest.raises(ValueError, match="produced no output"):
        preprocess_eeg(
            short_recordings,
            y,
            metadata,
            l_freq=None,
            h_freq=None,
            epoch_duration=2.0,
        )


def test_preprocess_accepts_list_of_recordings(sample_data):
    """A list of two-dimensional recordings is supported."""
    X, y, metadata = sample_data

    X_out, y_out, meta_out = preprocess_eeg(
        [X[0], X[1]],
        y,
        metadata,
        l_freq=None,
        h_freq=None,
    )

    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == X.shape
    assert y_out.tolist() == y.tolist()
    assert len(meta_out) == 2


def test_preprocess_does_not_mutate_input_metadata(sample_data):
    """Channel selection must not modify the caller's metadata."""
    X, y, metadata = sample_data
    original_metadata = [
        {**item, "channel_names": item["channel_names"].copy()} for item in metadata
    ]

    preprocess_eeg(
        X,
        y,
        metadata,
        include_channels=["C3"],
        l_freq=None,
        h_freq=None,
    )

    assert metadata == original_metadata


def test_preprocess_returns_list_for_incompatible_recordings(sample_data):
    """Different channel configurations can produce list output."""
    X, y, metadata = sample_data

    metadata[1]["channel_names"] = ["F3", "F4", "Fz", "EOG"]

    X_out, _, meta_out = preprocess_eeg(
        X,
        y,
        metadata,
        l_freq=None,
        h_freq=None,
    )

    assert isinstance(X_out, list)
    assert len(X_out) == 2
    assert X_out[0].shape == X_out[1].shape
    assert meta_out[0]["channel_names"] != meta_out[1]["channel_names"]


def test_preprocess_returns_list_for_different_sampling_frequencies(sample_data):
    """Different sampling frequencies prevent array stacking."""
    X, y, metadata = sample_data
    metadata[1]["sampling_frequency"] = 200

    X_out, _, _ = preprocess_eeg(
        X,
        y,
        metadata,
        l_freq=None,
        h_freq=None,
    )

    assert isinstance(X_out, list)
    assert len(X_out) == 2
