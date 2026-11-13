"""Keep supplied channel facts distinct from source sidecar availability."""

import pytest

from aeon_neuro.datasets import EEGDatasetInfo


@pytest.mark.parametrize(
    "paths", [None, {}, {"events": "events.tsv"}, {"channels": "channels.tsv"}]
)
def test_partial_sidecar_paths_preserve_explicit_channel_metadata(paths):
    """An unspecified channel path cannot hide supplied source channel rows."""
    rows = [dict(name="C3", type="EEG", status="bad", units="V")]
    record = dict(
        recording_id="r",
        participant_id="01",
        channel_names=["C3"],
        sampling_frequency=128,
        n_samples=16,
        channel_metadata=rows,
    )
    if paths is not None:
        record["sidecar_paths"] = paths
    info = EEGDatasetInfo(recordings=[record])
    channel_info = info.recordings[0]
    assert channel_info.channels == rows
    assert channel_info.bad_channels() == ("C3",)
    assert channel_info.channels_of_type("EEG") == ("C3",)
    assert channel_info.channels_frame["name"].tolist() == ["C3"]
    assert EEGDatasetInfo(info.to_dict()).recordings[0].channels == rows


@pytest.mark.parametrize("rows", [[], None])
def test_explicit_absent_channel_sidecar_stays_unavailable(rows):
    """Inspector-normalized empty rows cannot assert a known channel status."""
    record = dict(
        recording_id="r",
        participant_id="01",
        channel_names=["C3"],
        sampling_frequency=128,
        n_samples=16,
        channel_metadata=rows,
        sidecar_paths={"events": "events.tsv", "channels": None},
    )
    channel_info = EEGDatasetInfo(recordings=[record]).recordings[0]
    assert channel_info.channels is None
    assert channel_info.channels_frame is None
    with pytest.raises(ValueError, match="unavailable"):
        channel_info.bad_channels()
