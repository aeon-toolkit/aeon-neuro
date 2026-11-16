"""Stage calibrated MATLAB signals with their complete BIDS event tables."""

import json
import tempfile
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from mne_bids import BIDSPath

from aeon_neuro.utils._bids_metadata import (
    _json_bytes,
    _publish,
    _read_table,
    _table_bytes,
)
from aeon_neuro.utils.matlab import (
    _cell,
    _raw_signature,
    _validate_event_fields,
    _validate_metadata,
)


def _convert_matlab_recording(recording, destination, format, allow_preload, overwrite):
    """Reuse the existing signal writer while preserving structured event data."""
    from aeon_neuro.utils.bids import convert_eeg_to_bids

    if (
        not isinstance(destination, BIDSPath)
        or not destination.root
        or not destination.subject
        or not destination.task
    ):
        raise ValueError("MATLAB export needs a BIDSPath with root, subject and task.")
    if destination.datatype not in (None, "eeg") or destination.suffix not in (
        None,
        "eeg",
    ):
        raise ValueError("Continuous MATLAB export is limited to EEG recordings.")
    if not allow_preload or format != "BrainVision":
        raise ValueError(
            "MATLAB recordings require allow_preload=True and "
            "format='BrainVision'; other export formats are not supported."
        )
    if overwrite:
        raise ValueError(
            "MATLAB export protects existing recordings; use overwrite=False and "
            "a new destination."
        )
    _validate_recording(recording)
    raw = recording.raw
    rows = recording.events
    # Native markers aid non-BIDS readers; the BIDS table remains authoritative.
    # Its source values and durations are not restricted to MNE integer codes.
    event_id = {
        name: i + 1
        for i, name in enumerate(dict.fromkeys(row["trial_type"] for row in rows))
    }
    events = np.asarray(
        [[row["sample"], 0, event_id[row["trial_type"]]] for row in rows], dtype=int
    ).reshape(-1, 3)
    destination = destination.copy().update(datatype="eeg", suffix="eeg")
    with tempfile.TemporaryDirectory(prefix="aeon-matlab-bids-") as folder:
        staged = destination.copy().update(root=folder)
        written = convert_eeg_to_bids(
            raw,
            staged,
            events=events if len(events) else None,
            event_id=event_id if len(events) else None,
            format=format,
            allow_preload=True,
        )
        eeg_json = written.copy().update(extension=".json").fpath
        acquisition = json.loads(eeg_json.read_text(encoding="utf-8"))
        acquisition.update(recording.metadata)
        # MNE-BIDS infers this from the export encoding, which need not identify
        # the original acquisition hardware (Tremmel combines two devices).
        acquisition["Manufacturer"] = recording.metadata.get("Manufacturer", "n/a")
        acquisition["MatlabSource"] = recording.provenance
        acquisition["ConversionDiagnostics"] = recording.diagnostics
        eeg_json.write_bytes(_json_bytes(acquisition))
        channel_path = written.copy().update(suffix="channels", extension=".tsv").fpath
        channel_fields, channels = _read_table(channel_path)
        for channel in channels:
            for column, key in [
                ("low_cutoff", "HighpassHz"),
                ("high_cutoff", "LowpassHz"),
            ]:
                channel[column] = _cell(recording.provenance[key])
            # A new RawArray has no quality assessment. Its default empty bads
            # list does not establish that every source signal is good.
            channel["status"] = "bad" if channel["name"] in raw.info["bads"] else "n/a"
        channel_path.write_bytes(_table_bytes(channel_fields, channels))
        if rows:
            fields = ["onset", "duration", "trial_type", "sample"]
            fields += sorted(set().union(*(row.keys() for row in rows)) - set(fields))
            table = [{k: _cell(row.get(k)) for k in fields} for row in rows]
            written.copy().update(suffix="events", extension=".tsv").fpath.write_bytes(
                _table_bytes(fields, table)
            )
            dictionary = {
                "onset": {
                    "Description": "Event onset on the exported sample grid.",
                    "Units": "s",
                },
                "duration": {
                    "Description": (
                        "Source event duration, retained even when the recording ends"
                        " during the event."
                    ),
                    "Units": "s",
                },
            }
            dictionary.update(
                {
                    k: {"Description": v} if isinstance(v, str) else v
                    for k, v in recording.event_descriptions.items()
                }
            )
            written.copy().update(suffix="events", extension=".json").fpath.write_bytes(
                _json_bytes(dictionary)
            )
        _publish(Path(folder), destination.root, {})
        return written.copy().update(root=destination.root)


def _validate_recording(recording):
    """Reject stale sample coordinates and nonserializable data before writing."""
    raw, source = recording.raw, recording.provenance
    if (
        raw.first_samp != 0
        or len(raw.annotations)
        or raw.n_times != source["SampleCount"]
        or raw.ch_names != source["ChannelNames"]
        or raw.get_channel_types() != source["ChannelTypes"]
        or raw.info["sfreq"] != source["SamplingFrequency"]
    ):
        raise ValueError(
            "MATLAB recording layout/timing changed after loading; rebuild its "
            "event mapping before export."
        )
    if _raw_signature(raw) != recording._raw_signature:
        raise ValueError(
            "MATLAB signals or filter settings changed after loading; export "
            "the original recording and preprocess a separate Raw copy."
        )
    _validate_metadata(recording.metadata)
    frequency = recording.metadata.get("PowerLineFrequency")
    frequency = None if frequency == "n/a" else frequency
    if frequency != raw.info["line_freq"]:
        raise ValueError("MATLAB metadata and Raw mains frequency disagree.")
    for row in recording.events:
        if not {"onset", "duration", "sample", "trial_type"} <= row.keys():
            raise ValueError("MATLAB events are missing required columns.")
        sample, onset = row["sample"], row["onset"]
        if (
            not isinstance(sample, Integral)
            or isinstance(sample, bool)
            or not 0 <= sample < raw.n_times
            or not isinstance(onset, Real)
            or not np.isfinite(onset)
            or abs(onset * raw.info["sfreq"] - sample) > 1e-6
        ):
            raise ValueError(
                "MATLAB event sample and onset must agree within the recording."
            )
        _validate_event_fields(row, recording.event_descriptions)
    json.dumps(
        [recording.provenance, recording.diagnostics, recording.event_descriptions],
        allow_nan=False,
    )
