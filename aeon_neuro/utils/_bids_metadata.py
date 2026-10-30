"""Preserve metadata when copying an identified, unchanged BIDS EEG recording."""

import csv
import io
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

from aeon_neuro.utils._bids_json import _read_eeg_json
from aeon_neuro.utils.bids import _preserve_stim_units
from aeon_neuro.utils.data_io import _check_edf_layout

# These describe acquisition. Duration, sampling rate and channel counts must
# continue to describe the decoded signal rather than a possibly stale template.
_DESCRIPTIVE_FIELDS = {
    "Manufacturer",
    "ManufacturersModelName",
    "EEGReference",
    "EEGGround",
    "EEGPlacementScheme",
    "TaskDescription",
    "TaskName",
    "InstitutionAddress",
    "InstitutionName",
    "InstitutionalDepartmentName",
    "SoftwareFilters",
    "HardwareFilters",
    "PowerLineFrequency",
}


def _read_table(path):
    """Read a TSV without interpreting behavioral labels or missing values."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(None in r or any(v is None for v in r.values()) for r in rows)
    ):
        raise ValueError(f"Invalid TSV columns: {path}")
    return fields, rows


def _table_bytes(fields, rows):
    """Serialize selected rows, preserving column order and string values."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value):
    """Write standard JSON, rejecting non-finite metadata values."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _validate_events(path, raw):
    """Reject timing that cannot describe this unchanged recording."""
    _, rows = _read_table(path)
    sfreq, span = raw.info["sfreq"], raw.n_times / raw.info["sfreq"]
    for number, row in enumerate(rows, 1):
        try:
            onset = float(row["onset"])
            duration = row["duration"]
            length = 0.0 if duration == "n/a" else float(duration)
            valid = (
                np.isfinite([onset, length]).all()
                and 0 <= onset < span
                and length >= 0
                and onset + length <= span + 1e-9
            )
            for column in ("sample", "onset_sample"):
                sample = row.get(column, "n/a")
                if sample != "n/a":
                    position = float(sample)
                    valid = valid and (
                        np.isfinite(position)
                        and position.is_integer()
                        and abs(position - onset * sfreq) <= 1e-6
                    )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"Event row {number} has invalid timing in {path}.")
    if any(row.get("trial_type") == "discontinuity" for row in rows):
        warnings.warn(
            f"{path}: published discontinuity events are preserved; review "
            "intervals before downstream analysis. Gap lengths are unknown.",
            UserWarning,
            stacklevel=3,
        )


def _participants(source, destination):
    """Retain the selected participant row and protect previously copied rows."""
    path = source.root / "participants.tsv"
    if not path.exists():
        return {}
    fields, rows = _read_table(path)
    selected = [r for r in rows if r.get("participant_id") == f"sub-{source.subject}"]
    if len(selected) != 1:
        raise ValueError(f"Expected one participant row for sub-{source.subject}.")
    target = destination.root / "participants.tsv"
    previous = []
    if target.exists():
        previous_fields, previous = _read_table(target)
        if previous_fields != fields:
            raise ValueError("Destination participant columns differ from the source.")
        matching = [
            r for r in previous if r["participant_id"] == selected[0]["participant_id"]
        ]
        if matching and matching != selected:
            raise ValueError(
                "Destination participant metadata conflicts with the source."
            )
        if matching:
            selected = []
    files = {target: _table_bytes(fields, previous + selected)}
    dictionary = source.root / "participants.json"
    if dictionary.exists():
        files[destination.root / dictionary.name] = dictionary.read_bytes()
    return files


def _publish(staging, destination, participants):
    """Check every destination before moving a completely prepared recording."""
    staged = [p for p in staging.rglob("*") if p.is_file()]
    for path in staged:
        target = destination / path.relative_to(staging)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Destination path escapes its dataset: {target}")
        if not target.exists():
            continue
        if path.suffix in (".vhdr", ".eeg", ".vmrk", ".edf", ".bdf", ".set", ".fdt"):
            raise FileExistsError(
                f"Destination recording file already exists: {target}"
            )
        if path.name.endswith("_scans.tsv") or (
            path.name == "participants.tsv" and target not in participants
        ):
            fields, incoming = _read_table(path)
            old_fields, previous = _read_table(target)
            key = "filename" if path.name.endswith("_scans.tsv") else "participant_id"
            if fields != old_fields:
                raise ValueError(f"Destination table columns differ: {target}")
            ids = {r[key]: r for r in previous}
            for row in incoming:
                if row[key] in ids and row != ids[row[key]]:
                    raise ValueError(f"Destination table row conflicts: {target}")
                if row[key] not in ids:
                    previous.append(row)
            path.write_bytes(_table_bytes(fields, previous))
        elif target not in participants and path.read_bytes() != target.read_bytes():
            raise FileExistsError(f"Conflicting destination file: {target}")
    # Writer failures and metadata conflicts leave the destination untouched.
    # Moving individual files is not a transaction against filesystem failures;
    # inspect an interrupted copy before retrying, and do not run concurrent appends.
    for path in staged:
        target = destination / path.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)


def _convert_bids_recording(source, destination, format, allow_preload, overwrite):
    """Copy one BIDS recording while retaining metadata outside MNE Raw."""
    if format != "auto" or allow_preload or overwrite:
        raise ValueError(
            "BIDSPath input requires format='auto', allow_preload=False and "
            "overwrite=False to preserve the original encoding and metadata."
        )
    if not isinstance(destination, BIDSPath):
        raise TypeError("The BIDS destination must be a BIDSPath.")
    if source.root is None or destination.root is None:
        raise ValueError("Source and destination BIDSPath must have a root.")
    if any(
        path.datatype not in (None, "eeg") or path.suffix not in (None, "eeg")
        for path in (source, destination)
    ):
        raise ValueError("BIDSPath preservation is limited to EEG recordings.")
    source = source.copy().update(datatype="eeg", suffix="eeg")
    destination = destination.copy().update(datatype="eeg", suffix="eeg")
    source_root, target_root = source.root.resolve(), destination.root.resolve()
    if (
        source_root == target_root
        or source_root in target_root.parents
        or target_root in source_root.parents
    ):
        raise ValueError(
            "Source and destination datasets must be separate, non-nested folders."
        )
    if not source.subject or not source.task or source.entities != destination.entities:
        raise ValueError(
            "BIDSPath input requires unchanged source recording identifiers."
        )
    extension = source.fpath.suffix.lower()
    if extension not in (".vhdr", ".edf", ".bdf", ".set"):
        raise ValueError(
            "BIDSPath input supports .vhdr, .edf, .bdf and continuous .set."
        )
    if destination.extension not in (None, extension):
        raise ValueError("BIDSPath input must retain the source recording extension.")
    source.update(extension=extension)
    destination.update(extension=extension)

    if extension in (".edf", ".bdf"):
        # MNE-BIDS bypasses the native loader, whose layout guard must still run
        # before a reader can infer a replacement record count or omit a tail.
        _check_edf_layout(source.fpath, 2 if extension == ".edf" else 3)
    metadata, provenance = _read_eeg_json(source)
    frequency = metadata.get("PowerLineFrequency")
    if frequency not in (None, "n/a") and (
        isinstance(frequency, bool)
        or not isinstance(frequency, (int, float))
        or not np.isfinite(frequency)
        or frequency <= 0
    ):
        raise ValueError("Published PowerLineFrequency must be finite and positive.")
    # MNE-BIDS handles channel/event tables, while effective EEG JSON fields
    # also include applicable ancestors missing from its selected sidecar.
    raw = read_raw_bids(source)
    if raw.preload:
        raise ValueError(
            "BIDSPath preservation requires a recording that can be read "
            "without preload."
        )
    if metadata.get("SamplingFrequency", raw.info["sfreq"]) != raw.info["sfreq"]:
        raise ValueError(
            "Published SamplingFrequency disagrees with the decoded signal."
        )
    if frequency not in (None, "n/a"):
        raw.info["line_freq"] = float(frequency)

    copied = {}
    sidecars = {}
    for suffix, ext in (
        ("events", ".tsv"),
        ("events", ".json"),
        ("channels", ".tsv"),
        ("electrodes", ".tsv"),
        ("coordsystem", ".json"),
    ):
        path = source.find_matching_sidecar(
            suffix=suffix, extension=ext, on_error="ignore"
        )
        sidecars[suffix, ext] = path
        if path is None:
            continue
        if suffix in ("electrodes", "coordsystem"):
            # Coordinate files are shared at subject/session scope and omit task.
            target = destination.root / path.relative_to(source.root)
        else:
            target = destination.copy().update(suffix=suffix, extension=ext).fpath
        copied[target] = path.read_bytes()
    events_path = sidecars["events", ".tsv"]
    if events_path is not None:
        _validate_events(events_path, raw)
    channels_path = sidecars["channels", ".tsv"]
    if channels_path is not None:
        _, channels = _read_table(channels_path)
        if [r.get("name") for r in channels] != raw.ch_names:
            raise ValueError(
                "Published channel order differs from decoded channel names."
            )
    coordinates = [
        sidecars[s, e] for s, e in (("electrodes", ".tsv"), ("coordsystem", ".json"))
    ]
    if any(p is not None for p in coordinates) and not all(
        p is not None for p in coordinates
    ):
        raise ValueError(
            "Preserving coordinates requires both electrodes.tsv and coordsystem.json."
        )

    for field, actual in (
        ("RecordingDuration", float(raw.n_times / raw.info["sfreq"])),
        ("EEGChannelCount", raw.get_channel_types().count("eeg")),
    ):
        if field in metadata and metadata[field] != actual:
            warnings.warn(
                f"{provenance['fields'][field]}: "
                f"published {field}={metadata[field]!r}; "
                f"decoded signal gives {actual!r}. Output uses the decoded value.",
                UserWarning,
                stacklevel=3,
            )

    description = json.loads(
        (source.root / "dataset_description.json").read_text(encoding="utf-8-sig")
    )
    source_doi = description.pop("DatasetDOI", None)
    sources = description.get("SourceDatasets", [])
    if not isinstance(sources, list):
        raise ValueError("SourceDatasets must be a list of source descriptions.")
    immediate = {"DOI": source_doi} if source_doi else {"URL": source_root.as_uri()}
    # Prepared datasets may already identify their original release. Keep that
    # chain when adding the recording subset's immediate source.
    if immediate not in sources:
        sources.append(immediate)
    description["SourceDatasets"] = sources
    description["Name"] += " (recording subset)"
    description.pop("BIDSVersion", None)
    description_path = destination.root / "dataset_description.json"
    if description_path.exists():
        previous = json.loads(description_path.read_text())
        previous.pop("BIDSVersion", None)
        if previous != description:
            raise ValueError("Destination dataset provenance differs from the source.")
    participants = _participants(source, destination)
    # Shared coordinate files and participant dictionaries may already exist on
    # later calls. Only identical content is safe to reuse without a review.
    for path, content in {**copied, **participants}.items():
        if (
            path.exists()
            and path.name != "participants.tsv"
            and path.read_bytes() != content
        ):
            raise FileExistsError(f"Conflicting destination metadata: {path}")
    for ext in (extension, ".json"):
        path = destination.copy().update(extension=ext).fpath
        if path.exists():
            raise FileExistsError(f"Destination recording already exists: {path}")

    prepared = raw.copy()
    if all(p is not None for p in coordinates):
        # Re-exporting imported MNI305 positions can create empty landmark arrays.
        # Retain the original coordinate system and units instead of synthesizing it.
        prepared.set_montage(None)
    # A private staging dataset also avoids MNE-BIDS rewriting custom participant
    # columns while appending another recording for an already copied subject.
    with tempfile.TemporaryDirectory(prefix="eeg-bids-") as temporary:
        staging = Path(temporary)
        written = write_raw_bids(prepared, destination.copy().update(root=staging))
        _preserve_stim_units(prepared, written)
        generated_path = written.copy().update(extension=".json").fpath
        generated = json.loads(generated_path.read_text())
        generated.update(
            {k: v for k, v in metadata.items() if k in _DESCRIPTIVE_FIELDS}
        )
        # RecordingDuration describes the sampled interval, including its final
        # sample, rather than just the timestamp of that sample.
        generated["RecordingDuration"] = float(raw.n_times / raw.info["sfreq"])
        generated_path.write_bytes(_json_bytes(generated))

        for path, content in {**copied, **participants}.items():
            staged_path = staging / path.relative_to(destination.root)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(content)
        staged_description = staging / "dataset_description.json"
        description["BIDSVersion"] = json.loads(staged_description.read_text())[
            "BIDSVersion"
        ]
        staged_description.write_bytes(_json_bytes(description))
        _publish(staging, destination.root, participants)
        return written.copy().update(root=destination.root)
