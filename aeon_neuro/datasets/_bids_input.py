"""Inspect local BIDS EEG sources without constructing signal arrays."""

import csv
import hashlib
import json
import re
import warnings
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np
from mne_bids import get_bids_path_from_fname

from aeon_neuro.datasets._bids_selection import _find_bids_paths
from aeon_neuro.datasets._classification import _samples
from aeon_neuro.utils._bids_json import _read_eeg_json
from aeon_neuro.utils.data_io import load_eeg_to_mne


def _path(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Source must remain inside the BIDS root: {name}")
    return path


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path, *, metadata=False):
    stat = path.stat()
    # Decimal text preserves nanosecond precision through JavaScript JSON clients.
    result = {"size": stat.st_size, "mtime_ns": str(stat.st_mtime_ns)}
    if metadata:
        result["sha256"] = _hash(path)
    return result


def _table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate table columns: {path.name}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Inconsistent table row width: {path.name}")
    return fields, rows


def _text(path):
    return path.read_bytes().decode("utf-8-sig")


def _json_object(path):
    value = json.loads(_text(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON metadata must contain an object: {path.name}")
    json.dumps(value, allow_nan=False)
    return value


def _metadata(root, path, files, reader):
    name = path.relative_to(root).as_posix()
    path = _path(root, name)
    before = _snapshot(path, metadata=True)
    value = reader(path)
    # Bind the returned values to the same bytes as the saved snapshot. These
    # checks detect changes during a read, without claiming a filesystem lock.
    if _snapshot(path, metadata=True) != before:
        raise ValueError(f"Metadata changed during inspection: {name}")
    files[name] = before
    return value


def _dataset_info(root, paths, issues):
    files = {}
    result = dict(
        description=None,
        readme=None,
        participants=None,
        participants_json=None,
        selected_participants=sorted({f"sub-{path.subject}" for path in paths}),
        paths={},
        files=files,
    )
    readmes = [
        root / name
        for name in ("README", "README.md", "README.rst", "README.txt")
        if (root / name).exists()
    ]
    if len(readmes) > 1:
        raise ValueError("Multiple dataset README files; keep one unambiguous source")
    sources = {
        "description": root / "dataset_description.json",
        "readme": readmes[0] if readmes else None,
        "participants": root / "participants.tsv",
        "participants_json": root / "participants.json",
    }
    for key, path in sources.items():
        result["paths"][key] = None
        if path is None or not path.exists():
            _issue(
                issues, None, f"{key}_missing", f"Dataset {key} metadata is unavailable"
            )
            continue
        reader = _json_object
        if key == "participants":
            reader = _table
        elif key == "readme":
            reader = _text
        value = _metadata(root, path, files, reader)
        if key == "participants":
            columns, rows = value
            if "participant_id" not in columns:
                raise ValueError("participants.tsv needs participant_id")
            identities = [row["participant_id"] for row in rows]
            if len(identities) != len(set(identities)):
                raise ValueError("Duplicate participant rows in participants.tsv")
            value = dict(columns=columns, rows=rows)
        result[key] = value
        result["paths"][key] = path.relative_to(root).as_posix()
    return result


def _issue(issues, recording_id, code, message, severity="warning"):
    issues.append(
        dict(
            recording_id=recording_id,
            code=code,
            severity=severity,
            message=str(message),
        )
    )


def _header(raw):
    return {
        "channel_names": list(raw.ch_names),
        "sampling_frequency": float(raw.info["sfreq"]),
        "n_samples": int(raw.n_times),
        "first_samp": int(raw.first_samp),
        "channel_types": raw.get_channel_types(),
        "channel_units_fiff": [int(channel["unit"]) for channel in raw.info["chs"]],
    }


def _sidecar(root, bids_path, suffix, extension=".tsv"):
    # MNE-BIDS uses the same error mode for absent and ambiguous sidecars.
    # Establish absence first, then retain its strict best-match resolution.
    directory = bids_path.fpath.parent
    while True:
        for path in directory.glob(f"*{suffix}{extension}"):
            entities = (
                {}
                if path.name == f"{suffix}{extension}"
                else get_bids_path_from_fname(path.name, check=False).entities
            )
            if all(
                value is None
                or bids_path.entities.get(key) == value
                or (
                    key == "space"
                    and bids_path.space is None
                    and suffix in ("electrodes", "coordsystem")
                )
                for key, value in entities.items()
            ):
                return bids_path.find_matching_sidecar(
                    suffix=suffix, extension=extension, on_error="raise"
                )
        if directory == root:
            return None
        directory = directory.parent


def _recording_sidecars(root, bids_path, descriptor, files, issues):
    recording_id = descriptor["recording_id"]
    descriptor.update(electrodes=None, coordsystem=None, scan=None)
    paths = descriptor["sidecar_paths"]
    electrode_path = _sidecar(root, bids_path, "electrodes")
    geometry_path = bids_path
    if electrode_path is not None:
        space = get_bids_path_from_fname(electrode_path.name, check=False).space
        # A geometry file can name a space absent from the raw recording name.
        # Resolve its coordinate system in that same space, never another one.
        geometry_path = bids_path.copy().update(space=space, check=False)
    coordinate_path = _sidecar(root, geometry_path, "coordsystem", ".json")
    # scans.tsv belongs to this subject/session, rather than inheriting a
    # different session's acquisition rows through a generic sidecar search.
    scan_path = (
        bids_path.copy()
        .update(
            task=None,
            acquisition=None,
            run=None,
            processing=None,
            recording=None,
            space=None,
            split=None,
            description=None,
            datatype=None,
            suffix="scans",
            extension=".tsv",
            check=False,
        )
        .fpath
    )
    for suffix, path in (
        ("electrodes", electrode_path),
        ("coordsystem", coordinate_path),
        ("scans", scan_path if scan_path.exists() else None),
    ):
        paths[suffix] = None
        if path is None:
            _issue(
                issues,
                recording_id,
                f"{suffix}_missing",
                f"No applicable {suffix} metadata",
            )
            continue
        paths[suffix] = path.relative_to(root).as_posix()
        if suffix == "coordsystem":
            descriptor["coordsystem"] = _metadata(root, path, files, _json_object)
            continue
        columns, rows = _metadata(root, path, files, _table)
        if suffix == "electrodes":
            if not {"name", "x", "y", "z"} <= set(columns):
                raise ValueError("Electrode table needs name, x, y and z columns")
            names = [row["name"] for row in rows]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate electrode names")
            descriptor["electrodes"] = dict(columns=columns, rows=rows)
        else:
            if "filename" not in columns:
                raise ValueError("Scans table needs a filename column")
            name = PurePosixPath(bids_path.fpath.relative_to(path.parent).as_posix())
            matches = [
                (i, row)
                for i, row in enumerate(rows)
                if PurePosixPath(row["filename"]) == name
            ]
            if len(matches) > 1:
                raise ValueError("Duplicate scans rows for this recording")
            if matches:
                index, row = matches[0]
                descriptor["scan"] = dict(row_index=index, row=row)
            else:
                _issue(
                    issues,
                    recording_id,
                    "scan_row_missing",
                    "No scans row for this recording",
                )
    if electrode_path is not None and (descriptor["coordsystem"] or {}).get(
        "EEGCoordinateUnits"
    ) in (None, "", "n/a"):
        _issue(
            issues,
            recording_id,
            "coordinate_units_missing",
            "Electrode coordinate units are unknown; source values retained",
        )


def _events(rows, recording_id, sfreq, n_samples):
    events = []
    for index, original in enumerate(rows):
        row = dict(original)
        reserved = {
            "recording_id",
            "bids_event_id",
            "bids_event_row",
            "bids_onset_sample",
        } & row.keys()
        if reserved:
            raise ValueError(
                f"Event columns shadow adapter identities: {sorted(reserved)}"
            )
        try:
            onset = float(row["onset"])
            sample = _samples(onset, sfreq, "event onset")
            duration = row["duration"]
            length = None if duration == "n/a" else float(duration)
            valid = 0 <= sample < n_samples and (
                length is None or (np.isfinite(length) and length >= 0)
            )
            for column in ("sample", "onset_sample"):
                if row.get(column, "n/a") != "n/a":
                    position = float(row[column])
                    valid = valid and position.is_integer() and position == sample
            if not valid:
                raise ValueError("Inconsistent event timing")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Event row {index} has invalid onset/duration/sample timing"
            ) from exc
        # Row identities deliberately describe this published table, without
        # pretending to recover original acquisition-trial identities.
        row.update(
            recording_id=recording_id,
            bids_event_id=f"row-{index}",
            bids_event_row=index,
            bids_onset_sample=sample,
        )
        events.append(row)
    return events


def _inspect(root, bids_path, issues, dataset):
    recording_id = bids_path.fpath.relative_to(root).as_posix()
    if not bids_path.subject:
        raise ValueError("A recording needs an explicit BIDS subject identity")
    if bids_path.suffix != "eeg" or bids_path.extension not in (".edf", ".bdf"):
        raise ValueError(
            "This adapter supports EDF/BDF; companion-file formats "
            "need a separate adapter"
        )
    signal = _path(root, recording_id)
    initial = _snapshot(signal)
    metadata, origins = _read_eeg_json(bids_path)
    files = {**deepcopy(dataset["files"]), recording_id: initial}

    def remember(path):
        name = path.relative_to(root).as_posix()
        files[name] = _snapshot(_path(root, name), metadata=True)

    for name in origins["sources"]:
        remember(root / name)
    descriptor = {
        "recording_id": recording_id,
        "filepath": recording_id,
        "subject": bids_path.subject,
        "participant_id": f"sub-{bids_path.subject}",
        "session": bids_path.session,
        "task": bids_path.task,
        "acq": bids_path.acquisition,
        "run": bids_path.run,
        "eeg_metadata": metadata,
        "metadata_origins": origins,
        "sidecar_paths": {},
    }
    if dataset["description"] is not None:
        descriptor["dataset_metadata"] = deepcopy(dataset["description"])
    if dataset["participants"] is not None:
        rows = dataset["participants"]["rows"]
        matches = [
            row for row in rows if row["participant_id"] == descriptor["participant_id"]
        ]
        if len(matches) > 1:
            raise ValueError("Duplicate participant rows for this recording")
        if matches:
            # Flat prefixed keys are ordinary binding columns, not a new path
            # language. They cannot shadow recording IDs or source-header facts.
            descriptor.update(
                {f"participant.{key}": value for key, value in matches[0].items()}
            )
        else:
            _issue(
                issues,
                recording_id,
                "participant_missing",
                "No matching participant row",
            )
    channels = _sidecar(root, bids_path, "channels")
    channel_rows = []
    if channels is not None:
        fields, channel_rows = _metadata(root, channels, files, _table)
        if "name" not in fields:
            raise ValueError("Channel table needs a name column")
        if not channel_rows:
            raise ValueError("Channel table is empty")
        if "type" not in fields:
            _issue(
                issues,
                recording_id,
                "channel_types_missing",
                "Channel table has no type column; no roles are inferred",
            )
    else:
        _issue(
            issues,
            recording_id,
            "channels_missing",
            "No channel table; use decoded header names and review channel roles",
        )
    event_path = _sidecar(root, bids_path, "events")
    event_rows = []
    if event_path is not None:
        _, event_rows = _metadata(root, event_path, files, _table)
    else:
        _issue(
            issues,
            recording_id,
            "events_missing",
            "No event table; event-epoch problems remain unresolved",
        )
    descriptor["sidecar_paths"].update(
        channels=(
            channels.relative_to(root).as_posix() if channels is not None else None
        ),
        events=(
            event_path.relative_to(root).as_posix() if event_path is not None else None
        ),
    )
    _recording_sidecars(root, bids_path, descriptor, files, issues)
    with load_eeg_to_mne(signal, preload=False, verbose=False) as raw:
        if raw.preload:
            raise ValueError("Source reader unexpectedly preloaded the recording")
        descriptor.update(_header(raw))
        rate = metadata.get("SamplingFrequency")
        if rate is None:
            _issue(
                issues,
                recording_id,
                "sampling_frequency_missing",
                "EEG JSON has no SamplingFrequency; decoded header value retained",
            )
        if rate is not None and (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not np.isfinite(rate)
            or rate != descriptor["sampling_frequency"]
        ):
            raise ValueError("EEG JSON SamplingFrequency disagrees with decoded header")
        if channel_rows and [row["name"] for row in channel_rows] != raw.ch_names:
            raise ValueError("Channel table order/names disagree with decoded header")
        descriptor["channel_metadata"] = channel_rows
        events = _events(event_rows, recording_id, raw.info["sfreq"], raw.n_times)
        if any(
            row["duration"] != "n/a"
            and float(row["onset"]) + float(row["duration"])
            > raw.n_times / raw.info["sfreq"]
            for row in events
        ):
            _issue(
                issues,
                recording_id,
                "event_duration_outside",
                "Published event duration extends beyond the recording; "
                "rows retained for explicit interval review",
            )
        duration = metadata.get("RecordingDuration")
        if (
            isinstance(duration, (int, float))
            and duration != raw.n_times / raw.info["sfreq"]
        ):
            _issue(
                issues,
                recording_id,
                "duration_mismatch",
                "Published duration differs from decoded sample count/rate; "
                "original value retained",
            )
    if _snapshot(signal) != initial:
        raise ValueError("Source changed during header inspection")
    descriptor["bids_source"] = {"version": 1, "files": files}
    _issue(
        issues,
        recording_id,
        "quality_unreviewed",
        "Event inclusion and channel quality require an explicit problem/source policy",
    )
    _issue(
        issues,
        recording_id,
        "content_unverified",
        "Source stats/headers inspected; signal content has not been hashed",
    )
    if any(row.get("trial_type") == "discontinuity" for row in events):
        _issue(
            issues,
            recording_id,
            "discontinuity",
            "Published discontinuity marker retained; "
            "review intervals before classification",
        )
    return descriptor, events


def inspect_bids_eeg(
    root, *, subjects=None, sessions=None, tasks=None, acqs=None, runs=None
):
    """Inspect local BIDS EEG headers and tables for explicit problem planning.

    Parameters
    ----------
    root : path-like
        Existing local BIDS root. No downloads or complete signal hashes occur.
    subjects, sessions, tasks, acqs, runs : str, list of str or None, default=None
        MNE-BIDS entity selections, without entity prefixes. None selects all.
        EDF/BDF are supported; discovered BrainVision/EEGLAB sources are reported
        as unsupported rather than silently omitted.

    Returns
    -------
    dict
        Version-2 inspection report with ready, source_ids, dataset, recordings,
        events and issues. Dataset facts retain optional description/README,
        the complete participant table/dictionary and selected participant IDs.
        Issues have severity, code, recording_id and message.
        Any blocking error sets recordings/events to None, preventing a silently
        partial catalogue. Warnings preserve unknown facts and caller decisions.
        Successful descriptors retain header facts, EEG JSON field origins,
        original tables and local file snapshots. Participant attributes use
        literal flat keys such as participant.group. Events preserve columns
        and add bids_event_id/bids_event_row, identifying published table rows,
        plus exact zero-based integer bids_onset_sample coordinates.
        Recording descriptors retain optional electrodes/coordsystem sidecars,
        a matching scans row and sidecar paths. TSV values stay strings; geometry
        values and row order do not alter the decoded channel layout or units.

    Notes
    -----
    Inspection opens supported native headers, never get_data. The native reader
    supplies names, sample rate/count, types and FIFF unit codes; sidecars do not
    rescale/retype samples. Sampling-rate/channel-order conflicts block planning.
    Missing optional tables are warnings, not invented labels or identities.
    No targets, quality eligibility, processing or evaluation are inferred.
    This is source inspection, not full BIDS validation or content certification.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS root is not a directory: {root}")
    try:
        paths = _find_bids_paths(
            root,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            acqs=acqs,
            runs=runs,
            extensions=[".edf", ".bdf", ".vhdr", ".set"],
        )
    except ValueError as exc:
        issues = []
        _issue(issues, None, "selection_failed", exc, "error")
        return dict(
            version=2,
            ready=False,
            source_ids=[],
            issues=issues,
            recordings=None,
            events=None,
            dataset=None,
        )
    paths = sorted(paths, key=lambda path: str(path.fpath))
    issues, recordings, events = [], [], []
    source_ids = [path.fpath.relative_to(root).as_posix() for path in paths]
    try:
        dataset = _dataset_info(root, paths, issues)
    except (OSError, ValueError, RuntimeError) as exc:
        _issue(issues, None, "dataset_metadata_failed", exc, "error")
        return dict(
            version=2,
            ready=False,
            source_ids=source_ids,
            dataset=None,
            recordings=None,
            events=None,
            issues=issues,
        )
    for path in paths:
        recording_id = path.fpath.relative_to(root).as_posix()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                descriptor, rows = _inspect(root, path, issues, dataset)
                recordings.append(descriptor)
                events.extend(rows)
            except (OSError, ValueError, RuntimeError) as exc:
                _issue(issues, recording_id, "inspection_failed", exc, "error")
        for warning in caught:
            _issue(issues, recording_id, "reader_warning", warning.message)
    ready = not any(issue["severity"] == "error" for issue in issues)
    return dict(
        version=2,
        ready=ready,
        source_ids=source_ids,
        issues=issues,
        recordings=recordings if ready else None,
        events=events if ready else None,
        dataset=dataset,
    )


def open_bids_eeg_recording(root, descriptor):
    """Verify an inspected local source snapshot and return an owned lazy Raw.

    Parameters
    ----------
    root : path-like
        Root containing the same inspected local files.
    descriptor : dict
        A recording from inspect_bids_eeg. Optional expected_sha256 supplies a
        trusted primary-signal checksum, verified here by reading all file bytes.
        Without it, unchanged size/mtime and decoded headers do not certify
        unchanged content. Metadata snapshots always use complete SHA256 hashes.

    Returns
    -------
    mne.io.BaseRaw
        Newly owned, non-preloaded native recording. The caller must close it;
        Context-manager use closes the handle on success and failure.

    Notes
    -----
    Saved metadata files are pinned; new files do not silently redefine the
    saved catalogue. Reinspect to adopt changed metadata or source selections.
    This local adapter neither downloads files nor executes the problem schema.
    """
    root = Path(root).expanduser().resolve()
    snapshot = descriptor.get("bids_source", {})
    if type(snapshot.get("version")) is not int or snapshot["version"] != 1:
        raise ValueError("Unsupported BIDS source snapshot version")
    recording_id = descriptor["recording_id"]
    if recording_id not in snapshot["files"]:
        raise ValueError("Source snapshot omits its signal file")
    for name, expected in snapshot["files"].items():
        actual = _snapshot(_path(root, name), metadata="sha256" in expected)
        if actual != expected:
            raise ValueError(f"Inspected source changed: {name}")
    signal = _path(root, recording_id)
    digest = descriptor.get("expected_sha256")
    if digest is not None:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA256 string")
        if _hash(signal) != digest:
            raise ValueError("Signal checksum differs from expected_sha256")
    raw = load_eeg_to_mne(signal, preload=False, verbose=False)
    try:
        header = _header(raw)
        if raw.preload or header != {key: descriptor[key] for key in header}:
            raise ValueError("Decoded source header differs from inspection")
    except BaseException:
        raw.close()
        raise
    return raw
