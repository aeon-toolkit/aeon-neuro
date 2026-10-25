"""Check downloaded EEG recordings and prepare metadata without copying signals."""

import csv
import io
import json
import math
import warnings
from pathlib import PurePosixPath

from mne_bids import find_matching_paths, read_raw_bids

from aeon_neuro.utils._bids_json import _read_eeg_json
from aeon_neuro.utils.data_io import _check_edf_layout

_METADATA_FIELDS = {
    "PowerLineFrequency",
    "EEGReference",
    "EEGGround",
    "EEGPlacementScheme",
    "Manufacturer",
    "ManufacturersModelName",
    "TaskDescription",
}
_CHANNEL_TYPES = {"EEG", "EOG", "ECG", "EMG", "MISC", "TRIG"}
_COUNT_FIELDS = {
    "eeg": "EEGChannelCount",
    "eog": "EOGChannelCount",
    "ecg": "ECGChannelCount",
    "emg": "EMGChannelCount",
    "misc": "MiscChannelCount",
    "stim": "TriggerChannelCount",
}


class BIDSPreparationError(ValueError):
    """EEG preparation failed, with per-recording findings in ``report``."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            f"EEG preparation failed for {report['summary']['failed']} recording(s). "
            "Inspect the exception's report; no completed cache was published."
        )


def _validate_corrections(corrections):
    if corrections is None:
        return {}
    if not isinstance(corrections, dict):
        raise ValueError("metadata_corrections must map recording paths to corrections")
    # Check before downloading: correction identity must refer to an exact
    # recording, never a basename that might match several subjects or sessions.
    for name, values in corrections.items():
        if (
            not isinstance(name, str)
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or not name.startswith("sub-")
            or not isinstance(values, dict)
            or set(values) - {"source", "metadata", "channel_types"}
            or not isinstance(values.get("source"), str)
            or not values["source"].strip()
        ):
            raise ValueError(
                "Each correction needs an exact relative recording path and source"
            )
        metadata, types = values.get("metadata", {}), values.get("channel_types", {})
        if not isinstance(metadata, dict) or set(metadata) - _METADATA_FIELDS:
            raise ValueError("Unsupported metadata correction field")
        for field, value in metadata.items():
            if field == "PowerLineFrequency":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "PowerLineFrequency correction must be finite and positive"
                    )
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} correction must be a nonempty string")
        if not isinstance(types, dict) or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or value not in _CHANNEL_TYPES
            for name, value in types.items()
        ):
            raise ValueError(
                "channel_types corrections must use supported BIDS type names"
            )
        if not metadata and not types:
            raise ValueError("A correction must specify metadata or channel_types")
    return corrections


def _table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(
            None in row or any(value is None for value in row.values()) for row in rows
        )
    ):
        raise ValueError(f"Malformed TSV: {path.name}")
    return fields, rows


def _write_table(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def _issue(record, code, message):
    record["issues"].append({"code": code, "message": message})


def _change(record, field, before, after, source):
    record["corrections"].append(
        {"field": field, "before": before, "after": after, "source": source}
    )


def _check_events(path, raw, record):
    if path is None:
        _issue(record, "missing_events", "No matched event table was supplied.")
        return
    _, rows = _table(path)
    outside, discontinuities = 0, 0
    span, sfreq = raw.n_times / raw.info["sfreq"], raw.info["sfreq"]
    for number, row in enumerate(rows, 1):
        try:
            onset = float(row["onset"])
            duration = 0.0 if row["duration"] == "n/a" else float(row["duration"])
            if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
                raise ValueError
            samples = set()
            for column in ("sample", "onset_sample"):
                sample = row.get(column, "n/a")
                if sample != "n/a":
                    sample = float(sample)
                    if (
                        not sample.is_integer()
                        or abs(sample - onset * sfreq) > 0.500001
                    ):
                        raise ValueError
                    samples.add(sample)
            # At exactly half a sample, two different integers can both meet
            # the onset tolerance. Supplied aliases must still identify one sample.
            if len(samples) > 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Event row {number} has invalid timing or sample position"
            ) from exc
        outside += onset < 0 or onset >= span or onset + duration > span + 1e-9
        discontinuities += row.get("trial_type") == "discontinuity"
    record["events"] = len(rows)
    if outside:
        # BIDS permits negative event onsets. Keep source timing and leave the
        # usable interval policy to case construction instead of clipping rows.
        _issue(
            record,
            "events_outside_signal",
            f"{outside} event(s) extend outside the signal interval.",
        )
    if discontinuities:
        _issue(
            record,
            "discontinuities",
            f"{discontinuities} discontinuity marker(s); gap lengths are unknown.",
        )


def _prepare_recording(path, root, corrections, record):
    if path.extension in (".edf", ".bdf"):
        # MNE-BIDS calls native readers directly; preserve the native loader's
        # rejection of inferred record counts, incomplete tails and time gaps.
        _check_edf_layout(path.fpath, 2 if path.extension == ".edf" else 3)
    relative = path.fpath.relative_to(root).as_posix()
    metadata, inheritance = _read_eeg_json(path)
    record["original_metadata"] = metadata.copy()
    record["metadata_sources"] = inheritance["sources"]
    record["metadata_origins"] = inheritance["fields"]
    record["actions"] = []
    values = corrections.get(relative, {})
    for field, value in values.get("metadata", {}).items():
        if metadata.get(field) != value:
            _change(record, field, metadata.get(field), value, values["source"])
            metadata[field] = value

    channels_path = path.find_matching_sidecar(
        suffix="channels", extension=".tsv", on_error="ignore"
    )
    fields, channels = _table(channels_path) if channels_path is not None else ([], [])
    types = values.get("channel_types", {})
    if types:
        if not channels:
            raise ValueError(
                "Channel corrections require a source channels.tsv with verified units"
            )
        names = {row["name"] for row in channels}
        if set(types) - names:
            raise ValueError("Channel correction names are absent from channels.tsv")
        for row in channels:
            after = types.get(row["name"], row["type"])
            if after != row["type"]:
                _change(
                    record,
                    f"channel_types.{row['name']}",
                    row["type"],
                    after,
                    values["source"],
                )
                row["type"] = after
        # Localize inherited tables so one recording's correction cannot change
        # another recording that shares its original sidecar.
        target = path.copy().update(suffix="channels", extension=".tsv").fpath
        _write_table(target, fields, channels)

    target_json = path.copy().update(extension=".json").fpath
    target_relative = target_json.relative_to(root).as_posix()
    materialize = any(
        name != target_relative for name in inheritance["fields"].values()
    )
    if materialize:
        record["actions"].append(
            {
                "path": target_relative,
                "reason": "Materialize inherited EEG JSON fields for BIDS reading",
            }
        )
    if materialize or values.get("metadata"):
        # MNE-BIDS selects one EEG JSON rather than combining ancestor fields.
        # Localize effective metadata in staging, preserving inherited sources.
        target_json.write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = read_raw_bids(path, verbose=False)
    try:
        if raw.preload:
            raise ValueError(
                "Preparation requires a recording that supports deferred reading"
            )
        record.update(
            channels=len(raw.ch_names),
            samples=int(raw.n_times),
            sfreq=float(raw.info["sfreq"]),
        )
        for warning in caught:
            _issue(
                record,
                "reader_warning",
                str(warning.message).replace(str(root), "<dataset>"),
            )
        if metadata.get("SamplingFrequency", raw.info["sfreq"]) != raw.info["sfreq"]:
            raise ValueError(
                "Published SamplingFrequency disagrees with the native signal"
            )
        if channels and [row["name"] for row in channels] != raw.ch_names:
            raise ValueError("Published channel order disagrees with the native signal")
        if not channels:
            _issue(
                record,
                "missing_channel_roles",
                "No channels.tsv; reader defaults are not verified channel roles. "
                "Published counts are retained.",
            )
        frequency = metadata.get("PowerLineFrequency", "n/a")
        if frequency == "n/a" or frequency is None:
            _issue(
                record,
                "unknown_mains_frequency",
                "No verified mains frequency was supplied.",
            )
        elif (
            isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not math.isfinite(frequency)
            or frequency <= 0
        ):
            raise ValueError(
                "Published PowerLineFrequency is invalid; supply a verified correction"
            )
        derived = {"RecordingDuration": float(raw.n_times / raw.info["sfreq"])}
        if channels:
            for kind, field in _COUNT_FIELDS.items():
                derived[field] = raw.get_channel_types().count(kind)
        for field, value in derived.items():
            if metadata.get(field) != value:
                _change(
                    record,
                    field,
                    metadata.get(field),
                    value,
                    "decoded signal with supplied channel roles",
                )
                metadata[field] = value
        _check_events(
            path.find_matching_sidecar(
                suffix="events", extension=".tsv", on_error="ignore"
            ),
            raw,
            record,
        )
        if record["corrections"]:
            target_json.write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        record["status"] = "prepared"
    finally:
        raw.close()


def _prepare_dataset(root, corrections):
    root = root.resolve()
    paths = find_matching_paths(
        root,
        datatypes="eeg",
        suffixes="eeg",
        extensions=[".vhdr", ".edf", ".bdf", ".set"],
        check=False,
    )
    # Preparation concerns primary recordings. Preserve derivative files exactly
    # as downloaded; they may describe a different signal or preprocessing step.
    paths = [
        path
        for path in paths
        if path.fpath.relative_to(root).parts[0].startswith("sub-")
    ]
    known = {path.fpath.relative_to(root).as_posix() for path in paths}
    if set(corrections) - known:
        raise ValueError(
            "Metadata corrections refer to recordings outside this selection"
        )
    report = {
        "schema": 1,
        "scope": (
            "EEG metadata, channel correspondence and event timing; "
            "not full BIDS validation"
        ),
        "recordings": [],
    }
    for path in paths:
        record = {
            "path": path.fpath.relative_to(root).as_posix(),
            "issues": [],
            "corrections": [],
        }
        report["recordings"].append(record)
        try:
            _prepare_recording(path, root, corrections, record)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            record.update(
                status="failed", error=str(exc).replace(str(root), "<dataset>")
            )
    report["summary"] = {
        "recordings": len(paths),
        "failed": sum(record["status"] == "failed" for record in report["recordings"]),
        "corrections": sum(
            len(record["corrections"]) for record in report["recordings"]
        ),
        "issues": sum(len(record["issues"]) for record in report["recordings"]),
    }
    if report["summary"]["failed"]:
        raise BIDSPreparationError(report)
    if not paths:
        report["summary"]["issues"] += 1
        report["issues"] = [
            "No supported primary EEG recordings were found; "
            "other modalities were left unchanged."
        ]
    report["dataset_changes"] = [
        action
        for record in report["recordings"]
        for action in record.get("actions", [])
    ]
    if (root / "datacite.yml").is_file():
        # DataLad/OpenNeuro releases can retain DataCite citation metadata at the
        # root. Keep its bytes, but identify this non-BIDS file to the validator.
        ignore = root / ".bidsignore"
        previous = ignore.read_text("utf-8") if ignore.exists() else ""
        if "datacite.yml" not in previous.splitlines():
            separator = "\n" if previous and not previous.endswith("\n") else ""
            ignore.write_text(previous + separator + "datacite.yml\n", encoding="utf-8")
            report["dataset_changes"].append(
                {
                    "path": ".bidsignore",
                    "reason": "Preserve non-BIDS DataCite citation metadata",
                }
            )
    if report["summary"]["corrections"] or report["dataset_changes"]:
        description_path = root / "dataset_description.json"
        description = json.loads(description_path.read_text(encoding="utf-8-sig"))
        report["original_dataset_description"] = description.copy()
        # A locally corrected copy must not claim the source release's DOI as
        # its own identifier. Keep source attribution and ancillary files.
        source_doi = description.pop("DatasetDOI", None)
        if source_doi:
            sources = list(description.get("SourceDatasets", []))
            if {"DOI": source_doi} not in sources:
                sources.append({"DOI": source_doi})
            description["SourceDatasets"] = sources
        description["Name"] += " (prepared copy)"
        description_path.write_text(
            json.dumps(description, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report
