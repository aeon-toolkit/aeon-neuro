"""Bind explicit problem definitions to already loaded recordings."""

import hashlib
import json
from copy import deepcopy
from os import PathLike
from pathlib import Path

import numpy as np

from aeon_neuro.datasets._classification import (
    _CASE_RESERVED_FIELDS,
    _channels,
    _event_cases,
    _fixed_length_cases,
    _identifier,
    _recordings,
    build_classification_dataset,
)
from aeon_neuro.datasets._dataset_info import EEGDatasetInfo


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate problem/binding JSON key: {key!r}")
        result[key] = value
    return result


def _document(value, name):
    if isinstance(value, (str, PathLike)):
        with Path(value).open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_unique_keys)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary or JSON filename")
    try:
        # Normalise both input routes to ordinary JSON values and detach them
        # from caller-owned dictionaries before recording effective defaults.
        return json.loads(
            json.dumps(value, allow_nan=False), object_pairs_hook=_unique_keys
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc


def _keys(value, required, optional, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    missing = set(required) - value.keys()
    unsupported = value.keys() - set(required) - set(optional)
    # A misspelling can cause both errors; report them together in stable order.
    details = []
    if missing:
        details.append(f"missing {sorted(missing)}")
    if unsupported:
        details.append(f"unsupported {sorted(unsupported)}")
    if details:
        raise ValueError(f"Invalid fields in {name}: {'; '.join(details)}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _selector(value, sources, name):
    _keys(value, {"source", "column"}, {}, name)
    if not isinstance(value["source"], str) or value["source"] not in sources:
        raise ValueError(f"{name}.source must be one of {sorted(sources)}")
    _text(value["column"], f"{name}.column")


def _resolve(problem, binding):
    problem, binding = _document(problem, "problem"), _document(binding, "binding")
    _keys(
        problem,
        {"format_version", "id", "version", "question", "class_labels", "segmentation"},
        {"channels"},
        "problem",
    )
    _keys(
        binding,
        {"format_version", "problem", "target", "group"},
        {"case_fields", "events"},
        "binding",
    )
    for name, value in (("problem", problem), ("binding", binding)):
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise ValueError(f"Unsupported {name} format_version")
    for name in ("id", "version", "question"):
        _text(problem[name], f"problem.{name}")
    if binding["problem"] != {key: problem[key] for key in ("id", "version")}:
        raise ValueError("Binding must reference the exact problem id and version")
    classes = problem["class_labels"]
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or any(not isinstance(label, str) or not label.strip() for label in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("class_labels must contain at least two unique class names")
    if "channels" in problem:
        problem["channels"] = _channels(problem["channels"])
    segment = problem["segmentation"]
    if not isinstance(segment, dict):
        raise ValueError("problem.segmentation must be a dictionary")
    if segment.get("mode") == "fixed_windows":
        _keys(
            segment,
            {"mode", "window_size"},
            {"step_size", "start", "stop"},
            "problem.segmentation",
        )
        segment.setdefault("step_size", segment["window_size"])
        segment.setdefault("start", 0.0)
        segment.setdefault("stop", None)
        sources = {"recording"}
        if "events" in binding:
            raise ValueError("Event bindings require event_epochs segmentation")
    elif segment.get("mode") == "event_epochs":
        _keys(
            segment,
            {"mode", "tmin", "tmax"},
            {"on_out_of_bounds"},
            "problem.segmentation",
        )
        segment.setdefault("on_out_of_bounds", "raise")
        if segment["on_out_of_bounds"] not in ("raise", "exclude"):
            raise ValueError("on_out_of_bounds must be 'raise' or 'exclude'")
        _keys(binding.get("events"), {"onset", "id"}, {}, "binding.events")
        for name in ("onset", "id"):
            _text(binding["events"][name], f"binding.events.{name}")
        sources = {"recording", "event"}
    else:
        raise ValueError("segmentation.mode must be fixed_windows or event_epochs")
    target = binding["target"]
    _keys(target, {"source", "column", "mapping"}, {"on_unmapped"}, "binding.target")
    _selector(
        {key: target[key] for key in ("source", "column")}, sources, "binding.target"
    )
    mapping = target["mapping"]
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(key, str) or not key.strip() for key in mapping)
        or any(not isinstance(label, str) for label in mapping.values())
        or set(mapping.values()) != set(classes)
    ):
        raise ValueError("Target mapping must cover exactly the declared class_labels")
    target.setdefault("on_unmapped", "raise")
    if target["on_unmapped"] not in ("raise", "exclude"):
        raise ValueError("on_unmapped must be 'raise' or 'exclude'")
    _selector(binding["group"], sources, "binding.group")
    fields = binding.setdefault("case_fields", {})
    if not isinstance(fields, dict):
        raise ValueError("case_fields must map output field names to selectors")
    for name, selector in fields.items():
        _text(name, "case field name")
        if name in _CASE_RESERVED_FIELDS:
            raise ValueError(f"Reserved case field: {name}")
        _selector(selector, sources, f"binding.case_fields.{name}")
    return problem, binding


def _value(recording, event, selector):
    row = recording if selector["source"] == "recording" else event
    column = selector["column"]
    if column not in row:
        raise ValueError(f"Missing {selector['source']} column: {column}")
    return row[column]


def _identity(value, name):
    value = _identifier(value, name)
    if value == "n/a":
        raise ValueError(f"Missing identity for {name}")
    return value


def _decorate(row, recording, event, problem, binding):
    keep, reason = (
        (True, "")
        if event is None
        else (event.get("included", True), event.get("exclusion_reason", ""))
    )
    if not isinstance(keep, (bool, np.bool_)) or not isinstance(reason, str):
        raise ValueError("Event included/exclusion_reason must be boolean/string")
    if (not keep and not reason.strip()) or (keep and reason):
        raise ValueError(
            "Excluded events need a reason; included events cannot have one"
        )
    label = None
    if keep:
        target = binding["target"]
        code = str(_identifier(_value(recording, event, target), "target value"))
        label = target["mapping"].get(code)
        if label is None:
            if target["on_unmapped"] == "raise":
                raise ValueError(f"Unmapped target value: {code!r}")
            keep, reason = False, "unmapped_target"
    row.update(
        label=label,
        group_id=_identity(_value(recording, event, binding["group"]), "group"),
        included=bool(keep),
        exclusion_reason=reason,
    )
    for name, selector in binding["case_fields"].items():
        row[name] = _identity(_value(recording, event, selector), name)
    if "channels" in problem:
        row["channel_names"] = list(problem["channels"])
    return row


def load_classification_problem(X, metadata, problem, binding, *, events=None):
    """Construct classification cases from loaded recordings and a definition.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Loaded (recording, channel, time) arrays or a list of 2D recordings,
        including outputs from load_bids_data. No signal files are read.
    metadata : list of dict
        Aligned recording metadata with recording_id (or filepath), channel_names
        and sampling_frequency. Preserve known units and earlier processing here.
    problem : dict or path-like
        Version-1 definition with id, version, question, class_labels and
        segmentation. Supports fixed_windows or event_epochs and optional ordered
        channels. See the loaded-problem guide for the complete vocabulary.
    binding : dict or path-like
        Version-1 dataset binding referencing the exact problem id/version.
        Explicit target mapping, group selector and optional case_fields resolve
        recording/event columns. Event epochs also require onset/id columns.
        No filename matching or inferred class meanings are used.
    events : iterable of dict or None, default=None
        Event rows in the desired output order. Each needs recording_id and the
        bound onset/id columns. Onsets are seconds from the first supplied sample,
        not acquisition-absolute MNE samples. Numeric TSV onset strings are
        accepted. Optional boolean included=False requires exclusion_reason;
        original rows are retained, including excluded rows. Required for epochs
        and rejected for fixed windows. Use table.to_dict("records") for pandas.

    Returns
    -------
    ClassificationDataset
        Cases, labels, groups and exclusions from the existing case builder.
        metadata["problem"] retains independent effective documents and their
        SHA256 identities. These bind configuration, not input signal contents.

    Notes
    -----
    The version-1 problem/binding format is experimental.
    Bounds are zero-based, stop-exclusive on the supplied recording grid, with
    no silent rounding, clipping or padding of epochs. Fixed-window stop/tail
    behaviour follows make_fixed_length_cases. Source target codes are string
    or integer values compared by their string spelling to JSON mapping keys.
    Class order is declared metadata, not an estimator probability-column order.
    No preprocessing, resampling, splitting, source-file hashing or fitting is
    performed. Validate actual evaluation assignments separately.
    """
    problem, binding = _resolve(problem, binding)
    _, info = _recordings(X, metadata)
    cases, boundary = _problem_cases(info, problem, binding, events)
    dataset = build_classification_dataset(X, info, cases, on_out_of_bounds=boundary)
    dataset.metadata["problem"] = _configuration(problem, binding)
    return dataset


def make_classification_problem(X, info, problem, binding):
    """Build a classification problem from loaded signals and shared source info.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Loaded recordings aligned with ``info``, in (channel, sample) order
        per recording. Complete source layouts must match; no signal files are
        opened or preprocessing performed.
    info : EEGDatasetInfo
        Shared source metadata, events and reported history, usually returned
        by ``load_bids_dataset`` or paired ``info.select``. Event epochs use
        its events automatically; fixed windows use recording facts only.
    problem : dict or path-like
        Experimental version-1 definition accepted by
        ``load_classification_problem``. A JSON filename is read explicitly.
    binding : dict or path-like
        Experimental version-1 binding with explicit target mappings and group
        choices. Recording selectors name exact descriptor keys, including BIDS
        entities and matched participant columns such as ``participant.group``.
        Event selectors name original event columns. No class meaning is inferred.

    Returns
    -------
    ClassificationDataset
        The existing case-builder result with detached provenance under
        ``metadata["provenance"]``: version, source report, reported history,
        effective problem/binding with hashes, and construction implementation.
        This route stores the problem configuration once inside provenance.
        The older loader retains its separate ``metadata["problem"]`` contract.

    Notes
    -----
    This experimental convenience reuses the existing binding and segmentation
    rules. It does not reread BIDS sidecars or implement another event parser.
    Original source facts and history scopes are preserved, including earlier
    processing scopes broader than the selected recordings. Source/configuration
    snapshots and caller-reported history do not certify unchanged signal bytes
    or leakage-free processing. Layout checks cannot detect equal-shaped external
    array swaps or value edits. A history note does not update derived layouts.
    Validate actual evaluation assignments separately.
    """
    if not isinstance(info, EEGDatasetInfo):
        raise ValueError("info must be an EEGDatasetInfo instance")
    info.validate_layout(X)
    problem, binding = _resolve(problem, binding)
    from aeon_neuro.datasets._info_view import _recording_inputs

    metadata, events = _recording_inputs(
        info, binding.get("events", {}).get("onset", "onset")
    )
    # Keep one binding/segmentation authority, including exclusions and effective
    # configuration hashes. The object supplies facts, not a second case engine.
    dataset = load_classification_problem(
        X,
        metadata,
        problem,
        binding,
        events=(events if problem["segmentation"]["mode"] == "event_epochs" else None),
    )
    dataset.metadata["provenance"] = _provenance(
        info,
        dataset.metadata.pop("problem"),
        "aeon_neuro.datasets.make_classification_problem",
    )
    return dataset


def _problem_cases(info, problem, binding, events):
    lookup = {item["recording_id"]: item for item in info}
    segment = problem["segmentation"]
    if segment["mode"] == "fixed_windows":
        if events is not None:
            raise ValueError("events are only used by event_epochs segmentation")
        # Map targets after enumeration so an explicitly unmapped recording can
        # retain all of its excluded windows in the ordinary case table.
        rows = _fixed_length_cases(
            info,
            labels=[
                str(_identifier(_value(item, None, binding["target"]), "target value"))
                for item in info
            ],
            groups=[
                _identity(_value(item, None, binding["group"]), "group")
                for item in info
            ],
            **{key: value for key, value in segment.items() if key != "mode"},
        )
        cases = [
            _decorate(row, lookup[row["recording_id"]], None, problem, binding)
            for row in rows
        ]
        boundary = "raise"
    else:
        if events is None or isinstance(events, (str, dict)):
            raise ValueError("Event epochs require an iterable of event dictionaries")
        prepared = []
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError("Each event must be a dictionary")
            event = deepcopy(event)
            recording_id = event.get("recording_id")
            if not isinstance(recording_id, str) or recording_id not in lookup:
                raise ValueError(f"Unknown event recording_id: {recording_id!r}")
            item = lookup[recording_id]
            event_id = _identity(
                _value(
                    item, event, {"source": "event", "column": binding["events"]["id"]}
                ),
                "event id",
            )
            onset = _value(
                item, event, {"source": "event", "column": binding["events"]["onset"]}
            )
            if isinstance(onset, (bool, np.bool_)):
                raise ValueError("Event onset must be finite seconds")
            try:
                onset = float(onset)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Event onset must be finite seconds") from exc
            row = {
                # A structured identity avoids delimiter and string/integer
                # collisions across recording-local event identifiers.
                "case_id": "event:" + json.dumps([recording_id, event_id]),
                "recording_id": recording_id,
                "event_id": event_id,
                "onset": onset,
                "source_event_row": index,
                "source_event": event,
            }
            prepared.append(_decorate(row, item, event, problem, binding))
        cases = _event_cases(info, prepared, tmin=segment["tmin"], tmax=segment["tmax"])
        boundary = segment["on_out_of_bounds"]
    return cases, boundary


def _configuration(problem, binding):
    return {
        "definition": problem,
        "binding": binding,
        **{
            f"{name}_sha256": hashlib.sha256(
                json.dumps(value, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for name, value in (("definition", problem), ("binding", binding))
        },
    }


def _provenance(info, configuration, implementation):
    # The shared case layer owns source/history; authored definitions add only
    # their resolved configuration around that same provenance contract.
    from aeon_neuro.datasets._dataset_info import _source_provenance

    provenance = _source_provenance(info, implementation)
    provenance["problem"] = deepcopy(configuration)
    return provenance
