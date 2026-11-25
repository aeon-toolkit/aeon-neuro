"""Plan from metadata, then explicitly open selected sources for extraction."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import aeon_neuro
from aeon_neuro.datasets import _classification, _dataset_info, _info_view, _problem
from aeon_neuro.datasets._classification import (
    ClassificationDataset,
    _identifiers,
    _prepare_cases,
    _recording_info,
)
from aeon_neuro.datasets._dataset_info import EEGDatasetInfo


def _plain(value):
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Plan metadata keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Plan metadata must contain ordinary JSON values")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def resolve_classification_problem(problem, binding):
    """Validate a definition and binding without recording metadata or signals.

    Parameters
    ----------
    problem, binding : dict or path-like
        Experimental version-1 documents accepted by load_classification_problem.
        Only the supplied JSON files, if any, are read.

    Returns
    -------
    dict
        Effective definition/binding and configuration hashes. This validates
        document structure, not source availability, label meaning or case counts.
    """
    return _problem._configuration(*_problem._resolve(problem, binding))


def plan_classification_problem(metadata, problem, binding, *, events=None):
    """Describe classification cases without opening or fetching signal files.

    Parameters
    ----------
    metadata : list of dict or EEGDatasetInfo
        Recording descriptors with recording_id (or filepath), channel_names,
        sampling_frequency and exact positive n_samples. Additional source/release
        references, expected checksums and sizes are retained as declarations.
        Supply facts from a small catalogue, sidecars or previously read headers;
        this function does not discover remote metadata or verify source files.
        A shared info object supplies its own events and preserves the complete
        source report and reported history in a version-2 plan.
    problem, binding : dict or path-like
        Version-1 documents used by load_classification_problem. Definition-only
        validation can run before these source descriptors are available.
    events : iterable of dict or None, default=None
        Bound event rows for event epochs. Only metadata is consumed. Source row
        order, explicit eligibility and original event identities are retained.
        Must be omitted for shared info objects; their own events are used for
        epochs and no events are passed for fixed windows.

    Returns
    -------
    dict
        JSON-compatible plan with cases, exclusions, source descriptors, effective
        configuration, plan identity and a float64 case-array storage estimate.
        Heterogeneous outputs can be planned; choose a compatible subset before
        materialization. Signal values and declared file hashes remain unchecked.

    Notes
    -----
    No signal opener, source hashing or download is invoked. Only small local
    implementation files are hashed to detect code changes before materializing.
    A metadata plan is not a content-verified source receipt or evaluation split.
    """
    definition, binding = _problem._resolve(problem, binding)
    shared = metadata if isinstance(metadata, EEGDatasetInfo) else None
    if shared is not None:
        if events is not None:
            raise ValueError("Separate events cannot override an EEGDatasetInfo object")
        from aeon_neuro.datasets._info_view import _recording_inputs

        metadata, source_events = _recording_inputs(
            shared, binding.get("events", {}).get("onset", "onset")
        )
        events = (
            source_events
            if definition["segmentation"]["mode"] == "event_epochs"
            else None
        )
    info = _recording_info(metadata)
    cases, boundary = _problem._problem_cases(info, definition, binding, events)
    included, excluded, _, _ = _prepare_cases(
        info, cases, on_out_of_bounds=boundary, require_compatible=False
    )
    lookup = {item["recording_id"]: item for item in info}
    layouts = {
        (
            tuple(row["output_channel_names"]),
            lookup[row["recording_id"]]["sampling_frequency"],
            row["stop_sample"] - row["start_sample"],
        )
        for row in included
    }
    shape = None
    if len(layouts) == 1:
        names, _, length = next(iter(layouts))
        shape = [len(included), len(names), length]
    plan = _plain(
        {
            "format_version": 2 if shared is not None else 1,
            "problem": _problem._configuration(definition, binding),
            "recordings": info,
            "cases": included,
            "exclusions": excluded,
            "dense_shape": shape,
            "estimated_case_bytes_float64": sum(
                len(row["channel_names"])
                * (row["stop_sample"] - row["start_sample"])
                * 8
                for row in included
            ),
            "source_verification": "metadata_only; declared checksums not verified",
            "implementation_sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in (
                    ("case_rules", _classification.__file__),
                    ("definition", _problem.__file__),
                    ("planner", __file__),
                )
            },
        }
    )
    if shared is not None:
        # Bind the original input scope, including earlier processing over a
        # broader population. Selected loading must not narrow those claims.
        plan["input_info"] = _plain(shared.to_dict())
        plan["implementation_sha256"].update(
            {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in (
                    ("metadata", _dataset_info.__file__),
                    ("current_layout", _info_view.__file__),
                    ("package", aeon_neuro.__file__),
                )
            }
        )
    plan["plan_id"] = _digest(plan)
    return plan


def _validated_plan(plan):
    if (
        not isinstance(plan, dict)
        or type(plan.get("format_version")) is not int
        or plan["format_version"] not in (1, 2)
    ):
        raise ValueError("Unsupported problem plan format_version")
    try:
        events = None
        config = plan["problem"]
        if plan["format_version"] == 2:
            metadata = EEGDatasetInfo(plan["input_info"])
        else:
            metadata = plan["recordings"]
        if (
            plan["format_version"] == 1
            and config["definition"]["segmentation"]["mode"] == "event_epochs"
        ):
            rows = sorted(
                [*plan["cases"], *plan["exclusions"]],
                key=lambda row: row["source_event_row"],
            )
            if [row["source_event_row"] for row in rows] != list(range(len(rows))):
                raise ValueError("Event source rows are incomplete or duplicated")
            events = [row["source_event"] for row in rows]
        rebuilt = plan_classification_problem(
            metadata, config["definition"], config["binding"], events=events
        )
        if rebuilt != plan:
            raise ValueError("Problem plan changed or is stale; inspect it again")
        return rebuilt
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed problem plan") from exc


def materialize_classification_problem(plan, open_recording, *, case_ids=None):
    """Open only selected recordings and extract planned intervals for training.

    Parameters
    ----------
    plan : dict
        Saved result of plan_classification_problem. Regenerated and compared
        before the first source is opened, including configuration and code.
    open_recording : callable
        Called once per selected recording with its independent descriptor dict.
        Return a newly owned public mne.io.BaseRaw handle. The adapter may fetch
        and cache that recording and its companion files at this point. It must
        pin the source release and verify any expected content checksums before
        returning. This function checks decoded headers, reads selected intervals
        and closes each handle, including on failure. It supplies no downloader.
    case_ids : sequence of str or None, default=None
        Nonempty unique eligible case IDs, or all eligible cases. Membership is
        selected in the plan's canonical order, not the order of this argument.

    Returns
    -------
    ClassificationDataset
        Selected cases, native-valued arrays and provenance with the plan ID.
        Equal lengths are stacked; ragged lengths remain a list. Arrays are
        materialized in memory; this is not streaming estimator fitting.
        Shared-info plans retain the eager route's source/history/problem
        provenance sections, with this function named as their constructor.
        Plain-input plans retain the existing separate problem metadata.

    Notes
    -----
    Reader-side preload and network behaviour belong to the source adapter.
    An opened format may require the complete selected file, even for a short
    epoch. No recording outside the selected cases is opened by this function.
    Header checks cannot establish full content identity; checksum verification
    belongs to the explicit adapter. Signal finiteness is checked when read.
    Shared-info provenance retains the full planned input scope, including
    sources not opened by a case selection. The materialization record identifies
    actual selected cases, and recording metadata lists only opened sources.
    """
    checked = _validated_plan(plan)
    if not callable(open_recording):
        raise TypeError("open_recording must be an explicit callable")
    if case_ids is None:
        selected = checked["cases"]
    else:
        if (
            not isinstance(case_ids, (list, tuple))
            or not case_ids
            or any(not isinstance(value, str) or not value for value in case_ids)
            or len(set(case_ids)) != len(case_ids)
        ):
            raise ValueError("case_ids must be a nonempty unique list of strings")
        requested = set(case_ids)
        if requested - {row["case_id"] for row in checked["cases"]}:
            raise ValueError("Unknown or excluded case IDs")
        selected = [row for row in checked["cases"] if row["case_id"] in requested]
    included, _, channels, sfreq = _prepare_cases(checked["recordings"], selected)
    by_source = {}
    for row in included:
        by_source.setdefault(row["recording_id"], []).append(row)
    descriptors = {
        item["recording_id"]: item
        for item in checked["recordings"]
        if item["recording_id"] in by_source
    }
    signals = [None] * len(included)
    from mne.io import BaseRaw

    for recording_id, rows in by_source.items():
        descriptor = descriptors[recording_id]
        raw = open_recording(deepcopy(descriptor))
        if not isinstance(raw, BaseRaw):
            raise TypeError("open_recording must return a newly owned MNE Raw handle")
        try:
            if (
                raw.n_times != descriptor["n_samples"]
                or list(raw.ch_names) != descriptor["channel_names"]
                or float(raw.info["sfreq"]) != descriptor["sampling_frequency"]
                or (
                    "first_samp" in descriptor
                    and raw.first_samp != descriptor["first_samp"]
                )
            ):
                raise ValueError(
                    f"Recording header differs from the plan: {recording_id}"
                )
            for row in rows:
                values = raw.get_data(
                    picks=row["channel_names"],
                    start=row["start_sample"],
                    stop=row["stop_sample"],
                )
                expected = (len(channels), row["stop_sample"] - row["start_sample"])
                # Enforce the same real-numeric array contract as eager inputs.
                if (
                    not isinstance(values, np.ndarray)
                    or values.shape != expected
                    or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        "Reader returned invalid or nonfinite real case samples"
                    )
                signals[row["row"]] = values.copy()
        finally:
            raw.close()
    if all(value.shape == signals[0].shape for value in signals):
        signals = np.stack(signals)
    dataset = ClassificationDataset(
        X=signals,
        y=_identifiers([row["label"] for row in included], "labels"),
        groups=_identifiers([row["group_id"] for row in included], "groups"),
        cases=included,
        exclusions=deepcopy(checked["exclusions"]),
        metadata={
            "recordings": list(descriptors.values()),
            "channel_names": channels,
            "sampling_frequency": sfreq,
            "sample_coordinates": "zero-based, stop-exclusive; supplied recordings",
            "problem": checked["problem"],
            "materialization": {
                "plan_id": checked["plan_id"],
                "selected_case_ids": [row["case_id"] for row in included],
                "unselected_cases": len(checked["cases"]) - len(included),
                "source_verification": (
                    "decoded headers checked; content verification delegated to adapter"
                ),
            },
        },
    )
    if checked["format_version"] == 2:
        dataset.metadata["provenance"] = _problem._provenance(
            EEGDatasetInfo(checked["input_info"]),
            dataset.metadata.pop("problem"),
            "aeon_neuro.datasets.materialize_classification_problem",
        )
    return dataset
