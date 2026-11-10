"""Validate current case layouts independently of original acquisition facts."""

from copy import deepcopy
from math import ceil, floor
from numbers import Real

import numpy as np

from aeon_neuro.datasets._classification import (
    _channels,
    _identifier,
    _integer,
    _samples,
)
from aeon_neuro.datasets._info_records import _CurrentView


def _collection(X):
    if isinstance(X, np.ndarray):
        if X.ndim != 3:
            raise ValueError("X must have case, channel and sample axes")
    elif not isinstance(X, (list, tuple)):
        raise ValueError("X must be a real numeric collection")
    for array in X:
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 2
            or 0 in array.shape
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()
        ):
            raise ValueError("Cases must be finite real numeric channel/sample arrays")
    return X


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _layout(names, rate, units, length, *, reference=None, origin=0.0):
    return {
        "channel_names": list(names),
        "sampling_frequency": rate,
        "units": list(units),
        "n_samples": length,
        "reference": reference,
        "source_origin_sample": origin,
        "source_sample_step": 1.0,
        "temporal_context": "pointwise",
    }


def _validate_view(view, sources):
    try:
        if (
            not isinstance(view, dict)
            or type(view.get("version")) is not int
            or view["version"] != 1
            or view["kind"] not in ("cases", "recordings")
            or not isinstance(view["rows"], list)
            or not isinstance(view["layouts"], list)
            or len(view["rows"]) != len(view["layouts"])
        ):
            raise ValueError("Invalid current-layout view")
        seen = set()
        for row, layout in zip(view["rows"], view["layouts"], strict=True):
            identity = row["case_id"]
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("Current case identities must be unique nonempty text")
            seen.add(identity)
            if row["recording_id"] not in sources and sources:
                raise ValueError("Current case refers to an unknown source")
            for key in ("participant_id", "group_id"):
                if row[key] is not None:
                    _identifier(row[key], key)
            bounds = row["source_support"]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("Source support must be a half-open sample interval")
            start, stop = (_integer(v, "source support") for v in bounds)
            if start < 0 or stop <= start:
                raise ValueError(
                    "Source support must have nonnegative increasing bounds"
                )
            source = sources.get(row["recording_id"])
            if source is not None and stop > source["n_samples"]:
                raise ValueError("Source support exceeds its recording")
            names = layout["channel_names"]
            if layout.get("temporal_context", "whole_row") not in (
                "pointwise",
                "whole_row",
            ):
                raise ValueError("Unknown current temporal context")
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(v, str) or not v.strip() for v in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError("Current channels must be unique nonempty names")
            source_names = layout.get("source_channel_names", names)
            if (
                not isinstance(source_names, list)
                or len(source_names) != len(names)
                or any(not isinstance(v, str) or not v.strip() for v in source_names)
                or len(set(source_names)) != len(source_names)
            ):
                raise ValueError("Source channel mapping must align with current names")
            _positive(layout["sampling_frequency"], "sampling_frequency")
            _positive(layout["source_sample_step"], "source_sample_step")
            if (
                not isinstance(layout["source_origin_sample"], Real)
                or isinstance(layout["source_origin_sample"], bool)
                or not np.isfinite(layout["source_origin_sample"])
                or layout["source_origin_sample"] < 0
            ):
                raise ValueError("Source origin must be a finite nonnegative sample")
            length = _integer(layout["n_samples"], "n_samples")
            if length <= 0:
                raise ValueError("Current n_samples must be positive")
            first = layout["source_origin_sample"]
            last = first + (length - 1) * layout["source_sample_step"]
            if first < start or last >= stop + 1e-8:
                raise ValueError(
                    "Current sample grid exceeds its original source support"
                )
            if source is not None:
                expected_step = (
                    source["sampling_frequency"] / layout["sampling_frequency"]
                )
                if not np.isclose(
                    layout["source_sample_step"], expected_step, rtol=1e-10, atol=0
                ):
                    raise ValueError(
                        "Current rate disagrees with its source sample map"
                    )
                if not set(source_names) <= set(source["channel_names"]):
                    raise ValueError("Current channels lack a source mapping")
            units = layout["units"]
            if (
                not isinstance(units, list)
                or len(units) != len(names)
                or any(
                    v is not None and (not isinstance(v, str) or not v) for v in units
                )
            ):
                raise ValueError("Current units must align with channel names")
        return _CurrentView.from_dict(view)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete current-layout view") from exc


def _source_units(descriptor, names):
    # MNE returns decoded SI values. BIDS sidecar units describe the stored file
    # and must not be copied onto arrays that the native reader already scaled.
    from mne.io.constants import FIFF

    labels = {int(FIFF.FIFF_UNIT_V): "V", int(FIFF.FIFF_UNIT_T): "T"}
    codes = descriptor.channel_units_fiff
    indices = {name: i for i, name in enumerate(descriptor.channel_names)}
    return [
        labels.get(codes[indices[name]]) if indices[name] < len(codes) else None
        for name in names
    ]


def _from_arrays(X, sfreq, channel_names, units, subjects, case_ids, groups):
    arrays = _collection(X)
    rate = _positive(sfreq, "sfreq")
    count = len(arrays)
    names = _channels(channel_names)
    units = [None] * len(names) if units is None else list(units)
    subjects = [None] * count if subjects is None else list(subjects)
    groups = subjects if groups is None else list(groups)
    ids = [f"case:{i}" for i in range(count)] if case_ids is None else list(case_ids)
    if not all(len(values) == count for values in (subjects, groups, ids)):
        raise ValueError("Subjects, groups and case IDs must align with X")
    rows, layouts = [], []
    for i, array in enumerate(arrays):
        if array.shape[0] != len(names):
            raise ValueError("Channel names do not match X")
        rows.append(
            {
                "case_id": ids[i],
                "recording_id": None,
                "participant_id": (
                    None if subjects[i] is None else _identifier(subjects[i], "subject")
                ),
                "group_id": (
                    None if groups[i] is None else _identifier(groups[i], "group")
                ),
                "source_support": [0, int(array.shape[1])],
            }
        )
        layouts.append(_layout(names, rate, units, int(array.shape[1])))
    return _validate_view(
        {"version": 1, "kind": "cases", "rows": rows, "layouts": layouts}, {}
    )


def _case_view(info, dataset):
    sources = {row["recording_id"]: row for row in info.metadata}
    source_records = {record.recording_id: record._source for record in info.recordings}
    current = (
        {
            row.recording_id: (row, layout)
            for row, layout in zip(info._view.rows, info._view.layouts, strict=True)
        }
        if info._view is not None and info._view.kind == "recordings"
        else {}
    )
    names = list(dataset.metadata["channel_names"])
    rate = dataset.metadata["sampling_frequency"]
    rows, layouts = [], []
    if not (
        len(dataset.X) == len(dataset.y) == len(dataset.groups) == len(dataset.cases)
    ):
        raise ValueError(
            "Classification signals, labels, groups and cases are not aligned"
        )
    for i, candidate in enumerate(dataset.cases):
        row = deepcopy(candidate)
        source = sources.get(row["recording_id"])
        if source is None:
            raise ValueError("Case has no matching source recording")
        if row["label"] != dataset.y[i] or row["group_id"] != dataset.groups[i]:
            raise ValueError("Case labels/groups disagree with their arrays")
        # A case can identify a person within a shared recording. Keep that
        # explicit case identity instead of replacing it with the BIDS entity.
        row.setdefault("participant_id", source["participant_id"])
        input_names = row["channel_names"]
        for key in ("session", "run"):
            row.setdefault(key, source.get(key, source.get("entities", {}).get(key)))
        input_row, input_layout = current.get(row["recording_id"], (None, None))
        row["source_support"] = (
            deepcopy(input_row.source_support)
            if input_row is not None
            else [row["start_sample"], row["stop_sample"]]
        )
        rows.append(row)
        layout = _layout(
            names,
            rate,
            (
                [
                    input_layout.units[input_layout.channel_names.index(name)]
                    for name in input_names
                ]
                if input_layout is not None
                else _source_units(source_records[row["recording_id"]], input_names)
            ),
            row["stop_sample"] - row["start_sample"],
            reference=(
                input_layout.reference
                if input_layout is not None
                else source.get("eeg_metadata", {}).get("EEGReference")
            ),
            origin=row["start_sample"],
        )
        if input_names != names:
            # Case builders allow explicit logical aliases. Preserve the actual
            # input picks so units and source geometry do not follow an alias.
            layout["source_channel_names"] = list(input_names)
        if input_layout is not None:
            layout["source_origin_sample"] = (
                input_layout.source_origin_sample
                + row["start_sample"] * input_layout.source_sample_step
            )
            layout["source_sample_step"] = input_layout.source_sample_step
            layout["temporal_context"] = input_layout.temporal_context
            if layout["temporal_context"] == "pointwise":
                # Pointwise operations do not mix times. Map only this case,
                # unlike filtering/resampling which retain the full input row.
                first = layout["source_origin_sample"]
                stop = first + layout["n_samples"] * layout["source_sample_step"]
                row["source_support"] = [
                    floor(first),
                    min(ceil(stop), input_row.source_support[1]),
                ]
        layouts.append(layout)
    view = _validate_view(
        {"version": 1, "kind": "cases", "rows": rows, "layouts": layouts}, sources
    )
    _check_layout(dataset.X, view)
    return view


def _recording_view(info):
    rows, layouts = [], []
    for recording in info.recordings:
        source = recording._source
        rows.append(
            {
                "case_id": "recording:" + source.recording_id,
                "recording_id": source.recording_id,
                "participant_id": source.participant_id,
                "group_id": source.participant_id,
                "source_support": [0, source.n_samples],
            }
        )
        layouts.append(
            _layout(
                source.channel_names,
                source.sampling_frequency,
                _source_units(source, source.channel_names),
                source.n_samples,
                reference=source.eeg_metadata.get("EEGReference"),
            )
        )
    return _CurrentView.from_dict(
        {"version": 1, "kind": "recordings", "rows": rows, "layouts": layouts}
    )


def _current_metadata(info):
    if info._view is None:
        return info.metadata
    if info._view.kind != "recordings":
        raise ValueError(
            "Case construction requires recordings, not already built cases"
        )
    sources = {row.recording_id: row for row in info.metadata}
    descriptors = []
    for row, layout in zip(info._view.rows, info._view.layouts, strict=True):
        descriptor = deepcopy(sources[row.recording_id])
        descriptor.update(
            # Descriptors are public detached views, so their channel list must
            # not expose the mutable layout used by later estimator checks.
            channel_names=list(layout.channel_names),
            sampling_frequency=layout.sampling_frequency,
            n_samples=layout.n_samples,
        )
        descriptors.append(descriptor)
    return descriptors


def _recording_inputs(info, onset_column="onset"):
    descriptors = _current_metadata(info)
    if info._view is None:
        return descriptors, info.events
    sources = {row.recording_id: row for row in info.metadata}
    events, grouped = [], {}
    for original in info.events:
        grouped.setdefault(original["recording_id"], []).append(original)
    for row, layout in zip(info._view.rows, info._view.layouts, strict=True):
        source = sources[row.recording_id]
        for original in grouped.get(row.recording_id, []):
            event = deepcopy(original)
            # Bound onsets remain seconds from the first current sample; retain
            # the entire published row independently for source-level joins.
            if onset_column not in event:
                raise ValueError(f"Missing bound event onset: {onset_column}")
            if "acquisition_event" in event:
                raise ValueError(
                    "Source event column collides with derived acquisition_event"
                )
            event["acquisition_event"] = deepcopy(original)
            event[onset_column] = (
                float(event[onset_column])
                - layout.source_origin_sample / source["sampling_frequency"]
            )
            events.append(event)
    return descriptors, events


def _check_layout(X, view):
    arrays = _collection(X)
    if len(arrays) != len(view.rows):
        raise ValueError("X and current metadata have different case counts")
    for array, layout in zip(arrays, view.layouts, strict=True):
        if array.shape != (len(layout.channel_names), layout.n_samples):
            raise ValueError("X does not match its current channel/sample layout")


def _uniform(layouts, key):
    values = [getattr(layout, key) for layout in layouts]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"Current {key} is empty or differs across rows")
    value = values[0]
    return tuple(value) if isinstance(value, list) else value


def _events_for_cases(events, metadata, label_column, group_column):
    import json

    lookup = {row["recording_id"]: row for row in metadata}
    rows = []
    for event in events:
        source = lookup[event["recording_id"]]
        if label_column not in event:
            raise ValueError(f"Missing event label column: {label_column}")
        label = _identifier(event[label_column], "event label")
        if group_column not in source:
            raise ValueError(f"Missing recording group column: {group_column}")
        group = _identifier(source[group_column], "event group")
        rows.append(
            {
                "case_id": "event:"
                + json.dumps([event["recording_id"], event["bids_event_id"]]),
                "recording_id": event["recording_id"],
                "onset": float(event["onset"]),
                "label": label,
                "group_id": group,
                "event_id": event["bids_event_id"],
                "source_event": deepcopy(event),
            }
        )
    return rows


def _event_sample(onset, sfreq):
    """Return an exact grid coordinate, or None without rounding source facts."""
    onset = float(onset)
    if not np.isfinite(onset):
        raise ValueError("Event onset must be finite")
    try:
        return _samples(onset, sfreq, "event onset")
    except ValueError:
        return None
