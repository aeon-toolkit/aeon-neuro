"""Construct classification cases from recordings without losing their provenance."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass
class ClassificationDataset:
    """Signals, targets and provenance for a constructed classification problem.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Cases in (case, channel, time) order, or a list of (channel, time)
        arrays when case lengths differ. Channels and sampling rates agree.
    y : np.ndarray
        One string or integer class label per included case.
    groups : np.ndarray
        One explicitly supplied grouping identifier per included case.
    cases : list of dict
        Included case records in exactly the same order as X, y and groups.
        Source sample intervals are zero-based and stop-exclusive.
    exclusions : list of dict
        Excluded candidate records, with reasons. These have no output row.
    metadata : dict
        Recording metadata and the output channel order and sampling rate.

    Notes
    -----
    Use ``build_classification_dataset`` to construct a validated result. This
    container does not validate direct instantiation or subsequent mutations.
    It does not certify earlier preprocessing as leakage-free.
    """

    X: np.ndarray | list[np.ndarray]
    y: np.ndarray
    groups: np.ndarray
    cases: list[dict]
    exclusions: list[dict]
    metadata: dict


def make_recording_cases(X, metadata, *, labels, groups):
    """Describe one classification case per complete recording.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Recordings returned by ``load_bids_data``.
    metadata : list of dict
        Aligned recording metadata returned by ``load_bids_data``.
    labels : sequence of str or int
        Explicit recording labels, aligned with X. Derive these from the
        appropriate participant or recording attributes before calling.
    groups : sequence of str or int
        Explicit split groups, aligned with X. These need not be BIDS subjects.

    Returns
    -------
    list of dict
        Case records ready for ``build_classification_dataset`` or editing.
    """
    _, info = _recordings(X, metadata)
    labels, groups = _recording_targets(info, labels, groups)
    return [
        _case(item, 0, item["n_samples"], label, group)
        for item, label, group in zip(info, labels, groups)
    ]


def make_fixed_length_cases(
    X, metadata, *, labels, groups, window_size, step_size=None, start=0.0, stop=None
):
    """Describe fixed windows with inherited recording labels and groups.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Recordings returned by ``load_bids_data``.
    metadata : list of dict
        Aligned recording metadata returned by ``load_bids_data``.
    labels : sequence of str or int
        One explicit label per recording.
    groups : sequence of str or int
        One explicit split group per recording.
    window_size : float
        Window duration in seconds, positive and on each recording's sample grid.
    step_size : float or None, default=None
        Step in seconds. None uses window_size (nonoverlapping windows).
    start : float, default=0.0
        Nonnegative start time in seconds relative to each recording.
    stop : float or None, default=None
        Exclusive end time in seconds. None uses each recording's end.
        An explicit end is capped at the recording's end.

    Returns
    -------
    list of dict
        Complete windows in recording/time order. Trailing partial windows are
        not generated. Recordings shorter than a complete window contribute no
        cases. No annotation-based eligibility decisions are made here.

    Notes
    -----
    Durations must fall on the sample grid; no silent rounding is performed.
    Apply dataset-specific eligibility rules to the returned candidates, setting
    ``included=False`` and ``exclusion_reason`` on rejected windows.
    """
    _, info = _recordings(X, metadata)
    labels, groups = _recording_targets(info, labels, groups)
    step_size = window_size if step_size is None else step_size
    rows = []
    for item, label, group in zip(info, labels, groups):
        sfreq = item["sampling_frequency"]
        size = _samples(window_size, sfreq, "window_size", positive=True)
        step = _samples(step_size, sfreq, "step_size", positive=True)
        first = _samples(start, sfreq, "start")
        if first < 0:
            raise ValueError("start must be nonnegative.")
        last = item["n_samples"]
        if stop is not None:
            requested_stop = _samples(stop, sfreq, "stop")
            if requested_stop <= first:
                raise ValueError("stop must be greater than start.")
            last = min(last, requested_stop)
        rows.extend(
            _case(item, offset, offset + size, label, group)
            for offset in range(first, last - size + 1, step)
        )
    return rows


def make_event_cases(X, metadata, events, *, tmin, tmax):
    """Describe event-relative cases from explicit event targets and groups.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Recordings returned by ``load_bids_data``.
    metadata : list of dict
        Aligned recording metadata returned by ``load_bids_data``.
    events : iterable of dict
        Each row supplies ``recording_id``, ``onset`` (seconds relative to the
        first sample), ``label`` and ``group_id``. Optional fields, including
        ``case_id``, ``event_id``, ``participant_id``, ``channel_names``,
        ``output_channel_names`` and exclusion decisions, are preserved.
        Expand a multi-player event into one row per player before calling.
        For a pandas table, pass ``table.to_dict("records")``.
    tmin : float
        Inclusive start offset from each event in seconds; may be negative.
    tmax : float
        Exclusive stop offset from each event in seconds; greater than tmin.

    Returns
    -------
    list of dict
        Cases in input event order. Out-of-bounds intervals remain candidates;
        the builder can raise or record them as exclusions.

    Notes
    -----
    Onsets and offsets must fall on the recording's sample grid. Dataset-specific
    event conventions (such as one-based source sample columns) must be resolved
    by the recipe. No event column is automatically interpreted as a target.
    """
    _, info = _recordings(X, metadata)
    offsets = {}
    rows = []
    # Validate offsets even when events is empty.
    for item in info:
        sfreq = item["sampling_frequency"]
        first = _samples(tmin, sfreq, "tmin")
        last = _samples(tmax, sfreq, "tmax")
        if last <= first:
            raise ValueError("tmax must be greater than tmin.")
        offsets[item["recording_id"]] = (sfreq, first, last)
    for index, event in enumerate(events):
        row = deepcopy(dict(event))
        _required(row, ("recording_id", "onset", "label", "group_id"))
        recording_id = row["recording_id"]
        if not isinstance(recording_id, str) or recording_id not in offsets:
            raise ValueError(f"Unknown recording_id: {recording_id!r}.")
        sfreq, first, last = offsets[recording_id]
        onset = _samples(row["onset"], sfreq, "onset")
        row["start_sample"] = onset + first
        row["stop_sample"] = onset + last
        row.setdefault("case_id", f"{recording_id}::event-{index}")
        row.setdefault("included", True)
        rows.append(row)
    return rows


def build_classification_dataset(X, metadata, cases, *, on_out_of_bounds="raise"):
    """Extract explicitly defined cases and preserve labels, groups and provenance.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Recordings, each with shape (channels, samples), as returned by
        ``load_bids_data``. Input arrays are not modified.
    metadata : list of dict
        Aligned recording metadata. Each entry needs ``filepath`` (or an explicit
        unique ``recording_id``), ``channel_names`` and ``sampling_frequency``.
        Other metadata is retained. If supplied, ``n_samples`` must match X.
    cases : iterable of dict
        Rows from a case helper or a custom recipe. Required fields are
        ``case_id``, ``recording_id``, ``start_sample``, ``stop_sample``, ``label``
        and ``group_id``. Intervals are zero-based and stop-exclusive in the
        supplied recording's sample coordinates. Optional ``channel_names``
        selects/reorders source channels; ``output_channel_names`` explicitly
        names their output identities (useful for different player banks).
        ``included=False`` requires a nonempty ``exclusion_reason``; excluded
        candidates may have a missing label (None). Other fields are preserved.
    on_out_of_bounds : {"raise", "exclude"}, default="raise"
        Raise on intervals outside the recording, or retain them in exclusions
        with reason ``outside_recording``. Intervals are never clipped or padded.

    Returns
    -------
    ClassificationDataset
        Equal-length cases are stacked; unequal-length cases remain a list.
        All included cases must share output channel names/order and sampling
        frequency. No filtering, resampling, fitting or splitting is performed.

    Raises
    ------
    ValueError
        If inputs, labels, identifiers, channels or intervals are invalid, no
        included cases remain, or included cases are incompatible.

    Notes
    -----
    Channel renaming is an explicit recipe assertion, not automatic spatial
    harmonisation. The builder cannot infer leakage from existing signals.
    Record previous processing in metadata and validate the evaluation protocol
    separately. Output signals and metadata do not share mutable state with the
    inputs. Labels and groups must each be uniformly strings or integers.
    """
    if on_out_of_bounds not in {"raise", "exclude"}:
        raise ValueError("on_out_of_bounds must be 'raise' or 'exclude'.")
    recordings, info = _recordings(X, metadata)
    lookup = {
        item["recording_id"]: (
            x,
            item,
            {name: i for i, name in enumerate(item["channel_names"])},
        )
        for x, item in zip(recordings, info)
    }
    signals, included, excluded, ids = [], [], [], set()
    output_channels, output_sfreq = None, None
    for candidate in cases:
        row = deepcopy(dict(candidate))
        _required(
            row,
            (
                "case_id",
                "recording_id",
                "start_sample",
                "stop_sample",
                "label",
                "group_id",
            ),
        )
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError(f"case_id must be a unique nonempty string: {case_id!r}.")
        ids.add(case_id)
        recording_id = row["recording_id"]
        if not isinstance(recording_id, str) or recording_id not in lookup:
            raise ValueError(f"Unknown recording_id: {recording_id!r}.")
        recording, item, channel_indices = lookup[recording_id]
        start = _integer(row["start_sample"], "start_sample")
        stop = _integer(row["stop_sample"], "stop_sample")
        if stop <= start:
            raise ValueError(
                f"{case_id}: stop_sample must be greater than start_sample."
            )
        keep = row.get("included", True)
        if not isinstance(keep, (bool, np.bool_)):
            raise ValueError(f"{case_id}: included must be a boolean.")
        reason = row.get("exclusion_reason", "")
        if not isinstance(reason, str) or (not keep and not reason.strip()):
            raise ValueError(f"{case_id}: excluded cases need an exclusion_reason.")
        if keep and reason:
            raise ValueError(
                f"{case_id}: included cases cannot have an exclusion_reason."
            )
        row.update(start_sample=start, stop_sample=stop)
        row["group_id"] = _identifier(row["group_id"], "group_id")
        if row.get("participant_id") is not None:
            row["participant_id"] = _identifier(row["participant_id"], "participant_id")
        row["source_sampling_frequency"] = item["sampling_frequency"]
        row["source_file"] = item.get("filepath", recording_id)
        row.pop("row", None)
        if start < 0 or stop > recording.shape[1]:
            if keep and on_out_of_bounds == "raise":
                raise ValueError(f"{case_id}: interval is outside recording bounds.")
            if keep:
                keep, reason = False, "outside_recording"
        row.update(included=bool(keep), exclusion_reason=reason)
        if not keep:
            excluded.append(row)
            continue
        row["label"] = _identifier(row["label"], "label")
        names = _channels(row.get("channel_names", item["channel_names"]))
        missing = set(names) - channel_indices.keys()
        if missing:
            raise ValueError(f"{case_id}: unknown source channels: {sorted(missing)}.")
        renamed = _channels(row.get("output_channel_names", names))
        if len(renamed) != len(names):
            raise ValueError(
                f"{case_id}: output_channel_names must match channel count."
            )
        if output_channels is None:
            output_channels, output_sfreq = renamed, item["sampling_frequency"]
        if renamed != output_channels or item["sampling_frequency"] != output_sfreq:
            raise ValueError(
                "Included cases must have the same output channel names/order and "
                "sampling frequency. Select/reorder channels or resample explicitly."
            )
        picks = [channel_indices[name] for name in names]
        # Advanced indexing already copies, including for unequal-length output.
        signal = recording[picks, start:stop]
        if not np.isfinite(signal).all():
            raise ValueError(f"{case_id}: extracted signal contains non-finite values.")
        row.update(channel_names=names, output_channel_names=renamed, row=len(included))
        included.append(row)
        signals.append(signal)
    if not signals:
        raise ValueError("No included cases remain; inspect the candidate case table.")
    y = _identifiers([row["label"] for row in included], "labels")
    groups = _identifiers([row["group_id"] for row in included], "groups")
    if all(x.shape == signals[0].shape for x in signals):
        signals = np.stack(signals)
    return ClassificationDataset(
        X=signals,
        y=y,
        groups=groups,
        cases=included,
        exclusions=excluded,
        metadata={
            "recordings": info,
            "channel_names": output_channels,
            "sampling_frequency": output_sfreq,
            "sample_coordinates": "zero-based, stop-exclusive; supplied recordings",
        },
    )


def validate_group_split(dataset, train_indices, test_indices):
    """Validate a group-disjoint split without fitting or selecting a model.

    Parameters
    ----------
    dataset : ClassificationDataset
        A result from ``build_classification_dataset``.
    train_indices : sequence of int
        Nonempty, unique case indices for training.
    test_indices : sequence of int
        Nonempty, unique held-out case indices (validation or test).

    Raises
    ------
    ValueError
        If indices are invalid, or a case, group, recording or explicitly
        supplied participant_id occurs on both sides.

    Notes
    -----
    Returns None on success. Subset splits are supported for nested CV. This
    validator is for independent-group evaluation, not within-recording temporal
    splits. It cannot validate earlier fitting, temporal support, class coverage
    or the scientific correctness of the supplied grouping policy.
    """
    n_cases = len(dataset.y)
    if not (len(dataset.X) == len(dataset.groups) == len(dataset.cases) == n_cases):
        raise ValueError("Dataset signals, labels, groups and cases are not aligned.")
    train = _indices(train_indices, n_cases)
    test = _indices(test_indices, n_cases)
    if set(train) & set(test):
        raise ValueError("Training and held-out case indices overlap.")
    for key in ("group_id", "recording_id", "participant_id"):
        left = {dataset.cases[i].get(key) for i in train}
        right = {dataset.cases[i].get(key) for i in test}
        shared = (left & right) - {None}
        if shared:
            raise ValueError(f"Training and held-out cases share {key}: {shared}.")
    if set(dataset.groups[train]) & set(dataset.groups[test]):
        raise ValueError("Training and held-out cases share groups.")


def _recordings(X, metadata):
    if (
        not isinstance(X, (list, tuple, np.ndarray))
        or (isinstance(X, np.ndarray) and X.ndim == 0)
        or len(X) == 0
    ):
        raise ValueError("X must be a nonempty collection of 2D recordings.")
    if len(X) != len(metadata):
        raise ValueError("X and recording metadata must have the same length.")
    info, ids = [], set()
    for recording, entry in zip(X, metadata):
        if (
            not isinstance(recording, np.ndarray)
            or recording.ndim != 2
            or 0 in recording.shape
            or recording.dtype.kind not in "fiu"
        ):
            raise ValueError("Each recording must be a nonempty real numeric 2D array.")
        item = deepcopy(dict(entry))
        _required(item, ("channel_names", "sampling_frequency"))
        recording_id = item.get("recording_id", item.get("filepath"))
        if (
            not isinstance(recording_id, str)
            or not recording_id.strip()
            or recording_id in ids
        ):
            raise ValueError(
                "Recording metadata needs a unique nonempty recording_id or filepath."
            )
        ids.add(recording_id)
        item["recording_id"] = recording_id
        item["channel_names"] = _channels(item["channel_names"])
        if len(item["channel_names"]) != recording.shape[0]:
            raise ValueError(f"{recording_id}: channel metadata does not match X.")
        rate = item["sampling_frequency"]
        if (
            not isinstance(rate, Real)
            or isinstance(rate, bool)
            or not np.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError("sampling_frequency must be finite and positive.")
        item["sampling_frequency"] = float(rate)
        if (
            "n_samples" in item
            and _integer(item["n_samples"], "n_samples") != recording.shape[1]
        ):
            raise ValueError(f"{recording_id}: n_samples does not match X.")
        item["n_samples"] = recording.shape[1]
        info.append(item)
    return X, info


def _required(row, keys):
    missing = set(keys) - row.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}.")


def _channels(names):
    if not isinstance(names, (list, tuple, np.ndarray)) or (
        isinstance(names, np.ndarray) and names.ndim != 1
    ):
        raise ValueError("Channel names must be a nonempty sequence of unique strings.")
    names = list(names)
    if (
        not names
        or any(not isinstance(n, str) or not n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Channel names must be a nonempty sequence of unique strings.")
    return names


def _integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not {value!r}.")
    return int(value)


def _samples(seconds, sfreq, name, positive=False):
    if (
        not isinstance(seconds, Real)
        or isinstance(seconds, bool)
        or not np.isfinite(seconds)
    ):
        raise ValueError(f"{name} must be finite seconds.")
    value = float(seconds) * sfreq
    if not np.isfinite(value) or not np.isclose(value, round(value), rtol=0, atol=1e-7):
        raise ValueError(f"{name} must lie on the sample grid at {sfreq} Hz.")
    result = int(round(value))
    if positive and result <= 0:
        raise ValueError(f"{name} must span at least one sample.")
    return result


def _identifier(value, name):
    if isinstance(value, str) and value.strip():
        return str(value)
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError(f"{name} must be a nonempty string or integer, not {value!r}.")


def _identifiers(values, name):
    result = [_identifier(value, name) for value in values]
    if len({type(value) for value in result}) > 1:
        raise ValueError(f"{name} must be uniformly strings or integers.")
    return np.asarray(result)


def _recording_targets(info, labels, groups):
    for values, name in ((labels, "labels"), (groups, "groups")):
        if isinstance(values, str) or len(values) != len(info):
            raise ValueError(f"{name} must contain one value per recording.")
    return (
        _identifiers(labels, "labels").tolist(),
        _identifiers(groups, "groups").tolist(),
    )


def _case(item, start, stop, label, group):
    recording_id = item["recording_id"]
    return {
        "case_id": f"{recording_id}::samples-{start}-{stop}",
        "recording_id": recording_id,
        "start_sample": start,
        "stop_sample": stop,
        "label": label,
        "group_id": group,
        "included": True,
    }


def _indices(values, n_cases):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError(
            "Split indices must be nonempty one-dimensional integer arrays."
        )
    if (
        np.any(values < 0)
        or np.any(values >= n_cases)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("Split indices must be unique and within case bounds.")
    return values
