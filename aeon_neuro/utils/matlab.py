"""Read continuous MATLAB EEG using explicit schemas and dataset adapters."""

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import mne
import numpy as np


@dataclass(frozen=True)
class MatlabSchema:
    """Describe one continuous signal matrix without inferring its meaning.

    Parameters
    ----------
    signal_key : str
        Top-level MATLAB variable containing a two-dimensional signal matrix.
    sfreq : float
        Declared sampling rate in Hz.
    channel_names : sequence of str
        Ordered names of the selected physiological signals.
    channel_types : str or sequence of str
        MNE types: eeg, eog, ecg or emg. At least one EEG channel is required.
    units : str or sequence of str
        Physical units of source values: V, mV, uV (also µV/μV), or nV.
    sample_axis : int, default=0
        Axis containing samples in the original MATLAB matrix, 0 or 1.
    signal_columns : sequence of int or None, default=None
        Zero-based indices on the channel axis. None selects every column.
        Unselected columns are recorded in provenance, never inferred as EEG.
    scale : float or sequence of float, default=1.0
        Positive calibration multiplier before converting the specified units
        to volts; for example, IVa integer counts require 0.1 uV per count.
    highpass : float or None, default=None
        Known source high-pass cutoff in Hz; None means unknown. This describes
        processing already applied to the source and does not filter signals.
    lowpass : float or None, default=None
        Known source low-pass cutoff in Hz; None means unknown.
    timestamp_column : int or None, default=None
        Zero-based timestamp column on the original matrix's channel axis.
        It must not also be selected as a signal.
    timestamps_key : str or None, default=None
        Alternatively, a top-level MATLAB variable holding sample timestamps
        in seconds. Mutually exclusive with timestamp_column.
    timestamp_policy : {"nominal", "fit"}, default="nominal"
        Check timestamps against sfreq, or explicitly use the rate fitted to
        their endpoints. Fit permits at most 1 percent nominal clock drift.
        Neither policy interpolates signal samples.
    timing_tolerance : float, default=0.25
        Maximum timestamp residual in sample periods, in [0, 0.5).
    event_parser : callable or None, default=None
        Called with the decoded MATLAB dictionary; returns event dictionaries
        with onset, duration (seconds, or "n/a") and trial_type. Additional
        scalar columns are retained. This is trusted Python, not code from MAT.
    event_time_unit : {"seconds", "samples", "timestamps"}, default="seconds"
        Coordinate system of the parser's onset values. Seconds are relative
        to the first stored sample; timestamps use the signal's clock.
    event_index_base : {0, 1}, default=0
        Source sample origin for event_time_unit="samples". Other units use 0.
    event_descriptions : dict, default={}
        Descriptions for custom columns, as strings or BIDS JSON dictionaries.
    metadata : dict, default={}
        Descriptive EEG acquisition fields for the BIDS JSON sidecar.
    provenance : dict, default={}
        Adapter evidence, source processing history and other source context.

    Notes
    -----
    Units and calibration must come from the dataset documentation or caller.
    Epoch arrays, discontinuities and non-voltage measurements need another
    representation. Channel locations and wall-clock dates are not inferred.
    """

    signal_key: str
    sfreq: float
    channel_names: Sequence[str]
    channel_types: str | Sequence[str]
    units: str | Sequence[str]
    sample_axis: int = 0
    signal_columns: Sequence[int] | None = None
    scale: float | Sequence[float] = 1.0
    highpass: float | None = None
    lowpass: float | None = None
    timestamp_column: int | None = None
    timestamps_key: str | None = None
    timestamp_policy: str = "nominal"
    timing_tolerance: float = 0.25
    event_parser: Callable[[dict], Sequence[Mapping]] | None = None
    event_time_unit: str = "seconds"
    event_index_base: int = 0
    event_descriptions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


@dataclass
class MatlabRecording:
    """Continuous signals with event tables and explicit conversion provenance.

    Parameters
    ----------
    raw : mne.io.RawArray
        Calibrated signals in volts, with no annotations. Use events for the
        complete table; annotations cannot preserve every behavioural field.
    events : list of dict
        Events in source order. Onsets are aligned to the sample grid, with
        original coordinates retained in source_onset/source_sample/
        source_timestamp. Durations remain unchanged, including terminal trials
        that extend beyond the recorded interval.
    event_descriptions : dict
        BIDS JSON definitions of event columns.
    metadata : dict
        Descriptive EEG acquisition metadata, including explicit unknowns.
    provenance : dict
        Source filename and SHA-256, schema, calibration and timing decisions.
        Original per-sample timestamps remain in the source MAT file.
    diagnostics : list of dict
        Findings such as incomplete trial windows or unavailable references.

    Notes
    -----
    Construct with load_matlab_recording. Pass the entire result to
    convert_eeg_to_bids to preserve its event table and provenance. Passing only
    raw intentionally omits that additional information. The writer checks
    consistency again and stages output before publication. Signals and filter
    settings must remain unchanged; preprocess a separate raw.copy() for analysis.
    """

    raw: mne.io.BaseRaw
    events: list[dict]
    event_descriptions: dict
    metadata: dict
    provenance: dict
    diagnostics: list[dict]
    _raw_signature: tuple[float, float, str] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """Remember calibrated samples and filter settings before caller changes."""
        self._raw_signature = _raw_signature(self.raw)


def _raw_signature(raw):
    """Fingerprint samples in bounded chunks and retain MNE's filter settings."""
    digest = hashlib.sha256()
    for start in range(0, raw.n_times, 10000):
        block = raw.get_data(start=start, stop=min(start + 10000, raw.n_times))
        digest.update(block.tobytes())
    return raw.info["highpass"], raw.info["lowpass"], digest.hexdigest()


_METADATA_FIELDS = {
    "Manufacturer",
    "ManufacturersModelName",
    "EEGReference",
    "EEGGround",
    "EEGPlacementScheme",
    "TaskDescription",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "SoftwareFilters",
    "HardwareFilters",
    "PowerLineFrequency",
}
_EVENT_DESCRIPTIONS = {
    "trial_type": "Source event or trial category; not an analysis target selection.",
    "sample": "Zero-based sample index of the exported event onset.",
    "value": "Original source event value, where supplied; n/a means unavailable.",
    "source_onset": {
        "Description": "Original onset relative to the first sample.",
        "Units": "s",
    },
    "source_sample": "Original source sample index, before index-base conversion.",
    "source_timestamp": {
        "Description": "Original timestamp in the source clock.",
        "Units": "s",
    },
    "timing_error": {
        "Description": "Exported onset minus original relative onset.",
        "Units": "s",
    },
    "recording_covers_event": "Whether the complete event duration is recorded.",
}


def load_matlab_recording(
    path, *, schema=None, adapter=None, units=None, metadata=None, verbose=None
):
    """Read one continuous MATLAB recording and retain its event information.

    Parameters
    ----------
    path : str or os.PathLike
        MATLAB file (pre-7.3 or HDF5-based 7.3).
    schema : MatlabSchema or None, default=None
        Explicit mapping. Supply exactly one of schema and adapter.
    adapter : str, callable or None, default=None
        "tremmel2024", "bci_competition_iii_iva", or a callable accepting
        (decoded_mat, units=units) and returning a MatlabSchema. Built-in
        adapters use published layouts, not filename-based detection.
    units : str or sequence of str or None, default=None
        Required for Tremmel; all 30 source signal units must be supplied.
        IVa uses its documented 0.1 uV/count calibration and rejects overrides.
        For a custom schema, specify units in that schema instead.
    metadata : dict or None, default=None
        Explicit descriptive acquisition fields overriding the adapter/schema.
        Unknown reference and filters remain "n/a". Derived signal dimensions
        and sample rate cannot be overridden here.
    verbose : bool, str, int or None, default=None
        MNE logging verbosity.

    Returns
    -------
    MatlabRecording
        Signals, aligned events, source metadata and diagnostics.

    Raises
    ------
    ValueError
        For missing calibration, ambiguous layout, invalid events, unsupported
        epochs/discontinuities, or conflicting metadata.
    ImportError
        If the optional MATLAB reader is unavailable.

    Notes
    -----
    Requires aeon-neuro[eeg_io]. Samples are always loaded into memory.
    Event onsets outside the recording are rejected rather than clipped.
    Incomplete event durations are retained with a warning and a table flag;
    downstream epoch selection must decide how to handle them. No filtering,
    re-referencing, resampling, exclusion or classification splitting is done.
    """
    if (schema is None) == (adapter is None):
        raise ValueError("Supply exactly one of schema and adapter.")
    if schema is not None and units is not None:
        raise ValueError("For an explicit schema, set units in MatlabSchema.")
    try:
        from pymatreader import read_mat, whosmat
    except ImportError as error:
        raise ImportError(
            "MATLAB input requires aeon-neuro[eeg_io] (pymatreader>=1.3.2)."
        ) from error
    path = Path(path).expanduser()
    layouts = {name: shape for name, shape, _ in whosmat(path)}
    decoded = read_mat(path)
    adapter_name = "explicit_schema"
    if adapter is not None:
        from aeon_neuro.utils._matlab_adapters import _ADAPTERS

        if isinstance(adapter, str):
            if adapter not in _ADAPTERS:
                raise ValueError(
                    f"Unknown MATLAB adapter {adapter!r}; choose {tuple(_ADAPTERS)}."
                )
            adapter_name, adapter = adapter, _ADAPTERS[adapter]
        elif callable(adapter):
            adapter_name = getattr(adapter, "__name__", "custom")
        else:
            raise TypeError("adapter must be a registered name or callable.")
        schema = adapter(decoded, units=units)
    if not isinstance(schema, MatlabSchema):
        raise TypeError("schema/adapters must supply a MatlabSchema.")
    shape = layouts.get(schema.signal_key)
    if shape is None or len(shape) != 2:
        raise ValueError(
            "signal_key must identify a two-dimensional continuous MATLAB matrix;"
            " epochs are unsupported."
        )
    matrix = np.asarray(decoded[schema.signal_key])
    # pymatreader squeezes singleton channel axes; restore the known file shape.
    if matrix.ndim < 2:
        matrix = matrix.reshape(shape)
    elif matrix.shape != shape:
        raise ValueError(
            "Decoded signal dimensions differ from the MATLAB file layout."
        )
    raw, provenance = _signals(matrix, decoded, schema, verbose)
    rows = [] if schema.event_parser is None else schema.event_parser(decoded)
    events, descriptions, diagnostics = _events(rows, schema, raw, provenance)
    acquisition = deepcopy(schema.metadata)
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        acquisition.update(deepcopy(metadata))
    _validate_metadata(acquisition)
    acquisition.setdefault("EEGReference", "n/a")
    acquisition.setdefault("SoftwareFilters", "n/a")
    if acquisition["EEGReference"] == "n/a":
        diagnostics.append(
            {
                "code": "unknown_reference",
                "message": "Acquisition reference is not supplied; exported as n/a.",
            }
        )
    frequency = acquisition.get("PowerLineFrequency")
    raw.info["line_freq"] = None if frequency == "n/a" else frequency
    diagnostics.extend(deepcopy(schema.provenance.get("Diagnostics", [])))
    provenance.update(
        {
            "SourceFile": path.name,
            "SourceSHA256": _sha256(path),
            "Adapter": adapter_name,
            "AdapterContext": deepcopy(schema.provenance),
        }
    )
    json.dumps(provenance, allow_nan=False)
    return MatlabRecording(
        raw, events, descriptions, acquisition, provenance, diagnostics
    )


def _sha256(path):
    """Hash the source without an additional full-file allocation."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _broadcast(value, count, name):
    """Require one value per signal, permitting an explicit shared scalar."""
    values = [value] * count if isinstance(value, (str, Real)) else list(value)
    if len(values) != count:
        raise ValueError(f"{name} must have exactly one value per signal channel.")
    return values


def _signals(matrix, decoded, schema, verbose):
    """Validate signal axes, physical calibration and continuous sample timing."""
    if (
        not isinstance(schema.sample_axis, Integral)
        or isinstance(schema.sample_axis, bool)
        or schema.sample_axis not in (0, 1)
    ):
        raise ValueError("sample_axis must be 0 or 1.")
    if (
        not isinstance(schema.sfreq, Real)
        or isinstance(schema.sfreq, bool)
        or not np.isfinite(schema.sfreq)
        or schema.sfreq <= 0
    ):
        raise ValueError("sfreq must be finite and positive.")
    if matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all():
        raise ValueError("Signals and timestamps must be finite real numeric values.")
    matrix = matrix if schema.sample_axis == 0 else matrix.T
    if matrix.shape[0] < 2:
        raise ValueError("A continuous recording needs at least two samples.")
    columns = (
        list(range(matrix.shape[1]))
        if schema.signal_columns is None
        else list(schema.signal_columns)
    )
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(
            not isinstance(c, Integral)
            or isinstance(c, bool)
            or not 0 <= c < matrix.shape[1]
            for c in columns
        )
    ):
        raise ValueError("signal_columns must be unique, valid zero-based indices.")
    if isinstance(schema.channel_names, str):
        raise ValueError("channel_names must be a sequence of complete channel names.")
    names = list(schema.channel_names)
    if (
        len(names) != len(columns)
        or any(
            not isinstance(n, str) or not n.strip() or any(c in n for c in "\t\r\n")
            for n in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("channel_names must uniquely name every selected signal.")
    types = _broadcast(schema.channel_types, len(names), "channel_types")
    if "eeg" not in types or any(t not in {"eeg", "eog", "ecg", "emg"} for t in types):
        raise ValueError(
            "MATLAB EEG supports eeg/eog/ecg/emg types with at least one eeg signal."
        )
    if schema.units is None:
        raise ValueError(
            "Explicit physical units are required; calibration cannot be guessed."
        )
    units = _broadcast(schema.units, len(names), "units")
    factors = {"v": 1, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9}
    normalized = [
        u.lower().replace("µ", "u").replace("μ", "u") if isinstance(u, str) else ""
        for u in units
    ]
    if any(u not in factors for u in normalized):
        raise ValueError("units must be V, mV, uV/µV/μV or nV.")
    supplied_gains = _broadcast(schema.scale, len(names), "scale")
    if any(
        not isinstance(g, Real) or isinstance(g, (bool, np.bool_))
        for g in supplied_gains
    ):
        raise ValueError("scale must contain real numeric calibration factors.")
    gains = np.asarray(supplied_gains, dtype=float)
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("scale must contain finite positive calibration factors.")
    timestamps = None
    if schema.timestamp_column is not None:
        c = schema.timestamp_column
        if (
            schema.timestamps_key is not None
            or not isinstance(c, Integral)
            or isinstance(c, bool)
            or not 0 <= c < matrix.shape[1]
            or c in columns
        ):
            raise ValueError(
                "timestamp_column must be a separate valid column; do not also "
                "supply timestamps_key."
            )
        timestamps = matrix[:, c]
    elif schema.timestamps_key is not None:
        if schema.timestamps_key not in decoded:
            raise ValueError("timestamps_key is missing from the MATLAB file.")
        timestamps = np.asarray(decoded[schema.timestamps_key])
    sfreq, timing = _timing(timestamps, schema, len(matrix))
    volts_per_value = gains * [factors[u] for u in normalized]
    signal = matrix[:, columns].T * volts_per_value[:, None]
    if not np.isfinite(signal).all():
        raise ValueError("Calibration produced nonfinite signals.")
    for cutoff in (schema.highpass, schema.lowpass):
        if cutoff is not None and (
            not isinstance(cutoff, Real)
            or isinstance(cutoff, bool)
            or not np.isfinite(cutoff)
            or cutoff < 0
        ):
            raise ValueError("Source filter cutoffs must be finite nonnegative Hz.")
    if (
        schema.highpass is not None
        and schema.lowpass is not None
        and schema.highpass > schema.lowpass
    ):
        raise ValueError("Source highpass must not exceed lowpass.")
    info = mne.create_info(names, sfreq, types)
    # Import known history as native MNE readers do; do not apply a new filter.
    with info._unlock():
        if schema.highpass is not None:
            info["highpass"] = float(schema.highpass)
        if schema.lowpass is not None:
            info["lowpass"] = float(schema.lowpass)
    raw = mne.io.RawArray(signal, info, verbose=verbose)
    provenance = {
        "SignalVariable": schema.signal_key,
        "SampleAxis": int(schema.sample_axis),
        "SignalColumnsZeroBased": [int(c) for c in columns],
        "UnselectedColumnsZeroBased": sorted(
            set(range(matrix.shape[1])) - set(columns)
        ),
        "ChannelNames": names,
        "ChannelTypes": types,
        "SourceUnits": units,
        "VoltsPerSourceValue": volts_per_value.tolist(),
        "HighpassHz": None if schema.highpass is None else float(schema.highpass),
        "LowpassHz": None if schema.lowpass is None else float(schema.lowpass),
        "SampleCount": int(raw.n_times),
        "DeclaredSamplingFrequency": float(schema.sfreq),
        "SamplingFrequency": sfreq,
        "Timing": timing,
        "EventTimeUnit": schema.event_time_unit,
        "EventIndexBase": schema.event_index_base,
    }
    return raw, provenance


def _timing(values, schema, count):
    """Check both missing-sample gaps and drift against the chosen clock policy."""
    tolerance = schema.timing_tolerance
    if (
        not isinstance(tolerance, Real)
        or isinstance(tolerance, bool)
        or not np.isfinite(tolerance)
        or not 0 <= tolerance < 0.5
    ):
        raise ValueError("timing_tolerance must be in [0, 0.5).")
    if schema.timestamp_policy not in {"nominal", "fit"}:
        raise ValueError("timestamp_policy must be 'nominal' or 'fit'.")
    if values is None:
        if schema.timestamp_policy == "fit":
            raise ValueError("Fitting a time grid requires source timestamps.")
        return float(schema.sfreq), {"Policy": "nominal", "TimestampsAvailable": False}
    stamps = np.asarray(values, dtype=float)
    if stamps.ndim != 1 or len(stamps) != count or not np.isfinite(stamps).all():
        raise ValueError("Timestamps must be a finite vector matching the samples.")
    steps = np.diff(stamps)
    if np.any(steps <= 0) or np.any(steps * schema.sfreq > 1.5):
        raise ValueError(
            "Timestamps contain a gap, clock reset or nonmonotonic samples."
        )
    effective = (count - 1) / (stamps[-1] - stamps[0])
    if not np.isfinite(effective) or not np.isclose(
        effective / schema.sfreq, 1, rtol=0.01, atol=1e-12
    ):
        raise ValueError(
            "Timestamp rate disagrees with declared sfreq by more than 1 percent."
        )
    rate = float(effective if schema.timestamp_policy == "fit" else schema.sfreq)
    residual = (stamps - stamps[0]) * rate - np.arange(count)
    precision = 4 * np.spacing(np.max(np.abs(stamps))) * rate
    if precision >= 0.5 - tolerance or np.max(np.abs(residual)) > tolerance + precision:
        raise ValueError(
            "Timestamps do not fit the selected regular grid; review "
            "timestamp_policy and source continuity."
        )
    return rate, {
        "Policy": schema.timestamp_policy,
        "TimestampsAvailable": True,
        "FirstTimestamp": float(stamps[0]),
        "LastTimestamp": float(stamps[-1]),
        "EffectiveSamplingFrequency": float(effective),
        "MaxResidualSeconds": float(np.max(np.abs(residual)) / rate),
        "ToleranceSamples": float(tolerance),
        "SignalInterpolation": False,
    }


def _events(rows, schema, raw, provenance):
    """Align onsets explicitly and keep the original coordinates and durations."""
    unit, base = schema.event_time_unit, schema.event_index_base
    if (
        unit not in {"seconds", "samples", "timestamps"}
        or not isinstance(base, Integral)
        or isinstance(base, bool)
        or base not in (0, 1)
        or (unit != "samples" and base != 0)
    ):
        raise ValueError("Invalid event_time_unit/event_index_base combination.")
    if unit == "timestamps" and not provenance["Timing"]["TimestampsAvailable"]:
        raise ValueError(
            "Timestamp events require signal timestamps on the same clock."
        )
    events, incomplete = [], []
    rate = raw.info["sfreq"]
    timing = provenance["Timing"]
    source_column = {
        "seconds": "source_onset",
        "samples": "source_sample",
        "timestamps": "source_timestamp",
    }[unit]
    descriptions = deepcopy(_EVENT_DESCRIPTIONS)
    descriptions.update(deepcopy(schema.event_descriptions))
    for number, original in enumerate(rows):
        if not isinstance(original, Mapping):
            raise ValueError("event_parser must return event dictionaries.")
        row = deepcopy(dict(original))
        if not {"onset", "duration", "trial_type"} <= row.keys():
            raise ValueError("Each event needs onset, duration and trial_type.")
        if set(row) & {
            "sample",
            "source_onset",
            "source_sample",
            "source_timestamp",
            "timing_error",
            "recording_covers_event",
        }:
            raise ValueError("Event parser must not supply reserved alignment columns.")
        coordinate = row["onset"]
        if (
            not isinstance(coordinate, Real)
            or isinstance(coordinate, bool)
            or not np.isfinite(coordinate)
        ):
            raise ValueError("Event onsets must be finite real numbers.")
        if unit == "samples":
            if coordinate != int(coordinate):
                raise ValueError("Source event samples must be integers.")
            sample = int(coordinate) - base
            relative = sample / rate
        else:
            if timing["TimestampsAvailable"]:
                first, last = timing["FirstTimestamp"], timing["LastTimestamp"]
            else:
                first, last = 0.0, (raw.n_times - 1) / rate
            # Check the actual source span, not its jitter-free fitted endpoint.
            # A few ULPs accommodate floating-point subtraction/division only;
            # timing_tolerance does not authorize events outside the recording.
            precision = 4 * np.spacing(max(abs(first), abs(last)))
            if unit == "seconds":
                first, last = 0.0, last - first
            if not first - precision <= coordinate <= last + precision:
                raise ValueError(
                    f"Event {number} onset is outside the recorded sample span."
                )
            relative = coordinate - first
            sample = int(np.rint(relative * rate))
        if not 0 <= sample < raw.n_times:
            raise ValueError(
                f"Event {number} onset is outside the recorded sample span."
            )
        _validate_event_fields(row, descriptions)
        duration = row["duration"]
        covers = (
            "n/a"
            if duration == "n/a"
            else bool(sample / rate + duration <= raw.n_times / rate + 1e-9)
        )
        if covers is False:
            incomplete.append(number)
        row.update(
            onset=sample / rate,
            sample=sample,
            timing_error=sample / rate - relative,
            recording_covers_event=covers,
        )
        row[source_column] = coordinate
        events.append({key: _cell(value) for key, value in row.items()})
    fields = set().union(*(row.keys() for row in events)) if events else set()
    diagnostics = []
    if incomplete:
        diagnostics.append(
            {
                "code": "incomplete_event_windows",
                "rows_zero_based": incomplete,
                "message": (
                    "Event durations extend beyond the recording; retained without "
                    "clipping."
                ),
            }
        )
        warnings.warn(diagnostics[-1]["message"], UserWarning, stacklevel=3)
    return events, {k: v for k, v in descriptions.items() if k in fields}, diagnostics


def _validate_event_fields(row, descriptions):
    """Check event contents both when loading and before exporting mutable rows."""
    duration = row["duration"]
    if duration != "n/a" and (
        not isinstance(duration, Real)
        or isinstance(duration, bool)
        or not np.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("Event duration must be nonnegative seconds or 'n/a'.")
    if not isinstance(row["trial_type"], str) or not row["trial_type"].strip():
        raise ValueError("trial_type must be a nonempty string.")
    for key, value in row.items():
        if not isinstance(key, str) or not key or any(c.isspace() for c in key):
            raise ValueError(
                "Event column names must be nonempty strings without whitespace."
            )
        _cell(value)
        if key not in {"onset", "duration"}:
            description = descriptions.get(key)
            if (
                not isinstance(description, (str, dict))
                or not description
                or isinstance(description, dict)
                and not description.get("Description")
            ):
                raise ValueError(f"Missing or invalid event description for {key!r}.")


def _cell(value):
    """Require scalar TSV cells, writing explicit missing values as n/a."""
    if value is None:
        return "n/a"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if (
        not isinstance(value, (str, int, float, bool))
        or isinstance(value, float)
        and not np.isfinite(value)
    ):
        raise ValueError(
            "Event metadata must contain finite scalar values or explicit missing"
            " values."
        )
    if isinstance(value, str) and any(c in value for c in "\t\r\n"):
        raise ValueError("Event strings must not contain tabs or line breaks.")
    return value


def _validate_metadata(metadata):
    """Protect generated dimensions and reject invalid acquisition metadata."""
    if not isinstance(metadata, dict) or set(metadata) - _METADATA_FIELDS:
        raise ValueError("Only descriptive EEG acquisition metadata may be supplied.")
    for key, value in metadata.items():
        if key == "PowerLineFrequency":
            if value != "n/a" and (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError("PowerLineFrequency must be positive or 'n/a'.")
        elif key in {"SoftwareFilters", "HardwareFilters"}:
            if value != "n/a" and not isinstance(value, dict):
                raise ValueError(f"{key} must be a dictionary or 'n/a'.")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string.")
    json.dumps(metadata, allow_nan=False)
