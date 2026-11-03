"""Validate the continuous voltage-stream representation before constructing Raw."""

from collections.abc import Mapping
from numbers import Real

import mne
import numpy as np


def _read_xdf(
    path, *, preload=False, verbose=None, timing_tolerance=0.25, **reader_kwargs
):
    """Read one EEG stream while checking raw and corrected timestamp geometry."""
    if not isinstance(preload, bool):
        raise ValueError("XDF always loads into memory; preload must be a bool.")
    if (
        not isinstance(timing_tolerance, Real)
        or isinstance(timing_tolerance, (bool, np.bool_))
        or not np.isfinite(timing_tolerance)
        or not 0 <= timing_tolerance < 0.5
    ):
        raise ValueError(
            "timing_tolerance must be finite and satisfy 0 <= value < 0.5."
        )
    callback = reader_kwargs.pop("on_chunk", None)
    if callback is not None and not callable(callback):
        raise ValueError("XDF on_chunk must be callable.")
    original_times, errors = {}, []

    def capture(values, stamps, header, stream_id):
        original = values, stamps, header
        if errors:
            return original
        try:
            if callback is not None:
                values, stamps, header = callback(values, stamps, header, stream_id)
            if _kind(header) == "eeg":
                # PyXDF fits timestamps in place after decoding. Keep only the
                # original timestamp vectors, not a second copy of the signals.
                original_times.setdefault(stream_id, []).append(
                    np.array(stamps, dtype=float, copy=True)
                )
            return values, stamps, header
        except Exception as error:
            # PyXDF catches callback exceptions as recoverable chunk corruption.
            # Defer the error so that a failed callback cannot silently drop data.
            errors.append(error)
            return original

    import pyxdf

    streams, _ = pyxdf.load_xdf(str(path), on_chunk=capture, **reader_kwargs)
    if errors:
        raise ValueError("XDF on_chunk failed; no recording was accepted.") from errors[
            0
        ]
    eeg = [stream for stream in streams if _kind(stream) == "eeg"]
    if len(eeg) != 1:
        raise ValueError("Select exactly one EEG stream using select_streams.")
    stream = eeg[0]
    info = stream["info"]
    data = np.asarray(stream["time_series"], dtype=float)
    if data.ndim != 2 or len(data) < 2 or not data.shape[1]:
        raise ValueError("XDF EEG needs at least two samples and a 2D channel array.")
    if not np.isfinite(data).all():
        raise ValueError("XDF EEG samples must be finite.")
    if len(info.get("segments", [])) > 1:
        raise ValueError("XDF EEG has recording gaps; continuous data is required.")
    nominal = float(_text(info, "nominal_srate"))
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError("XDF EEG nominal sampling rate must be finite and positive.")
    chunks = original_times.get(info.get("stream_id"))
    if not chunks:
        raise ValueError("XDF original timestamps could not be verified.")
    original = np.concatenate(chunks)
    _time_grid(original, timing_tolerance, nominal=nominal)
    stamps, sfreq = _time_grid(stream["time_stamps"], timing_tolerance)
    if len(original) != len(stamps) or len(stamps) != len(data):
        raise ValueError("XDF timestamps and samples must have the same length.")
    effective = info.get("effective_srate")
    if (
        not isinstance(effective, Real)
        or isinstance(effective, (bool, np.bool_))
        or not np.isfinite(effective)
        or effective <= 0
        or not np.isclose(effective, sfreq, rtol=1e-7, atol=0)
    ):
        raise ValueError("XDF effective sampling rate disagrees with its timestamps.")
    # A regular grid can still describe the wrong rate. Allow one-percent clock
    # drift, with only roundoff allowance at the relative-rate boundary.
    if not np.isclose(sfreq / nominal, 1.0, rtol=0.01, atol=1e-12):
        raise ValueError(
            "XDF effective sampling rate differs from the nominal rate by more than 1%."
        )

    names, types, scales = _channels(info, data.shape[1])
    # Validate all dimensions before multiplication: a one-channel recording and
    # two descriptors would otherwise broadcast into two copies of that signal.
    data = data.T * np.asarray(scales)[:, None]
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, types), verbose=verbose)
    onsets, descriptions = _markers(streams, stamps)
    raw.set_annotations(mne.Annotations(onsets, np.zeros(len(onsets)), descriptions))
    return raw


def _kind(stream):
    """Ignore unrelated stream types without guessing missing EEG identity."""
    info = stream.get("info", {}) if isinstance(stream, Mapping) else {}
    values = info.get("type", []) if isinstance(info, Mapping) else []
    if not isinstance(values, (list, tuple)):
        return ""
    return (
        values[0].strip().lower()
        if len(values) == 1 and isinstance(values[0], str)
        else ""
    )


def _text(mapping, field, default=None):
    """Require an unambiguous XML text field rather than taking a partial value."""
    values = mapping.get(field, [default]) if isinstance(mapping, Mapping) else []
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 1
        or not isinstance(values[0], str)
        or not values[0].strip()
    ):
        raise ValueError(f"XDF channel/stream metadata needs a valid {field!r}.")
    return values[0]


def _time_grid(values, tolerance, *, nominal=None):
    """Bound the error incurred by replacing timestamps with a regular grid."""
    stamps = np.asarray(values, dtype=float)
    if stamps.ndim != 1 or len(stamps) < 2 or not np.isfinite(stamps).all():
        raise ValueError("XDF EEG timestamps must be finite with at least two samples.")
    steps = np.diff(stamps)
    if np.any(steps <= 0):
        raise ValueError("XDF EEG timestamps must be strictly increasing.")
    # Check nominal adjacency separately: a fitted average rate can otherwise
    # turn a dropped sample into a longer, apparently regular sample period.
    if nominal is not None and np.any(steps * nominal > 1.5):
        raise ValueError(
            "XDF timestamps contain a gap or disagree with the nominal rate."
        )
    period = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("XDF EEG timestamps need a finite positive sample interval.")
    residual = (stamps - stamps[0]) / period - np.arange(len(stamps))
    # LSL clocks can have a large absolute origin. Allow float representation
    # error in addition to the explicit tolerance expressed in sample periods.
    precision = 4 * np.spacing(np.max(np.abs(stamps))) / period
    if precision >= 0.5 - tolerance:
        raise ValueError("XDF timestamps have insufficient precision for this grid.")
    if np.any(np.abs(residual) > tolerance + precision):
        raise ValueError("XDF EEG timestamps do not fit the supported regular grid.")
    return stamps, 1.0 / period


def _channels(info, count):
    """Check declared channels, identities and voltage calibration together."""
    try:
        declared = int(_text(info, "channel_count"))
        channels = info["desc"][0]["channels"][0]["channel"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "XDF EEG needs channel descriptors and channel_count."
        ) from error
    if declared != count or len(channels) != count:
        raise ValueError(
            "XDF channel descriptors/count disagree with the signal array."
        )
    names = [_text(channel, "label") for channel in channels]
    if len(set(names)) != count:
        raise ValueError("XDF channel labels must be unique.")
    types = [_text(channel, "type", "EEG").strip().lower() for channel in channels]
    voltage_types = {"eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs", "bio"}
    if any(kind not in voltage_types for kind in types):
        raise ValueError(
            "XDF supports voltage channel types, not dimensionless channels."
        )
    units = [_text(channel, "unit").strip().lower() for channel in channels]
    units = [unit.replace("µ", "u").replace("μ", "u") for unit in units]
    scales = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9}
    if any(unit not in scales for unit in units):
        raise ValueError("XDF channels need voltage units: V, mV, uV/µV or nV.")
    return names, types, [scales[unit] for unit in units]


def _markers(streams, eeg_stamps):
    """Reject unsupported marker coordinates before MNE can crop them silently."""
    onsets, descriptions = [], []
    duration = eeg_stamps[-1] - eeg_stamps[0]
    precision = 4 * np.spacing(np.max(np.abs(eeg_stamps)))
    for stream in streams:
        if _kind(stream) not in {"marker", "markers", "event", "events"}:
            continue
        stamps = np.asarray(stream["time_stamps"], dtype=float)
        values = np.asarray(stream["time_series"], dtype=object)
        if not stamps.size and not values.size:
            continue
        if (
            stamps.ndim != 1
            or not np.isfinite(stamps).all()
            or values.ndim != 2
            or len(values) != len(stamps)
            or values.shape[1] != int(_text(stream["info"], "channel_count"))
        ):
            raise ValueError("XDF marker samples and finite timestamps must align.")
        relative = stamps - eeg_stamps[0]
        if np.any(relative < -precision) or np.any(relative > duration + precision):
            raise ValueError(
                "XDF marker timestamps lie outside the selected EEG samples."
            )
        # Only float roundoff at an endpoint is clipped; actual out-of-range
        # markers fail above rather than disappearing in Raw.set_annotations.
        onsets.extend(np.clip(relative, 0, duration))
        descriptions.extend("/".join(map(str, row)) for row in values)
    return onsets, descriptions
