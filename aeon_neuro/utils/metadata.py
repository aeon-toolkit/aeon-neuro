"""Prepare explicit EEG metadata corrections without changing samples."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from mne.io import BaseRaw


def prepare_eeg_metadata(
    raw, *, channel_types=None, line_freq=None, correction_source=None
):
    """Report mains-frequency issues and apply verified corrections on a copy.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to inspect. Samples, annotations and input metadata are kept.
    channel_types : dict or None, default=None
        Verified channel-name to MNE channel-type corrections. MNE validates
        names/types and retains warnings when declared units change.
    line_freq : float or None, default=None
        Verified finite, positive mains frequency in Hz. None leaves the
        supplied value unchanged; no geographical default is inferred.
    correction_source : str or None, default=None
        Nonempty evidence reference required when supplying corrections. The
        reference is recorded, not automatically authenticated.

    Returns
    -------
    prepared : mne.io.BaseRaw
        A copy with explicit metadata corrections and unchanged samples/events.
        Deferred input remains deferred.
    report : dict
        JSON-compatible current channel roles and mains frequency, unresolved
        frequency issues, and actual corrections with before/after values and
        source. Nonfinite supplied frequency values are represented as strings.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw or channel_types is not a mapping.
    ValueError
        If corrections lack an evidence reference, line_freq is not a finite
        positive number, or MNE rejects a channel correction.

    Notes
    -----
    Only mains frequency is assessed for missing/invalid values; channel roles
    are listed for review, not inferred. Unknown frequency can remain BIDS n/a.
    Role changes may alter declared units but never rescale samples or repair
    acquisition calibration. Reference/ground, manufacturer, sidecar authority
    and scientific labels are outside this helper. Retain the returned report
    with conversion provenance; it is not automatically written into BIDS.
    """
    if not isinstance(raw, BaseRaw):
        raise TypeError("raw must be an MNE Raw recording.")
    if channel_types is not None and not isinstance(channel_types, Mapping):
        raise TypeError("channel_types must map channel names to MNE types.")
    if line_freq is not None and (
        isinstance(line_freq, bool)
        or not isinstance(line_freq, Real)
        or not isfinite(line_freq)
        or line_freq <= 0
    ):
        raise ValueError("line_freq must be a finite positive number in Hz.")
    if channel_types or line_freq is not None:
        if not isinstance(correction_source, str) or not correction_source.strip():
            raise ValueError("correction_source is required for verified corrections.")
        correction_source = correction_source.strip()

    # Preserve deferred access for native copying and avoid modifying caller state.
    prepared = raw.copy()
    if channel_types:
        prepared.set_channel_types(dict(channel_types))
    if line_freq is not None:
        prepared.info["line_freq"] = float(line_freq)

    corrections = []
    before_types = raw.get_channel_types()
    after_types = prepared.get_channel_types()
    for i, (name, before, after) in enumerate(
        zip(raw.ch_names, before_types, after_types)
    ):
        if before != after:
            corrections.append(
                {
                    "field": "channel_type",
                    "channel": name,
                    "before": before,
                    "after": after,
                    "source": correction_source,
                    "unit_before": int(raw.info["chs"][i]["unit"]),
                    "unit_after": int(prepared.info["chs"][i]["unit"]),
                }
            )
    before_freq = raw.info["line_freq"]
    after_freq = prepared.info["line_freq"]
    if line_freq is not None and before_freq != after_freq:
        corrections.append(
            {
                "field": "line_freq",
                "before": _report_frequency(before_freq),
                "after": float(after_freq),
                "source": correction_source,
            }
        )
    issues = []
    if after_freq is None:
        issues.append(
            {
                "field": "line_freq",
                "code": "unknown",
                "message": "Mains frequency is unknown; BIDS n/a remains permitted.",
            }
        )
    elif not isfinite(after_freq) or after_freq <= 0:
        issues.append(
            {
                "field": "line_freq",
                "code": "invalid",
                "message": "Supplied mains frequency must be finite and positive.",
            }
        )
    return prepared, {
        "assessed_fields": ["line_freq"],
        "channel_types": dict(zip(prepared.ch_names, after_types)),
        "line_freq": _report_frequency(after_freq),
        "issues": issues,
        "corrections": corrections,
    }


def _report_frequency(value):
    """Keep invalid native values visible without producing nonstandard JSON."""
    if value is None:
        return None
    return float(value) if isfinite(value) else str(value)
