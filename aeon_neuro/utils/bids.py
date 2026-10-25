"""Convert native EEG recordings and MNE Raw objects to BIDS."""

import csv
from math import isfinite
from numbers import Integral
from os import PathLike

import mne
import numpy as np
from mne.io import BaseRaw
from mne_bids import write_raw_bids

from aeon_neuro.utils.data_io import load_eeg_to_mne


def _prepare_events(raw, events, event_id):
    """Check explicit event bounds and mappings before dataset files are created."""
    if event_id is not None:
        if not isinstance(event_id, dict) or any(
            not isinstance(name, str)
            or isinstance(code, bool)
            or not isinstance(code, Integral)
            for name, code in event_id.items()
        ):
            raise TypeError("event_id must map string descriptions to integer codes.")
    if events is not None:
        if event_id is None:
            raise ValueError("event_id is required when supplying events.")
        if isinstance(events, (str, PathLike)):
            events = mne.read_events(events)
        if (
            not isinstance(events, np.ndarray)
            or events.ndim != 2
            or events.shape[1] != 3
        ):
            raise ValueError(
                "events must be an MNE event file or an array of shape (n, 3)."
            )
        if not np.issubdtype(events.dtype, np.integer):
            raise TypeError("events must contain integer sample indices and codes.")
        if np.any(events[:, 0] < raw.first_samp) or np.any(
            events[:, 0] > raw.last_samp
        ):
            raise ValueError(
                f"Event samples must lie between {raw.first_samp} and {raw.last_samp}."
            )
        missing = set(events[:, 2]) - set(event_id.values())
        if missing:
            raise ValueError(
                f"event_id is missing descriptions for codes: {sorted(missing)}"
            )
        if events.size and raw.first_samp != 0 and raw.annotations.orig_time is None:
            # MNE-BIDS can shift explicit events by first_time again on export.
            # Reject the ambiguous time base instead of producing shifted markers.
            raise ValueError(
                "Explicit events with nonzero first_samp require a known time origin; "
                "use the complete zero-based recording when timing is unknown."
            )
    if event_id is not None:
        # MNE-BIDS supplies its own code for this reserved acquisition annotation.
        missing = set(raw.annotations.description) - set(event_id) - {"BAD_ACQ_SKIP"}
        if missing or (len(raw.annotations) and not event_id):
            raise ValueError(
                "event_id must include all native annotation descriptions."
            )
    return events


def convert_eeg_to_bids(
    source,
    bids_path,
    *,
    events=None,
    event_id=None,
    format="auto",
    allow_preload=False,
    overwrite=False,
):
    """Write a native EEG recording or an MNE Raw object to BIDS.

    Parameters
    ----------
    source : str, os.PathLike or mne.io.BaseRaw
        BrainVision .vhdr, EDF/EDF+ .edf, BioSemi BDF/BDF+ .bdf or continuous
        EEGLAB .set path with required companions, or an existing MNE Raw.
        For reader-specific settings, first use ``load_eeg_to_mne`` and pass Raw.
    bids_path : mne_bids.BIDSPath
        Destination with root, subject and task set. Supply session, run and
        datatype where needed; identifiers are not inferred from the source path.
    events : path-like, np.ndarray or None, default=None
        MNE events file or array of shape (n_events, 3), forwarded to MNE-BIDS.
        These are written in addition to native annotations. For Status-channel
        triggers, extract events explicitly with ``mne.find_events`` using the
        acquisition's trigger settings; no codes or classes are inferred here.
        Samples use MNE's absolute sample indices, including ``raw.first_samp``.
        Bounds, integer values and mapping coverage are checked before writing.
        Recordings with nonzero ``first_samp`` and unknown annotation time origin
        are rejected because MNE-BIDS may shift their exported timing.
    event_id : dict or None, default=None
        Description-to-code mapping. Required when supplying ``events``; include
        descriptions for existing annotations too, following MNE-BIDS rules.
    format : str, default="auto"
        Output encoding passed to ``mne_bids.write_raw_bids``. The default copies
        supported source files where possible. Preloaded recordings require an
        explicit encoding, such as ``"BrainVision"``, and its export backend.
    allow_preload : bool, default=False
        Permit writing preloaded Raw objects, including RawArray, when True.
        Follow MNE-BIDS requirements for the selected output encoding. BrainVision
        export requires the optional ``pybv`` package; unchanged copying does not.
    overwrite : bool, default=False
        Whether MNE-BIDS may overwrite an existing output recording.

    Returns
    -------
    written : mne_bids.BIDSPath
        Actual output path returned by MNE-BIDS, including the recording extension.

    Raises
    ------
    TypeError
        If source is neither a filename nor an MNE Raw object.
    ValueError
        If supplied line_freq is nonfinite or nonpositive, a filename has an
        unsupported extension, or MNE-BIDS rejects the recording or output
        configuration.

    Notes
    -----
    Known mains frequency must be finite and positive; None remains unknown.
    Other supplied metadata is assumed correct. No metadata repair, filtering, channel
    selection, re-referencing or epoching is performed. Annotations are written
    using MNE-BIDS conventions; event codes may be reassigned and onsets rounded
    to the nearest sample. Existing BIDS sidecars are not imported when reading
    a native filename. Status channels
    remain in the recording even when no explicit events are supplied.

    MNE and MNE-BIDS file, metadata and export errors propagate unchanged.
    Successful writing alone does not establish full BIDS validation.
    """
    if isinstance(source, (str, PathLike)):
        # Keep samples on disk so MNE-BIDS can copy the original recording.
        raw = load_eeg_to_mne(source)
    elif isinstance(source, BaseRaw):
        raw = source
    else:
        raise TypeError("source must be an EEG filename or an MNE Raw object.")

    # The writer can otherwise serialize NaN or negative frequency into metadata.
    frequency = raw.info["line_freq"]
    if frequency is not None and (not isfinite(frequency) or frequency <= 0):
        raise ValueError("line_freq must be finite and positive, or None if unknown.")

    # MNE-BIDS may omit out-of-range events or write metadata before discovering
    # an incomplete mapping. Check these inputs before handing it a destination.
    events = _prepare_events(raw, events, event_id)
    # Delegate metadata checks and preloaded export policy to MNE-BIDS.
    written = write_raw_bids(
        raw,
        bids_path,
        events=events,
        event_id=event_id,
        format=format,
        allow_preload=allow_preload,
        overwrite=overwrite,
    )
    _preserve_stim_units(raw, written)
    return written


def _preserve_stim_units(raw, written):
    """Keep native dimensionless trigger units in the generated channel table."""
    dimensionless = {
        channel["ch_name"]
        for channel, kind in zip(raw.info["chs"], raw.get_channel_types())
        if kind == "stim" and int(channel["unit"]) == -1
    }
    if not dimensionless:
        return
    path = written.copy().update(suffix="channels", extension=".tsv").fpath
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row["name"] in dimensionless:
            # MNE-BIDS exports unknown units as n/a, then reapplying STIM on read
            # can assign volts. Its supported 'arbitrary' spelling retains NONE.
            # Change only the unit label; integer codes and voltage channels stay
            # in their original numeric representation.
            row["units"] = "arbitrary"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
