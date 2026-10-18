"""Convert native EEG recordings and MNE Raw objects to BIDS."""

from numbers import Integral
from os import PathLike
from pathlib import Path

import mne
import numpy as np
from mne.io import BaseRaw
from mne_bids import write_raw_bids

from aeon_neuro.utils.data_io import load_brainvision_to_mne


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
        if events.size and raw.first_samp != 0:
            # MNE-BIDS can shift explicit events by first_time again on export.
            # Reject the ambiguous time base instead of producing shifted markers.
            raise ValueError(
                "Explicit events with nonzero first_samp are not supported; "
                "use the complete zero-based recording."
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
    """Write a BrainVision recording or an MNE Raw object to BIDS.

    Parameters
    ----------
    source : str, os.PathLike or mne.io.BaseRaw
        BrainVision .vhdr path with its linked signal and marker files, or an
        existing MNE Raw recording. Other filename formats are not yet supported.
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
        Recordings with nonzero ``first_samp`` are rejected for explicit events
        because MNE-BIDS may shift their exported timing.
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
        If a filename does not have the .vhdr extension, or MNE-BIDS rejects the
        recording or output configuration.

    Notes
    -----
    Supplied metadata is assumed correct. No metadata repair, filtering, channel
    selection, re-referencing or epoching is performed. Annotations are written
    using MNE-BIDS conventions; event codes may be reassigned. Existing BIDS
    sidecars are not imported when reading a native BrainVision path.

    MNE and MNE-BIDS file, metadata and export errors propagate unchanged.
    Successful writing alone does not establish full BIDS validation.
    """
    if isinstance(source, (str, PathLike)):
        if Path(source).suffix.lower() != ".vhdr":
            raise ValueError(
                "Only BrainVision .vhdr filenames are supported; "
                "pass an MNE Raw object for other input formats."
            )
        # Keep samples on disk so MNE-BIDS can copy the original recording.
        raw = load_brainvision_to_mne(source)
    elif isinstance(source, BaseRaw):
        raw = source
    else:
        raise TypeError("source must be a BrainVision filename or an MNE Raw object.")

    # MNE-BIDS may omit out-of-range events or write metadata before discovering
    # an incomplete mapping. Check these inputs before handing it a destination.
    events = _prepare_events(raw, events, event_id)
    return write_raw_bids(
        raw,
        bids_path,
        events=events,
        event_id=event_id,
        format=format,
        allow_preload=allow_preload,
        overwrite=overwrite,
    )
