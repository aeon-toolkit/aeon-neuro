"""Convert native EEG recordings and MNE Raw objects to BIDS."""

import csv
from os import PathLike

from mne.io import BaseRaw
from mne_bids import write_raw_bids

from aeon_neuro.utils.data_io import load_eeg_to_mne


def convert_eeg_to_bids(
    source, bids_path, *, format="auto", allow_preload=False, overwrite=False
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
        If a filename has an unsupported extension, or MNE-BIDS rejects the
        recording or output configuration.

    Notes
    -----
    Supplied metadata is assumed correct. No metadata repair, filtering, channel
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

    # Delegate metadata checks and preloaded export policy to MNE-BIDS.
    written = write_raw_bids(
        raw,
        bids_path,
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
