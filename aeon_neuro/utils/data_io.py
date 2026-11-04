"""Read EEG recordings and auxiliary dataset information."""

import json
import os
import re
import warnings
from pathlib import Path

import mne
import numpy as np

from aeon_neuro.utils._xdf import _read_xdf


def _check_edf_layout(path, bytes_per_sample):
    """Reject incomplete records and unsupported discontinuities before decoding."""
    with open(path, "rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError("Incomplete EDF/BDF header.")
        if fixed[192:236].startswith((b"EDF+D", b"BDF+D")):
            raise ValueError("Discontinuous EDF+D/BDF+D recordings are not supported.")
        try:
            header_bytes = int(fixed[184:192])
            records = int(fixed[236:244])
            channels = int(fixed[252:256])
            if channels <= 0 or header_bytes != 256 * (channels + 1) or records < -1:
                raise ValueError
            signal_header = stream.read(256 * channels)
            if len(signal_header) != 256 * channels:
                raise ValueError
            # EDF and BDF share these fixed-width fields; only sample width differs.
            counts = [
                int(
                    signal_header[216 * channels + 8 * i : 216 * channels + 8 * (i + 1)]
                )
                for i in range(channels)
            ]
            if any(count <= 0 for count in counts):
                raise ValueError
        except ValueError as error:
            raise ValueError("Invalid EDF/BDF record layout.") from error
        payload = os.fstat(stream.fileno()).st_size - header_bytes
        record_bytes = bytes_per_sample * sum(counts)
        # -1 is the legal unknown-record-count sentinel. It still requires
        # complete records, rather than MNE silently dropping a truncated tail.
        if payload % record_bytes or (
            records != -1 and payload != records * record_bytes
        ):
            raise ValueError(
                "EDF/BDF file size does not match complete declared records."
            )


def load_eeg_to_mne(path, *, preload=False, **reader_kwargs):
    """Read a native EEG recording without analysis preprocessing.

    Parameters
    ----------
    path : str or os.PathLike
        EEG recording file or EGI .mff directory. Formats include BrainVision,
        EDF/BDF, continuous EEGLAB, CNT, EGI, GDF, Nicolet, Nihon Kohden, Persyst,
        eXimia, FIF, BVRF and XDF. Keep linked files together.
    preload : bool or str, default=False
        Whether to load samples into memory, or an MNE memory-map filename.
        Deferred reading lets MNE-BIDS copy supported native files unchanged.
        BVRF, XDF and some embedded EEGLAB files are necessarily preloaded.
        BVRF and XDF do not support memory-map filenames.
    **reader_kwargs : dict
        Options passed to the selected reader, such as ``eog``, ``misc`` or
        ``uint16_codec``. Nicolet requires ``ch_type`` from the recording
        protocol. For XDF these are PyXDF options, including ``select_streams``.

    Returns
    -------
    raw : mne.io.BaseRaw
        Recording as decoded by the selected reader. XDF includes one EEG
        stream and annotations from selected marker streams.

    Raises
    ------
    ValueError
        If the filename extension is unsupported, EDF/BDF records are incomplete
        or discontinuous, or the MNE reader rejects the recording.

    Notes
    -----
    No filtering, referencing, channel selection or epoching is applied. MNE
    reader behaviour still applies: mixed-rate EDF/BDF channels are represented
    at one sampling rate. Discontinuous EDF+D/BDF+D is rejected because its gaps
    cannot be represented faithfully by the continuous reader route.
    Status-channel triggers remain in the signal; use ``mne.find_events`` with
    acquisition-appropriate settings when an explicit event table is needed.
    Native reading does not import existing BIDS sidecars or repair metadata.

    XDF requires unique channel labels and calibrated voltage channel types/units.
    Original timestamps and PyXDF-corrected timestamps must fit a continuous
    regular grid. ``timing_tolerance`` is an XDF-only option, default 0.25 sample
    periods, in [0, 0.5). Gaps, nonmonotonic timestamps and markers outside the
    EEG sample span are rejected. The effective rate must be within 1% of the
    nominal rate to allow bounded clock drift. PyXDF clock/dejitter options are
    retained; original timestamps are checked before dejittering can hide a gap.
    Signals are not interpolated. Other signal streams and per-sample timestamps
    are not retained. BVRF requires PyBVRF.
    """
    readers = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".ahdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".cnt": mne.io.read_raw_cnt,
        ".eeg": mne.io.read_raw_nihon,
        ".egi": mne.io.read_raw_egi,
        ".raw": mne.io.read_raw_egi,
        ".mff": mne.io.read_raw_egi,
        ".gdf": mne.io.read_raw_gdf,
        ".data": mne.io.read_raw_nicolet,
        ".lay": mne.io.read_raw_persyst,
        ".nxe": mne.io.read_raw_eximia,
        ".fif": mne.io.read_raw_fif,
        ".fif.gz": mne.io.read_raw_fif,
        ".bvrh": _read_bvrf,
        ".bvrd": _read_bvrf,
        ".bvrm": _read_bvrf,
        ".bvri": _read_bvrf,
        ".xdf": _read_xdf,
        ".xdfz": _read_xdf,
    }
    path = Path(path).expanduser()
    extension = (
        ".fif.gz" if path.name.lower().endswith(".fif.gz") else path.suffix.lower()
    )
    if extension in {".eeg", ".vmrk", ".amrk"}:
        header = _brainvision_header(path)
        if header is not None:
            path, extension = header, header.suffix.lower()
        elif extension != ".eeg":
            raise FileNotFoundError(f"No BrainVision header references {path.name}.")
        elif not _read_signature(path).startswith((b"EEG-", b"QI-", b"DAE-")):
            raise ValueError(
                f"Cannot identify {path.name}; supply its recording header."
            )
    elif extension == ".cnt":
        signature = _read_signature(path)
        if (signature[:4] == b"RIFF" and signature[8:12] == b"CNT ") or (
            signature[:4] == b"RF64" and signature[12:16] == b"CNT "
        ):
            readers[extension] = mne.io.read_raw_ant
        elif not signature.startswith(b"Version "):
            raise ValueError(f"Cannot identify the CNT vendor from {path.name}.")
    elif extension in {".egi", ".raw"}:
        if int.from_bytes(_read_signature(path)[:4], "big") not in {2, 4, 6}:
            raise ValueError(f"{path.name} is not a continuous EGI recording.")
    elif extension == ".data" and "ch_type" not in reader_kwargs:
        raise ValueError(
            "Nicolet .data files require ch_type from the recording protocol."
        )

    if extension not in readers:
        raise ValueError(
            f"Unsupported EEG extension {extension!r}; supply the recording entry file "
            "(for example .vhdr, .set or .lay) with its companions."
        )
    if extension in (".edf", ".bdf"):
        _check_edf_layout(path, 2 if extension == ".edf" else 3)
    # Reuse native decoders so calibration, companions and reader errors stay in MNE.
    return readers[extension](path, preload=preload, **reader_kwargs)


def _read_signature(path):
    """Read the bytes needed to distinguish formats sharing an extension."""
    with path.open("rb") as stream:
        return stream.read(16)


def _brainvision_header(path):
    """Find the sibling BrainVision header referencing a companion file."""
    key = "DataFile" if path.suffix.lower() == ".eeg" else "MarkerFile"
    matches = []
    for header in sorted(path.parent.iterdir()):
        if header.suffix.lower() not in {".vhdr", ".ahdr"} or not header.is_file():
            continue
        contents = header.read_bytes()
        codepage = re.search(rb"(?mi)^[ \t]*Codepage[ \t]*=[ \t]*([^\r\n]+)", contents)
        encoding = (
            codepage[1].decode("ascii", errors="replace").strip()
            if codepage
            else "utf-8-sig"
        )
        if encoding == "ANSI":
            encoding = "cp1252"
        try:
            text = contents.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # Match MNE's legacy fallback, which also avoids an unrelated
            # undecodable header breaking discovery of a readable companion.
            text = contents.decode("latin-1")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        common = re.search(
            r"(?ms)^[ \t]*\[Common [Ii]nfos\][ \t]*\n(.*?)(?=^[ \t]*\[|\Z)", text
        )
        if common is None:
            continue
        match = re.search(rf"(?mi)^[ \t]*{key}[ \t]*=[ \t]*(.+?)[ \t]*$", common[1])
        if match:
            linked = header.parent / match[1].replace("\\", "/")
            if linked.resolve() == path.resolve():
                matches.append(header)
    if len(matches) > 1:
        raise ValueError(
            "Multiple BrainVision headers reference this file; pass a header."
        )
    return matches[0] if matches else None


def _read_bvrf(path, *, preload=False, **reader_kwargs):
    """Read one in-memory BVRF recording with the optional PyBVRF reader."""
    if not isinstance(preload, bool):
        raise ValueError("BVRF always loads into memory; preload must be a bool.")
    if reader_kwargs.get("split"):
        raise ValueError("load_eeg_to_mne returns one Raw; split must be False.")

    import pybvrf

    return pybvrf.read_raw_bvrf(path, **reader_kwargs)


def load_auxiliary_info(path, dataset_name):
    """Read auxiliary JSON, warning and returning None if it cannot be loaded.

    Parameters
    ----------
    path : str
        Directory prefix, including its trailing separator.
    dataset_name : str
        Auxiliary filename without the .json extension.

    Returns
    -------
    aux_data : object or None
        Decoded JSON value, or None for file access or JSON decoding errors.
    """
    full_path = path + dataset_name + ".json"
    try:
        with open(full_path) as f:
            return json.load(f)
    except (OSError, ValueError) as error:
        # Keep optional metadata failures non-fatal without swallowing interrupts.
        warnings.warn(
            f"Unable to load auxiliary information at {full_path}: {error}",
            stacklevel=2,
        )
        return None


def load_brainvision_to_mne(path, *, preload=False):
    """Load a BrainVision recording while retaining its MNE metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .vhdr file, with its linked signal and marker files available.
    preload : bool or str, default=False
        Whether to load samples into memory. A string specifies an MNE memory-map
        file. The default leaves samples on disk for subsequent BIDS writing.

    Returns
    -------
    raw : mne.io.BaseRaw
        Recording with all channels, sampling information and annotations as
        read by MNE. No filtering, channel selection or epoching is applied.

    Notes
    -----
    Channel types follow MNE's BrainVision reader. Existing BIDS sidecars are
    not read; use MNE-BIDS when their additional metadata is needed.
    """
    return mne.io.read_raw_brainvision(path, preload=preload)


def load_brainvision_to_numpy(path, remove_non_EEG=True):
    """Load a BrainVision recording as a NumPy array.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .vhdr file, with its linked signal and marker files available.
    remove_non_EEG : bool, default=True
        Retain only channels identified as EEG by MNE when True.

    Returns
    -------
    numpy_data : np.ndarray
        Signal values with shape (n_channels, n_timepoints), without metadata.
    """
    # Preserve the array interface while allowing metadata-aware callers to use Raw.
    mne_data = load_brainvision_to_mne(path)
    if remove_non_EEG:
        mne_data = mne_data.pick_types(eeg=True)
    data = mne_data.load_data()
    return np.asarray(data.get_data())
