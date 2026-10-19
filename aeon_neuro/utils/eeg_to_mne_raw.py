"""Load EEG recordings with automatic reader selection."""

import re
from pathlib import Path

import mne


def load_eeg_to_mne(path, *, preload=False, verbose=None, **reader_kwargs):
    """Load an EEG recording into MNE Raw using its format's reader.

    Parameters
    ----------
    path : str or pathlib.Path
        Recording file or EGI .mff directory. Keep companion files together.
    preload : bool or str, default=False
        Load samples into memory or a memory-map file. BVRF always uses memory.
    verbose : bool, str, int or None, default=None
        MNE logging verbosity.
    **reader_kwargs
        Options for the selected reader, including ch_type for Nicolet.

    Returns
    -------
    raw : mne.io.BaseRaw
        Recording as returned by the reader, without extra preprocessing.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"EEG input does not exist: {path}")

    suffix = path.suffix.lower()
    if path.is_dir() and suffix != ".mff":
        raise ValueError("Pass a recording file or an EGI .mff directory.")

    reader = mne.io.read_raw
    if suffix in {".eeg", ".vmrk", ".amrk"}:
        header = _brainvision_header(path)
        if header is not None:
            path = header
            reader = mne.io.read_raw_brainvision
        elif suffix != ".eeg":
            raise FileNotFoundError(
                f"No BrainVision header references {path.name}. "
                "Keep the .vhdr/.ahdr header with its companion files."
            )
        elif _signature(path).startswith((b"EEG-", b"QI-", b"DAE-")):
            reader = mne.io.read_raw_nihon
        else:
            raise ValueError(
                f"Cannot identify {path.name}: .eeg is used by several formats. "
                "For BrainVision, supply its linked .vhdr header."
            )

    elif suffix == ".cnt":
        signature = _signature(path)
        is_ant = (signature[:4] == b"RIFF" and signature[8:12] == b"CNT ") or (
            signature[:4] == b"RF64" and signature[12:16] == b"CNT "
        )
        if not is_ant and not signature.startswith(b"Version "):
            raise ValueError(
                f"Cannot identify the CNT vendor from {path.name}'s header."
            )
        reader = mne.io.read_raw_ant if is_ant else mne.io.read_raw_cnt

    elif suffix in {".egi", ".raw"}:
        if int.from_bytes(_signature(path)[:4], "big") not in {2, 4, 6}:
            raise ValueError(
                f"{path.name} is not a supported continuous EGI binary recording."
            )
        reader = mne.io.read_raw_egi

    elif suffix in {".bvrh", ".bvrd", ".bvrm", ".bvri"}:
        return _read_bvrf(path, preload=preload, verbose=verbose, **reader_kwargs)

    elif suffix == ".data" and "ch_type" not in reader_kwargs:
        raise ValueError(
            "Nicolet .data files require ch_type from the recording protocol; "
            "the header cannot distinguish scalp EEG from intracranial channels."
        )

    elif suffix in {".fdt", ".dat", ".head"}:
        raise ValueError(
            f"{suffix} alone does not identify a supported EEG recording. "
            "Supply the EEGLAB .set, Persyst .lay, or Nicolet .data entry file."
        )

    return reader(path, preload=preload, verbose=verbose, **reader_kwargs)


def _signature(path):
    """Read the first 16 bytes for format detection."""
    with path.open("rb") as stream:
        return stream.read(16)


def _brainvision_header(path):
    """Find the header that explicitly references a companion file."""
    key = "DataFile" if path.suffix.lower() == ".eeg" else "MarkerFile"
    matches = []
    for header in sorted(path.parent.iterdir()):
        if header.suffix.lower() not in {".vhdr", ".ahdr"} or not header.is_file():
            continue
        contents = header.read_bytes()
        try:
            text = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = contents.decode("cp1252", errors="replace")
        if "[Common Infos]" not in text:
            continue
        common = text.split("[Common Infos]", 1)[1].split("\n[", 1)[0]
        match = re.search(rf"(?mi)^\s*{key}\s*=\s*(.+?)\s*$", common)
        if match:
            linked = header.parent / match[1].replace("\\", "/")
            if linked.resolve() == path.resolve():
                matches.append(header)
    if len(matches) > 1:
        names = ", ".join(header.name for header in matches)
        raise ValueError(
            f"Multiple BrainVision headers reference {path.name}: {names}. "
            "Pass the intended header explicitly."
        )
    return matches[0] if matches else None


def _read_bvrf(path, *, preload, verbose, **reader_kwargs):
    """Load BVRF as a single in-memory Raw."""
    if not isinstance(preload, bool):
        raise ValueError("BVRF always loads into memory; preload must be a bool.")
    if reader_kwargs.get("split"):
        raise ValueError("load_eeg_to_mne returns one Raw; split must be False.")

    try:
        import pybvrf
    except ModuleNotFoundError as error:
        if error.name != "pybvrf":
            raise
        raise ImportError("Reading BVRF requires pybvrf[mne] (Python >= 3.12).") from error
    return pybvrf.read_raw_bvrf(path, verbose=verbose, **reader_kwargs)
