"""Resolve EEG JSON inheritance while retaining the origin of effective fields."""

import json
from pathlib import Path

from mne_bids import get_bids_path_from_fname


def _read_eeg_json(bids_path):
    """Combine applicable ancestor JSON objects, rejecting same-level ambiguity."""
    if bids_path.root is None:
        raise ValueError("A BIDS root is required for EEG JSON inheritance.")
    root = Path(bids_path.root).resolve()
    directory = bids_path.fpath.parent.resolve()
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "The recording directory must be inside its BIDS root."
        ) from exc
    directories = [root]
    for part in relative.parts:
        directories.append(directories[-1] / part)

    metadata, origins, sources = {}, {}, []
    for directory in directories:
        candidates = []
        for path in sorted(directory.glob("*eeg.json")):
            if path.name.startswith(".") or not path.is_file():
                continue
            # The entity-free eeg.json spelling has no suffix in MNE's parser.
            if path.name == "eeg.json":
                entities = {}
            else:
                parsed = get_bids_path_from_fname(path.name, check=False)
                if parsed.suffix != "eeg":
                    continue
                entities = parsed.entities
            if all(
                value is None or bids_path.entities.get(key) == value
                for key, value in entities.items()
            ):
                candidates.append(path)
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple applicable EEG JSON files at one directory level: "
                f"{[path.name for path in candidates]}"
            )
        if not candidates:
            continue
        path = candidates[0]
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError(f"EEG JSON metadata must be an object: {path.name}")
        json.dumps(values, allow_nan=False)
        name = path.relative_to(root).as_posix()
        # Inheritance replaces top-level values; nested objects are not merged.
        # Visiting ancestors first also preserves keys absent from a later file.
        metadata.update(values)
        origins.update(dict.fromkeys(values, name))
        sources.append(name)
    if not sources:
        raise ValueError(f"No applicable EEG JSON metadata for {bids_path.basename}")
    return metadata, {"sources": sources, "fields": origins}
