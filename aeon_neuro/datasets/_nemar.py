"""Fetch explicit NEMAR releases and EEG selections into a verified cache."""

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

_EXTENSIONS = {".set", ".vhdr", ".edf", ".bdf"}


def fetch_nemar_dataset(
    dataset,
    *,
    version,
    subjects=None,
    task=None,
    recordings=None,
    cache_dir=None,
    download=True,
    return_report=False,
):
    """Fetch selected raw EEG recordings from an exact NEMAR release.

    Parameters
    ----------
    dataset : str
        NEMAR identifier such as ``"nm000132"`` or ``"on005505"``.
    version : str
        Explicit semantic release, such as ``"v1.1.1"``. The leading ``v``
        is optional. Moving aliases such as ``"latest"`` are rejected.
    subjects : str or iterable of str or None, default=None
        Explicit BIDS subject labels, with or without ``sub-``. Leading zeros
        are retained. Selects supported raw EEG recordings for every requested
        subject. Supply either subjects or recordings, not both.
    task : str or None, default=None
        Exact BIDS task label, optionally prefixed with ``task-``. Applies to
        subject selection only. Every requested subject must have a match.
    recordings : str or iterable of str or None, default=None
        Exact BIDS-relative primary filenames ending in ``_eeg.set``,
        ``_eeg.vhdr``, ``_eeg.edf`` or ``_eeg.bdf``. Companions and applicable
        ancestor metadata are included automatically. No catalogue is queried.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root; defaults to ``$AEON_DATA/nemar`` or ``~/aeon_data/nemar``.
        Releases and canonical selections have separate entries.
    download : bool, default=True
        Allow acquisition when absent. False requires a complete local entry
        and never imports nemar-py or accesses the network. Existing entries
        are always checked locally, without refreshing remote metadata.
    return_report : bool, default=False
        Return ``(root, report)``. The detached report records the requested
        selection, resolved recordings, manifest URL, downloader version,
        source checksums and local SHA-256 hashes. It contains no preparation
        or classification decisions.

    Returns
    -------
    pathlib.Path or tuple of pathlib.Path and dict
        Local BIDS root, optionally accompanied by its acquisition report.

    Raises
    ------
    ValueError
        Invalid identity/selection, missing recordings/companions, source
        verification failure or incomplete/changed cache.
    FileNotFoundError
        Missing cache with ``download=False``.
    ImportError
        Missing optional nemar-py when a transfer is needed.

    Notes
    -----
    Uses the public nemar-py client and its verified single-file transfers.
    Every selected file must have a declared size and supported checksum.
    EEGLAB same-stem FDT files and BrainVision header-declared companions are
    retained, along with compatible ancestor JSON/TSV, participant tables,
    geometry and root documentation. Other modalities, derivatives, stimuli
    and phenotype folders are outside this selected EEG copy. Participant
    tables are retained unchanged, including unselected rows.

    Metadata and signal bytes are retained as published; this is not a BIDS
    validation, readability or scientific-quality check. Checksum verification
    detects transfer/local changes, not independently authenticated provenance.
    Failed transfers are discarded before atomic publication. Treat the returned
    root as read-only and use a separate folder for conversion or preparation.
    """
    if not isinstance(dataset, str) or not re.fullmatch(r"(?:nm|on)[0-9]{6}", dataset):
        raise ValueError("dataset must be a NEMAR ID such as nm000132")
    if not isinstance(version, str) or not re.fullmatch(
        r"v?[0-9]+\.[0-9]+\.[0-9]+", version
    ):
        raise ValueError("An explicit NEMAR release such as v1.1.1 is required")
    version = "v" + version.removeprefix("v")
    if any(not isinstance(value, bool) for value in (download, return_report)):
        raise TypeError("download and return_report must be bools")
    if (subjects is None) == (recordings is None):
        raise ValueError("Supply exactly one of subjects or recordings")
    subjects = _labels(subjects, "sub") if subjects is not None else None
    if task is not None:
        if subjects is None or not isinstance(task, str):
            raise ValueError("task requires subject selection and a string label")
        task = _labels(task, "task")[0]
    recordings = _paths(recordings) if recordings is not None else None
    identity = dict(
        dataset=dataset,
        version=version,
        subjects=subjects,
        task=task,
        recordings=recordings,
    )
    selection = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    if cache_dir is None:
        cache_dir = Path(os.environ.get("AEON_DATA", "~/aeon_data")) / "nemar"
    entry = Path(cache_dir).expanduser() / dataset / version / selection
    if entry.exists() or entry.is_symlink():
        return _read_cache(entry, identity, return_report)
    if not download:
        raise FileNotFoundError(f"No cached NEMAR selection at {entry}")
    try:
        import nemar
    except ModuleNotFoundError as exc:
        if exc.name != "nemar":
            raise
        raise ImportError(
            "NEMAR downloads require 'pip install aeon-neuro[nemar]'."
        ) from exc

    with nemar.NEMARClient() as client:
        index = client.fetch_index(dataset)
        release = index.resolve_version(version)
        if index.dataset_id != dataset or release.version != version:
            raise ValueError("NEMAR returned a different dataset or release")
        manifest = client.fetch_manifest(index, release)
    files = {}
    for file in manifest:
        path = _relative(file.path)
        if path in files:
            raise ValueError(f"Duplicate manifest path: {path}")
        files[path] = file
    if len({path.casefold() for path in files}) != len(files):
        raise ValueError("Manifest paths collide on case-insensitive filesystems")
    chosen, needed = _select(files, subjects, task, recordings)
    entry.parent.mkdir(parents=True, exist_ok=True)
    # Transfers can leave partial files. Only a fully verified entry becomes
    # visible, and an interrupted attempt never replaces an existing selection.
    with TemporaryDirectory(prefix=".fetch-", dir=entry.parent) as temporary:
        stage = Path(temporary) / "entry"
        root = stage / "bids"
        root.mkdir(parents=True)
        transferred = {}

        def transfer(path):
            file = files[path]
            hashes = {
                key: getattr(file, key)
                for key in ("sha256", "md5", "git_sha1")
                if getattr(file, key) is not None
            }
            lengths = {"sha256": 64, "md5": 32, "git_sha1": 40}
            if (
                type(file.size) is not int
                or file.size < 0
                or not hashes
                or any(
                    not isinstance(value, str)
                    or not re.fullmatch(rf"[a-fA-F0-9]{{{lengths[key]}}}", value)
                    for key, value in hashes.items()
                )
            ):
                raise ValueError(f"Missing or invalid source size/checksum: {path}")
            result = nemar.download_one(file, root / path)
            if result.value != "ok":
                raise ValueError(
                    f"NEMAR verification failed for {path}: {result.value}"
                )
            transferred[path] = {"size": file.size, "source_checksums": hashes}

        for path in sorted(needed):
            transfer(path)
        # BrainVision permits companion names that differ from the header's
        # stem. Follow its explicit relative references, never guess replacements.
        for path in chosen:
            if path.endswith(".vhdr"):
                for companion in _brainvision_companions(root, path):
                    if companion not in files:
                        raise ValueError(f"Missing BrainVision companion: {companion}")
                    if companion not in transferred:
                        transfer(companion)
        inventory = _inventory(root)
        if set(inventory) != set(transferred) or any(
            fact["size"] != transferred[path]["size"]
            for path, fact in inventory.items()
        ):
            raise ValueError("Downloaded inventory differs from the source selection")
        description = json.loads(
            (root / "dataset_description.json").read_text("utf-8-sig")
        )
        if not isinstance(description, dict) or not description.get("BIDSVersion"):
            raise ValueError("Downloaded dataset_description.json lacks BIDSVersion")
        report = {
            "schema": 1,
            **identity,
            "selected_recordings": chosen,
            "manifest_url": manifest.manifest_url,
            "downloader": f"nemar-py {nemar.__version__}",
            "files": {
                path: {
                    **fact,
                    "source_checksums": transferred[path]["source_checksums"],
                }
                for path, fact in inventory.items()
            },
        }
        # Keep the report under the same integrity boundary as signal files;
        # an edited source/checksum receipt must not pass offline reuse.
        report_bytes = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
        (stage / "acquisition.json").write_bytes(report_bytes)
        complete = {
            "schema": 1,
            **identity,
            "files": inventory,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        }
        (stage / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
        try:
            stage.rename(entry)
        except OSError:
            # A concurrent caller may have published the same request. Verify
            # its winner rather than overwrite it or return our discarded copy.
            if not entry.exists():
                raise
    return _read_cache(entry, identity, return_report)


def _labels(values, prefix):
    if isinstance(values, str):
        values = [values]
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{prefix} labels must be strings") from exc
    if not values or any(
        not isinstance(value, str)
        or not re.fullmatch(rf"(?:{prefix}-)?[A-Za-z0-9]+", value)
        for value in values
    ):
        raise ValueError(f"{prefix} labels must be nonempty alphanumeric strings")
    return sorted({value.removeprefix(prefix + "-") for value in values})


def _relative(value):
    if (
        not isinstance(value, str)
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"Unsafe NEMAR path: {value!r}")
    return value


def _primary(value):
    path = PurePosixPath(value)
    parts = path.parts
    valid_layout = (
        len(parts) in (3, 4)
        and re.fullmatch(r"sub-[A-Za-z0-9]+", parts[0])
        and (len(parts) == 3 or re.fullmatch(r"ses-[A-Za-z0-9]+", parts[1]))
        and parts[-2] == "eeg"
        and path.suffix in _EXTENSIONS
        and path.stem.endswith("_eeg")
    )
    if not valid_layout:
        return False
    entities = _entities(path.name)
    # Subject/session filenames must agree with their containing folders;
    # filtering a conflicting path would otherwise misidentify its participant.
    return (
        entities is not None
        and entities.get("sub") == parts[0].removeprefix("sub-")
        and entities.get("ses")
        == (parts[1].removeprefix("ses-") if len(parts) == 4 else None)
        and "task" in entities
    )


def _paths(values):
    if isinstance(values, str):
        values = [values]
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError("recordings must contain BIDS-relative EEG paths") from exc
    if not values or any(not _primary(_relative(value)) for value in values):
        raise ValueError("recordings must contain supported raw BIDS EEG paths")
    return sorted(set(values))


def _entities(path):
    # Directory entities retain subject/session context for scans and geometry.
    path = PurePosixPath(path)
    stem = (
        path.name.removesuffix(".tsv.gz")
        if path.name.endswith(".tsv.gz")
        else path.stem
    )
    fields = {}
    for part in (*path.parts[:-1], stem):
        for token in part.split("_"):
            if "-" not in token:
                continue
            match = re.fullmatch(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", token)
            # A partial label must never become a different subject or task.
            if match is None:
                return None
            key, value = match.groups()
            if key in fields and fields[key] != value:
                return None
            fields[key] = value
    return fields


def _select(files, subjects, task, recordings):
    if recordings is not None:
        chosen = recordings
        if any(path not in files for path in chosen):
            raise ValueError("The release is missing a requested recording")
    else:
        chosen = sorted(
            path
            for path in files
            if _primary(path)
            and _entities(path).get("sub") in subjects
            and (task is None or _entities(path).get("task") == task)
        )
        if {_entities(path).get("sub") for path in chosen} != set(subjects):
            raise ValueError(
                "The release has no matching EEG recording for a requested subject"
            )
    needed = {"dataset_description.json", *chosen}
    # Parse metadata once and inspect only ancestor directories per recording.
    # Completed-cache validation uses this selection path too.
    metadata = defaultdict(list)
    for path in files:
        candidate = PurePosixPath(path)
        if candidate.suffix in {".json", ".tsv"} or candidate.name.endswith(".tsv.gz"):
            fields = _entities(path)
        elif candidate.suffix in {".txt", ".md", ".rst", ".yml", ".yaml"} or (
            candidate.parent == PurePosixPath(".") and not candidate.suffix
        ):
            # Documentation names are free text, not BIDS entity declarations.
            fields = _entities(candidate.parent.as_posix())
        else:
            continue
        if fields is not None:
            metadata[candidate.parent].append((path, fields))
    for name in chosen:
        raw = PurePosixPath(name)
        companion = raw.with_suffix(".fdt").as_posix()
        if raw.suffix == ".set" and companion in files:
            needed.add(companion)
        entities = _entities(name)
        for parent in raw.parents:
            for path, fields in metadata.get(parent, ()):
                # Geometry can declare a coordinate space absent from signal names.
                if all(
                    key == "space" or entities.get(key) == value
                    for key, value in fields.items()
                ):
                    needed.add(path)
    if any(path not in files for path in needed):
        raise ValueError("The release is missing dataset_description.json")
    return chosen, needed


def _brainvision_companions(root, name):
    text = (root / name).read_bytes().decode("utf-8-sig", errors="replace")
    values = {}
    in_common = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_common = line.casefold() == "[common infos]"
        elif in_common:
            # Free-form comments can contain assignment-like text; only the
            # Common Infos section declares the recording's dependencies.
            match = re.fullmatch(r"(DataFile|MarkerFile)\s*=\s*(.+)", line)
            if match:
                values[match[1]] = match[2].strip()
    if set(values) != {"DataFile", "MarkerFile"}:
        raise ValueError(f"Missing BrainVision DataFile/MarkerFile in {name}")
    parent = PurePosixPath(name).parent
    return [(parent / _relative(value)).as_posix() for value in values.values()]


def _inventory(root):
    result = {}
    if root.is_symlink() or not root.is_dir():
        raise ValueError("NEMAR cache root is missing or is a symbolic link")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("NEMAR cache cannot contain symbolic links")
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": digest,
            }
    return result


def _read_cache(entry, identity, return_report):
    try:
        if entry.is_symlink() or any(
            (entry / name).is_symlink()
            for name in ("complete.json", "acquisition.json")
        ):
            raise ValueError("Cache entry/report cannot be a symbolic link")
        complete = json.loads((entry / "complete.json").read_text("utf-8"))
        report_bytes = (entry / "acquisition.json").read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != complete["report_sha256"]:
            raise ValueError("Cached acquisition report changed")
        report = json.loads(report_bytes)
        if (
            complete["schema"] != 1
            or report["schema"] != 1
            or any(
                complete[key] != value or report[key] != value
                for key, value in identity.items()
            )
        ):
            raise ValueError("Cache identity differs from the requested selection")
        files = report["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("Cache has no verified files")
        expected = {
            _relative(path): {"size": fact["size"], "sha256": fact["sha256"]}
            for path, fact in files.items()
        }
        chosen, needed = _select(
            files, identity["subjects"], identity["task"], identity["recordings"]
        )
        if chosen != report["selected_recordings"] or not needed.issubset(files):
            raise ValueError("Cached recording selection is incomplete")
        if complete["files"] != expected or _inventory(entry / "bids") != expected:
            raise ValueError("Cached file inventory, sizes or SHA-256 hashes changed")
        for path in chosen:
            if path.endswith(".vhdr") and not set(
                _brainvision_companions(entry / "bids", path)
            ).issubset(files):
                raise ValueError("Cached BrainVision companion is missing")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"Invalid NEMAR cache at {entry}: {exc}. "
            "Use a fresh cache directory or remove this entry before retrying."
        ) from exc
    root = entry / "bids"
    return (root, report) if return_report else root
