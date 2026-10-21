"""Fetch pinned OpenNeuro subject folders into a verified local cache."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


def fetch_openneuro_dataset(
    dataset, *, subjects, version=None, cache_dir=None, download=True
):
    """Fetch selected subjects from a released OpenNeuro BIDS dataset.

    Parameters
    ----------
    dataset : str
        OpenNeuro ID such as ``ds008082``, or an HTTPS URL of the form
        ``https://openneuro.org/datasets/ds008082/versions/1.0.0``.
    subjects : str or iterable of str
        One or more subject labels, with or without ``sub-``. Labels retain
        leading zeros. All files under each selected subject are downloaded,
        along with root metadata files, including inherited JSON sidecars.
        Dataset-level folders such as ``phenotype`` and ``stimuli`` are excluded.
    version : str or None, default=None
        Explicit release, for example ``"1.0.0"``. Required for dataset IDs;
        inferred from a versioned URL. Conflicting versions are rejected.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. Defaults to ``$AEON_DATA/openneuro`` or
        ``~/aeon_data/openneuro``. Releases and subject selections have separate
        entries; overlapping selections can duplicate data.
    download : bool, default=True
        Allow a download if this selection is absent. False requires a complete
        cache entry and never imports the download client or accesses the network.

    Returns
    -------
    pathlib.Path
        Local BIDS root, suitable for constructing an MNE-BIDS ``BIDSPath``.
        This does not load signals, construct epochs or assign labels.

    Raises
    ------
    ValueError
        If the input is invalid or cached files are incomplete or modified.
    FileNotFoundError
        If the selection is absent and ``download=False``.
    ImportError
        If a download is needed and optional ``openneuro-py`` is unavailable.

    Notes
    -----
    Downloads use optional ``openneuro-py`` and its transfer verification.
    A SHA-256 manifest checks every local file on each subsequent call, including
    large signal files. These local hashes detect changes after download; they
    are not independently authenticated remote checksums. Completed selections
    are published together after download succeeds. Treat the returned folder as
    read-only and use a separate destination for conversion or other edits.
    """
    dataset, version = _parse_source(dataset, version)
    subjects = _parse_subjects(subjects)
    if not isinstance(download, bool):
        raise TypeError("download must be a bool")
    if cache_dir is None:
        cache_dir = Path(os.environ.get("AEON_DATA", "~/aeon_data")) / "openneuro"
    # Canonical selections share a cache even when their order or spelling differs.
    selection = hashlib.sha256(json.dumps(subjects).encode()).hexdigest()
    entry = Path(cache_dir).expanduser() / dataset / version / selection
    identity = {"dataset": dataset, "version": version, "subjects": subjects}
    if entry.exists():
        return _read_cache(entry, identity)
    if not download:
        raise FileNotFoundError(f"No cached OpenNeuro selection at {entry}")

    try:
        from openneuro import download as download_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "openneuro":
            raise
        raise ImportError(
            "OpenNeuro downloads require 'pip install aeon-neuro[openneuro]'."
        ) from exc

    entry.parent.mkdir(parents=True, exist_ok=True)
    # The client writes partial files while retrying. Keep those outside the
    # completed cache so interrupted downloads cannot appear ready for reuse.
    with TemporaryDirectory(prefix=".fetch-", dir=entry.parent) as temporary:
        stage = Path(temporary) / "entry"
        root = stage / "bids"
        root.mkdir(parents=True)
        download_dataset(
            dataset=dataset,
            tag=version,
            target_dir=root,
            # Root anchoring matters: bare globs match at every depth, and /*
            # also expands directory prefixes in the client. Requiring an
            # alphanumeric first character keeps root sidecars without matching
            # hidden .datalad directories or every sub-* folder in the dataset.
            include=["/[A-Za-z0-9]*.*", *[f"/sub-{subject}" for subject in subjects]],
            source="openneuro",
        )
        files = _inventory(root)
        _check_layout(root, subjects)
        manifest = {"schema": 1, **identity, "files": files}
        (stage / "complete.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            stage.rename(entry)
        except OSError:
            # Another process may have completed the same selection meanwhile.
            # Reuse it only if its completed contents pass the same checks.
            if not entry.exists():
                raise
            return _read_cache(entry, identity)
    return entry / "bids"


def _parse_source(dataset, version):
    if not isinstance(dataset, str):
        raise ValueError("dataset must be an OpenNeuro ID or versioned HTTPS URL")
    if dataset.startswith("https://"):
        url = urlsplit(dataset)
        match = re.fullmatch(
            r"/datasets/(ds[0-9]{6})/versions/([0-9]+\.[0-9]+\.[0-9]+)/?",
            url.path,
        )
        if url.netloc != "openneuro.org" or url.query or url.fragment or not match:
            raise ValueError("Use a versioned https://openneuro.org/datasets/ URL")
        dataset, url_version = match.groups()
        if version is not None and version != url_version:
            raise ValueError("version conflicts with the OpenNeuro URL")
        version = url_version
    if not re.fullmatch(r"ds[0-9]{6}", dataset):
        raise ValueError("dataset must be an OpenNeuro ID such as ds008082")
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", version
    ):
        raise ValueError("An explicit release version such as '1.0.0' is required")
    return dataset, version


def _parse_subjects(subjects):
    if isinstance(subjects, str):
        subjects = [subjects]
    try:
        subjects = list(subjects)
    except TypeError as exc:
        raise ValueError("subjects must contain one or more string labels") from exc
    if not subjects or any(
        not isinstance(subject, str)
        or not re.fullmatch(r"(?:sub-)?[A-Za-z0-9]+", subject)
        for subject in subjects
    ):
        raise ValueError("subjects must contain nonempty alphanumeric string labels")
    return sorted({subject.removeprefix("sub-") for subject in subjects})


def _check_layout(root, subjects):
    description = json.loads((root / "dataset_description.json").read_text("utf-8"))
    if not isinstance(description, dict) or not description.get("BIDSVersion"):
        raise ValueError("Downloaded dataset_description.json lacks BIDSVersion")
    for subject in subjects:
        folder = root / f"sub-{subject}"
        if not folder.is_dir() or not any(path.is_file() for path in folder.rglob("*")):
            raise ValueError(f"Downloaded selection is missing subject sub-{subject}")
    allowed = {f"sub-{subject}" for subject in subjects}
    if any(path.is_dir() and path.name not in allowed for path in root.iterdir()):
        raise ValueError("Downloaded selection contains unexpected directories")


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        # Cache entries contain actual downloaded files, never references outside
        # the selection. This also rejects a replaced file on later cache reads.
        if path.is_symlink():
            raise ValueError("OpenNeuro cache cannot contain symbolic links")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
    return files


def _read_cache(entry, identity):
    try:
        root = entry / "bids"
        if root.is_symlink() or entry.is_symlink():
            raise ValueError("Cache root cannot be a symbolic link")
        manifest = json.loads((entry / "complete.json").read_text("utf-8"))
        if manifest["schema"] != 1 or any(
            manifest[key] != value for key, value in identity.items()
        ):
            raise ValueError("Cache identity does not match the requested selection")
        files = _inventory(root)
        _check_layout(root, identity["subjects"])
        if files != manifest["files"]:
            raise ValueError("Cached file sizes or SHA-256 hashes have changed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid OpenNeuro cache at {entry}: {exc}. "
            "Use a fresh cache directory or remove this entry before retrying."
        ) from exc
    return root
