"""Test pinned OpenNeuro selection and cache behavior without network access."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aeon_neuro.datasets import fetch_openneuro_dataset


def _fetch(*args, **kwargs):
    # Transport/cache behavior is independent of the EEG preparation checks.
    return fetch_openneuro_dataset(*args, prepare=False, **kwargs)


@pytest.fixture
def client(monkeypatch):
    """Provide a small download client with the public backend signature."""
    calls = []

    def download(*, dataset, tag, target_dir, include, source):
        calls.append((dataset, tag, include, source))
        (target_dir / "dataset_description.json").write_text(
            json.dumps({"Name": "Example", "BIDSVersion": "1.10.0"})
        )
        (target_dir / "task-rest_eeg.json").write_text('{"TaskName": "rest"}')
        (target_dir / "participants.tsv").write_text("participant_id\nsub-01\n")
        patterns = ["/sub-01", "/sub-02"] if include is None else include[1:]
        for pattern in patterns:
            folder = target_dir / pattern.lstrip("/") / "eeg"
            folder.mkdir(parents=True)
            (folder / "signal.edf").write_bytes(b"example signal")
        if include is None:
            phenotype = target_dir / "phenotype"
            phenotype.mkdir()
            (phenotype / "clinical.tsv").write_text("participant_id\nsub-01\n")

    monkeypatch.setitem(sys.modules, "openneuro", SimpleNamespace(download=download))
    return calls


def test_selection_and_offline_reuse(client, tmp_path, monkeypatch):
    """Preserve zeros, canonicalize selections and reuse without a client."""
    root = _fetch(
        "ds008082",
        subjects=["02", "sub-01", "01"],
        version="1.0.0",
        cache_dir=tmp_path,
    )
    assert client == [
        ("ds008082", "1.0.0", ["/[A-Za-z0-9]*.*", "/sub-01", "/sub-02"], "openneuro")
    ]
    assert (root / "task-rest_eeg.json").is_file()
    assert (root / "participants.tsv").is_file()
    assert (root / "sub-01/eeg/signal.edf").read_bytes() == b"example signal"
    monkeypatch.setitem(sys.modules, "openneuro", None)
    assert (
        _fetch(
            "https://openneuro.org/datasets/ds008082/versions/1.0.0",
            subjects=["01", "02"],
            cache_dir=tmp_path,
            download=False,
        )
        == root
    )
    assert (
        _fetch(
            "ds008082",
            subjects=["02", "01"],
            version="1.0.0",
            cache_dir=tmp_path,
        )
        == root
    )


def test_distinct_releases_and_selections(client, tmp_path):
    """Never satisfy a new release or selection with an earlier cached subset."""
    roots = {
        _fetch("ds008082", subjects=subjects, version=version, cache_dir=tmp_path)
        for version, subjects in [("1.0.0", "01"), ("1.0.1", "01"), ("1.0.0", "02")]
    }
    assert len(roots) == len(client) == 3


def test_whole_dataset_and_offline_reuse(client, tmp_path, monkeypatch):
    """Fetch all snapshot files without requiring a separate subject listing."""
    root = _fetch("ds008082", subjects="all", version="1.0.0", cache_dir=tmp_path)
    assert client == [("ds008082", "1.0.0", None, "openneuro")]
    assert (root / "sub-01/eeg/signal.edf").is_file()
    assert (root / "sub-02/eeg/signal.edf").is_file()
    assert (root / "phenotype/clinical.tsv").is_file()
    monkeypatch.setitem(sys.modules, "openneuro", None)
    assert (
        _fetch(
            "ds008082",
            subjects="all",
            version="1.0.0",
            cache_dir=tmp_path,
            download=False,
        )
        == root
    )


def test_whole_dataset_separate_from_subset(client, tmp_path):
    """A cached subject subset cannot satisfy an entire-snapshot request."""
    subset = _fetch("ds008082", subjects="01", version="1.0.0", cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        _fetch(
            "ds008082",
            subjects="all",
            version="1.0.0",
            cache_dir=tmp_path,
            download=False,
        )
    whole = _fetch("ds008082", subjects="all", version="1.0.0", cache_dir=tmp_path)
    assert subset != whole
    assert len(client) == 2


@pytest.mark.parametrize("change", ["edit", "delete", "extra", "manifest", "identity"])
def test_corrupt_cache_rejected(client, tmp_path, change):
    """Reject altered content and incomplete manifests without redownloading."""
    root = _fetch("ds006761", subjects="01", version="1.0.0", cache_dir=tmp_path)
    signal = root / "sub-01/eeg/signal.edf"
    manifest_path = root.parent / "complete.json"
    if change == "edit":
        # Same length, so size alone is insufficient.
        signal.write_bytes(b"changed signal")
    elif change == "delete":
        signal.unlink()
    elif change == "extra":
        (root / "extra.json").write_text("{}")
    elif change == "manifest":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] = "1.0.1"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid OpenNeuro cache"):
        _fetch("ds006761", subjects="01", version="1.0.0", cache_dir=tmp_path)
    assert len(client) == 1


def test_missing_cache_and_dependency(tmp_path, monkeypatch):
    """A cache-only miss does not need the optional client."""
    monkeypatch.setitem(sys.modules, "openneuro", None)
    kwargs = dict(subjects="01", version="1.0.0", cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No cached OpenNeuro"):
        _fetch("ds008082", download=False, **kwargs)
    with pytest.raises(ImportError, match=r"aeon-neuro\[openneuro\]"):
        _fetch("ds008082", **kwargs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "failure", ["interrupted", "missing_subject", "missing_metadata"]
)
def test_failed_download_not_cached(tmp_path, monkeypatch, failure):
    """Partial transfers and invalid layouts never become completed entries."""

    def download(**kwargs):
        root = kwargs["target_dir"]
        (root / "partial.edf").write_bytes(b"partial")
        if failure == "interrupted":
            raise RuntimeError("transfer interrupted")
        if failure == "missing_subject":
            (root / "dataset_description.json").write_text('{"BIDSVersion": "1.10.0"}')

    monkeypatch.setitem(sys.modules, "openneuro", SimpleNamespace(download=download))
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        _fetch("ds008082", subjects="01", version="1.0.0", cache_dir=tmp_path)
    assert not list(tmp_path.rglob("complete.json"))
    assert not list(tmp_path.rglob("partial.edf"))


@pytest.mark.parametrize(
    "dataset, version, subjects",
    [
        ("ds008082", None, "01"),
        ("ds008082", "latest", "01"),
        ("ds008082", "../1.0.0", "01"),
        ("ds008082", "1.0.0", []),
        ("ds008082", "1.0.0", [1]),
        ("ds008082", "1.0.0", "../01"),
        ("ds008082", "1.0.0", "*"),
        ("ds008082", "1.0.0", "sub-"),
        ("https://example.org/datasets/ds008082/versions/1.0.0", None, "01"),
        ("https://openneuro.org/datasets/ds008082/versions/1.0.0", "1.0.1", "01"),
        ("https://openneuro.org/datasets/ds008082/versions/1.0.0?q=x", None, "01"),
        ("https://openneuro.org/datasets/ds008082", None, "01"),
    ],
)
def test_invalid_inputs_do_not_download(client, tmp_path, dataset, version, subjects):
    """Reject unpinned or unsafe selection inputs before invoking the backend."""
    with pytest.raises(ValueError):
        _fetch(dataset, version=version, subjects=subjects, cache_dir=tmp_path)
    assert not client
    assert not list(tmp_path.iterdir())


def test_default_cache_location(client, tmp_path, monkeypatch):
    """Respect the existing aeon data directory convention."""
    monkeypatch.setenv("AEON_DATA", str(tmp_path))
    root = _fetch("ds008082", subjects="01", version="1.0.0")
    assert root.is_relative_to(tmp_path / "openneuro")


def test_symlink_cache_rejected(client, tmp_path):
    """Do not follow a replaced cached signal outside its selection."""
    root = _fetch("ds008082", subjects="01", version="1.0.0", cache_dir=tmp_path)
    signal = root / "sub-01/eeg/signal.edf"
    outside = tmp_path / "outside.edf"
    signal.replace(outside)
    try:
        signal.symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform")
    with pytest.raises(ValueError, match="symbolic links"):
        _fetch("ds008082", subjects="01", version="1.0.0", cache_dir=tmp_path)
    assert len(client) == 1


def test_concurrent_cache_completion(client, tmp_path, monkeypatch):
    """Validate a competing completed entry when the final rename loses a race."""
    rename = Path.rename

    def concurrent_rename(stage, destination):
        shutil.copytree(stage, destination)
        return rename(stage, destination)

    monkeypatch.setattr(Path, "rename", concurrent_rename)
    root = _fetch("ds008082", subjects="01", version="1.0.0", cache_dir=tmp_path)
    assert (root / "sub-01/eeg/signal.edf").is_file()
    assert len(client) == 1
    assert not list(tmp_path.rglob(".fetch-*"))
