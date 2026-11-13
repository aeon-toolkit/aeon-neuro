"""Exercise pinned NEMAR acquisition independently of public network access."""

import hashlib
import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from aeon_neuro.datasets import fetch_nemar_dataset

RAW = "sub-021/eeg/sub-021_task-P3_eeg.set"
OTHER = "sub-026/eeg/sub-026_task-P3_eeg.edf"


@dataclass
class _File:
    path: str
    size: int
    sha256: str | None
    md5: str | None = None
    git_sha1: str | None = None
    url: str = "https://data.nemar.org/example"


@pytest.fixture
def provider(monkeypatch):
    """Emulate the public client seam and retain the exact source payloads."""
    content = {
        "dataset_description.json": b'{"Name":"Example","BIDSVersion":"1.10.0"}',
        "participants.tsv": b"participant_id\nsub-021\nsub-026\n",
        "participants.json": b"{}",
        "README": b"Original attribution",
        "task-P3_events.json": b'{"value": {"Description": "Source codebook"}}',
        "task-N400_events.json": b"{}",
        "sub-021/sub-021_scans.tsv": b"filename\n",
        "sub-021/eeg/sub-021_space-CapTrak_electrodes.tsv": b"name\tx\ty\tz\n",
        "sub-021/eeg/sub-021_space-CapTrak_coordsystem.json": b"{}",
        "sub-021/eeg/sub-021_task-P3_events.tsv": b"onset\tvalue\n0\t11\n",
        RAW: b"EEGLAB header",
        RAW.replace(".set", ".fdt"): b"binary signal",
        RAW.replace("P3", "N400"): b"different task",
        OTHER: b"different person",
        "derivatives/example/" + RAW: b"derivative",
        "stimuli/example.txt": b"stimulus",
    }
    files = {
        path: _File(path, len(data), hashlib.sha256(data).hexdigest())
        for path, data in content.items()
    }
    calls = []
    transfers = []
    manifest = SimpleNamespace(
        manifest_url="https://data.nemar.org/nm000132/v1.1.1/manifest.json"
    )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def fetch_index(self, dataset):
            calls.append(dataset)
            return SimpleNamespace(dataset_id=dataset, resolve_version=self.resolve)

        def resolve(self, version):
            calls.append(version)
            return SimpleNamespace(version=version)

        def fetch_manifest(self, index, release):
            return type(
                "Manifest",
                (),
                {
                    "manifest_url": manifest.manifest_url,
                    "__iter__": lambda self: iter(files.values()),
                },
            )()

    def download(file, destination):
        transfers.append(file.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content[file.path])
        return SimpleNamespace(value="ok")

    module = SimpleNamespace(
        NEMARClient=Client, download_one=download, __version__="0.3.3"
    )
    monkeypatch.setitem(sys.modules, "nemar", module)
    return SimpleNamespace(
        content=content, files=files, calls=calls, transfers=transfers, module=module
    )


def _fetch(tmp_path, **kwargs):
    options = dict(version="v1.1.1", subjects="021", task="P3", cache_dir=tmp_path)
    options.update(kwargs)
    return fetch_nemar_dataset("nm000132", **options)


def test_selection_receipt_and_offline_reuse(provider, tmp_path, monkeypatch):
    """Keep source companions/context while excluding unrelated recordings."""
    root, report = _fetch(tmp_path, return_report=True)
    expected = {
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
        "README",
        "task-P3_events.json",
        "sub-021/sub-021_scans.tsv",
        "sub-021/eeg/sub-021_space-CapTrak_electrodes.tsv",
        "sub-021/eeg/sub-021_space-CapTrak_coordsystem.json",
        "sub-021/eeg/sub-021_task-P3_events.tsv",
        RAW,
        RAW.replace(".set", ".fdt"),
    }
    assert set(report["files"]) == set(provider.transfers) == expected
    assert provider.calls == ["nm000132", "v1.1.1"]
    assert report["selected_recordings"] == [RAW]
    assert report["subjects"] == ["021"]
    for path in expected:
        assert (root / path).read_bytes() == provider.content[path]
    assert report["files"][RAW]["source_checksums"] == {
        "sha256": provider.files[RAW].sha256
    }
    report["files"][RAW]["size"] = 0
    monkeypatch.setitem(sys.modules, "nemar", None)
    reused, fresh = _fetch(
        tmp_path,
        subjects=["sub-021", "021"],
        task="task-P3",
        version="1.1.1",
        download=False,
        return_report=True,
    )
    assert reused == root
    assert fresh["files"][RAW]["size"] == len(provider.content[RAW])


@pytest.mark.parametrize(
    "name",
    [
        RAW.replace("sub-021_task", "sub-021-extra_task"),
        RAW.replace("P3", "P3-practice"),
        RAW.replace("P3", "P3.other"),
    ],
)
def test_malformed_recording_entities(provider, tmp_path, name):
    """Never truncate malformed entity labels into a requested identity."""
    data = provider.content.pop(RAW)
    del provider.files[RAW]
    provider.content[name] = data
    provider.files[name] = _File(name, len(data), hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="requested subject"):
        _fetch(tmp_path)
    assert provider.transfers == []
    with pytest.raises(ValueError, match="supported raw BIDS EEG paths"):
        _fetch(tmp_path, subjects=None, task=None, recordings=name)


def test_metadata_entities_and_root_documentation(provider, tmp_path):
    """Keep hyphenated documentation without inheriting partial task matches."""
    additions = {
        "README-v1.md": b"Versioned attribution",
        "task-P3-other_events.json": b"{}",
        "task-P3.other_events.json": b"{}",
        "task-P3_events.tsv.gz": b"compressed events",
        "sub-021/eeg/sub-021_task-P3_run-2_events.tsv": b"other run",
    }
    for path, data in additions.items():
        provider.content[path] = data
        provider.files[path] = _File(path, len(data), hashlib.sha256(data).hexdigest())
    root, report = _fetch(tmp_path, return_report=True)
    assert (root / "README-v1.md").read_bytes() == additions["README-v1.md"]
    assert "task-P3_events.tsv.gz" in report["files"]
    assert not (set(additions) - {"README-v1.md", "task-P3_events.tsv.gz"}) & set(
        report["files"]
    )
    assert _fetch(tmp_path, download=False) == root


def test_paths_and_requests_have_separate_cache_entries(provider, tmp_path):
    """Canonical ordering reuses a selection; new releases/requests cannot."""
    root = _fetch(tmp_path, subjects=None, task=None, recordings=[OTHER, RAW, RAW])
    assert _fetch(tmp_path, subjects=None, task=None, recordings=[RAW, OTHER]) == root
    assert provider.calls == ["nm000132", "v1.1.1"]
    for options in ({"version": "v1.1.2"}, {"task": None}, {"subjects": "026"}):
        with pytest.raises(FileNotFoundError):
            _fetch(tmp_path, download=False, **options)
    with pytest.raises(FileNotFoundError):
        _fetch(tmp_path, download=False)


@pytest.mark.parametrize(
    "options",
    [
        {"version": "latest"},
        {"version": None},
        {"version": "../v1.1.1"},
        {"subjects": None},
        {"subjects": []},
        {"subjects": [21]},
        {"subjects": "../021"},
        {"subjects": ""},
        {"recordings": RAW},
        {"subjects": None, "task": None, "recordings": []},
        {"subjects": None, "task": None, "recordings": "../" + RAW},
        {"subjects": None, "task": None, "recordings": "/" + RAW},
        {"subjects": None, "task": None, "recordings": RAW.replace("/", "\\")},
        {"subjects": None, "task": None, "recordings": "derivatives/pipeline/" + RAW},
        {"subjects": None, "recordings": RAW},
        {"task": ["P3"]},
        {"download": 0},
        {"return_report": 1},
    ],
)
def test_invalid_requests_never_contact_source(provider, tmp_path, options):
    """Reject ambiguous identity and unsafe paths before source access."""
    with pytest.raises((ValueError, TypeError)):
        _fetch(tmp_path, **options)
    assert provider.calls == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dataset", ["ds000132", "nm132", "../nm000132", 132])
def test_invalid_dataset(provider, tmp_path, dataset):
    """Require an exact supported NEMAR identifier."""
    with pytest.raises(ValueError, match="NEMAR ID"):
        fetch_nemar_dataset(
            dataset, version="v1.1.1", subjects="021", cache_dir=tmp_path
        )
    assert provider.calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"subjects": "21"},
        {"subjects": ["021", "999"]},
        {"task": "Unknown"},
        {"subjects": None, "task": None, "recordings": RAW.replace("021", "999")},
    ],
)
def test_missing_selection_does_not_download(provider, tmp_path, options):
    """A missing requested identity cannot silently yield a partial subset."""
    with pytest.raises(ValueError, match="requested"):
        _fetch(tmp_path, **options)
    assert provider.transfers == []


@pytest.mark.parametrize(
    "change",
    ["content", "delete", "extra", "receipt", "identity", "selection", "symlink"],
)
def test_changed_cache_never_redownloads(provider, tmp_path, change):
    """Check bytes, inventory, receipt identity and symbolic links on reuse."""
    root, report = _fetch(tmp_path, return_report=True)
    receipt = root.parent / "acquisition.json"
    if change == "content":
        (root / RAW).write_bytes(b"X" * len(provider.content[RAW]))
    elif change == "delete":
        (root / RAW).unlink()
    elif change == "extra":
        (root / "unexpected.edf").write_bytes(b"extra")
    elif change == "receipt":
        receipt.write_text("[]")
    elif change == "identity":
        report["version"] = "v9.0.0"
        receipt.write_text(json.dumps(report))
    elif change == "selection":
        report["selected_recordings"] = [OTHER]
        receipt.write_text(json.dumps(report))
    else:
        (root / RAW).unlink()
        try:
            (root / RAW).symlink_to(root / "README")
        except OSError:
            pytest.skip("Symbolic links are unavailable on this platform")
    before = list(provider.transfers)
    with pytest.raises(ValueError, match="Invalid NEMAR cache"):
        _fetch(tmp_path)
    assert provider.transfers == before


@pytest.mark.parametrize(
    "failure",
    [
        "hash_mismatch",
        "size_mismatch",
        "interrupt",
        "missing_checksum",
        "missing_size",
        "invalid_hash",
    ],
)
def test_failed_transfer_never_publishes(provider, tmp_path, failure):
    """Failed downloads and unusable manifest facts leave no completed entry."""
    original = provider.module.download_one

    def transfer(file, destination):
        if file.path != RAW:
            return original(file, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"partial")
        if failure == "interrupt":
            raise OSError("connection lost")
        return SimpleNamespace(value=failure)

    if failure == "missing_checksum":
        provider.files[RAW].sha256 = None
    elif failure == "missing_size":
        provider.files[RAW].size = None
    elif failure == "invalid_hash":
        provider.files[RAW].sha256 = "oops"
    else:
        provider.module.download_one = transfer
    with pytest.raises((ValueError, OSError)):
        _fetch(tmp_path)
    assert not list(tmp_path.rglob("complete.json"))
    assert not list(tmp_path.rglob(".fetch-*"))


def test_missing_cache_and_dependency(tmp_path, monkeypatch):
    """Offline misses and missing optional support are distinct failures."""
    monkeypatch.setitem(sys.modules, "nemar", None)
    with pytest.raises(FileNotFoundError, match="No cached NEMAR"):
        _fetch(tmp_path, download=False)
    with pytest.raises(ImportError, match=r"aeon-neuro\[nemar\]"):
        _fetch(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("section", ["Common Infos", "common infos"])
def test_brainvision_declared_companions(provider, tmp_path, section):
    """Resolve non-stem companion references and retain their exact bytes."""
    header = "sub-021/eeg/sub-021_task-P3_eeg.vhdr"
    content = {
        header: (
            b"Brain Vision Data Exchange Header File Version 1.0\n"
            + f"[{section}]\nDataFile=signal.eeg\nMarkerFile=markers.vmrk\n".encode()
            + b"[Comment]\nDataFile=comment.eeg\nMarkerFile=comment.vmrk\n"
        ),
        "sub-021/eeg/signal.eeg": b"signal bytes",
        "sub-021/eeg/markers.vmrk": b"markers",
    }
    for path, data in content.items():
        provider.content[path] = data
        provider.files[path] = _File(path, len(data), hashlib.sha256(data).hexdigest())
    root, report = _fetch(
        tmp_path, subjects=None, task=None, recordings=header, return_report=True
    )
    assert set(content).issubset(report["files"])
    assert RAW not in report["files"]
    assert (
        _fetch(tmp_path, subjects=None, task=None, recordings=header, download=False)
        == root
    )


def test_brainvision_companions_require_common_infos(provider, tmp_path):
    """Comment text cannot supply a missing companion declaration."""
    header = "sub-021/eeg/sub-021_task-P3_eeg.vhdr"
    data = b"[Common Infos]\nDataFile=signal.eeg\n[Comment]\nMarkerFile=markers.vmrk\n"
    content = {
        header: data,
        "sub-021/eeg/signal.eeg": b"signal",
        "sub-021/eeg/markers.vmrk": b"markers",
    }
    for path, payload in content.items():
        provider.content[path] = payload
        provider.files[path] = _File(
            path, len(payload), hashlib.sha256(payload).hexdigest()
        )
    with pytest.raises(ValueError, match="Missing BrainVision"):
        _fetch(tmp_path, subjects=None, task=None, recordings=header)
    assert not list(tmp_path.rglob("complete.json"))


@pytest.mark.parametrize(
    "reference", ["missing.eeg", "../outside.eeg", "/outside.eeg", "C:\\outside.eeg"]
)
def test_brainvision_missing_or_unsafe_companion(provider, tmp_path, reference):
    """Never publish a header whose declared dependency cannot be retained."""
    header = "sub-021/eeg/sub-021_task-P3_eeg.vhdr"
    data = f"[Common Infos]\nDataFile={reference}\nMarkerFile=markers.vmrk\n".encode()
    provider.content[header] = data
    provider.files[header] = _File(header, len(data), hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="companion|Unsafe"):
        _fetch(tmp_path, subjects=None, task=None, recordings=header)
    assert not list(tmp_path.rglob("complete.json"))


@pytest.mark.parametrize("algorithm", ["sha256", "md5", "git"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_real_client_verifies_manifest_transfers(
    tmp_path, monkeypatch, algorithm, corrupt
):
    """Run real nemar-py parsing and transfer verification over in-memory HTTP."""
    nemar = pytest.importorskip("nemar")
    import httpx

    content = {
        "dataset_description.json": b'{"BIDSVersion":"1.10.0"}',
        "sub-021/eeg/sub-021_task-P3_eeg.edf": b"signal bytes",
    }
    entries = []
    for path, data in content.items():
        hashed = f"blob {len(data)}\0".encode() + data if algorithm == "git" else data
        digest = hashlib.new(
            "sha1" if algorithm == "git" else algorithm, hashed
        ).hexdigest()
        entries.append(
            {
                "path": path,
                "size": len(data),
                "url": "https://data.nemar.org/bytes/" + path,
                "checksum_algorithm": algorithm,
                "checksum": digest,
            }
        )

    def respond(request):
        if request.url.path == "/nm000132/":
            return httpx.Response(
                200,
                json={
                    "dataset_id": "nm000132",
                    "versions": [
                        {
                            "version": "v1.1.1",
                            "manifest_url": "/nm000132/v1.1.1/manifest.json",
                        }
                    ],
                },
            )
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(200, json={"files": entries})
        data = content[request.url.path.removeprefix("/bytes/")]
        if corrupt and request.url.path.endswith(".edf"):
            data = b"X" * len(data)
        return httpx.Response(200, content=data)

    client_class = httpx.Client

    def client(**kwargs):
        return client_class(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "Client", client)
    if corrupt:
        with pytest.raises(ValueError, match="verification failed"):
            _fetch(tmp_path)
        assert not list(tmp_path.rglob("complete.json"))
    else:
        root, report = _fetch(tmp_path, return_report=True)
        assert report["downloader"] == f"nemar-py {nemar.__version__}"
        for path, data in content.items():
            assert (root / path).read_bytes() == data


def test_missing_source_description(provider, tmp_path):
    """Fail before transfer if required root metadata is not advertised."""
    del provider.files["dataset_description.json"]
    with pytest.raises(ValueError, match="dataset_description"):
        _fetch(tmp_path)
    assert provider.transfers == []


def test_source_identity_is_checked(provider, tmp_path):
    """Never accept a different dataset returned by a provider client."""
    original = provider.module.NEMARClient.fetch_index

    def index(self, dataset):
        result = original(self, dataset)
        result.dataset_id = "nm000133"
        return result

    provider.module.NEMARClient.fetch_index = index
    with pytest.raises(ValueError, match="different dataset or release"):
        _fetch(tmp_path)
    assert provider.transfers == []


def test_subject_folder_must_match_filename(provider, tmp_path):
    """Reject a conflicting BIDS identity in an explicit recording request."""
    with pytest.raises(ValueError, match="supported raw BIDS EEG paths"):
        _fetch(
            tmp_path,
            subjects=None,
            task=None,
            recordings="sub-026/eeg/sub-021_task-P3_eeg.set",
        )
    assert provider.calls == []


def test_case_colliding_manifest_is_rejected(provider, tmp_path):
    """Do not silently overwrite two source files on Windows/macOS caches."""
    original = provider.files[RAW]
    name = RAW.replace("P3", "p3")
    provider.files[name] = _File(name, original.size, original.sha256)
    with pytest.raises(ValueError, match="case-insensitive"):
        _fetch(tmp_path)
    assert provider.transfers == []
