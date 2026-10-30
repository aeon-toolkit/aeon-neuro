"""Test preparation through real BIDS reads without network or export backends."""

import hashlib
import json
import shutil
import sys
from types import SimpleNamespace

import pytest
from mne_bids import get_bids_path_from_fname, read_raw_bids

from aeon_neuro.datasets import BIDSPreparationError, fetch_openneuro_dataset


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Create a tiny native BDF with published metadata and a fake transport."""
    root = tmp_path / "source"
    folder = root / "sub-01/eeg"
    folder.mkdir(parents=True)
    stem = "sub-01_task-rest"
    signal = folder / f"{stem}_eeg.bdf"

    def field(value, width):
        return str(value).encode("ascii").ljust(width)

    header = b"\xffBIOSEMI" + b" " * 160 + b"01.01.24" + b"00.00.00"
    header += field(1024, 8) + field("24BIT", 44)
    header += field(1, 8) + field(1, 8) + field(3, 4)
    for width, values in (
        (16, ["Cz", "ECG", "Status"]),
        (80, ["", "", ""]),
        (8, ["uV", "uV", "Boolean"]),
        (8, [-262144, -262144, -8388608]),
        (8, [262143, 262143, 8388607]),
        (8, [-8388608] * 3),
        (8, [8388607] * 3),
        (80, ["", "", ""]),
        (8, [100] * 3),
        (32, ["", "", ""]),
    ):
        header += b"".join(field(value, width) for value in values)
    signal.write_bytes(
        header
        + b"".join(value.to_bytes(3, "little", signed=True) for value in range(-50, 50))
        * 3
    )
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Example",
                "BIDSVersion": "1.10.0",
                "DatasetDOI": "10.1234/example",
            }
        )
    )
    metadata = folder / f"{stem}_eeg.json"
    metadata.write_text(
        json.dumps(
            {
                "TaskName": "rest",
                "SamplingFrequency": 100,
                "RecordingDuration": 8,
                "EEGChannelCount": 8,
                "ECGChannelCount": 1,
                "PowerLineFrequency": 50,
                "EEGReference": "unknown",
                "CustomAcquisitionField": "retained",
            }
        )
    )
    channels = folder / f"{stem}_channels.tsv"
    channels.write_text(
        "name\ttype\tunits\nCz\tEEG\tuV\nECG\tECG\tuV\nStatus\tTRIG\tarbitrary\n"
    )
    events = folder / f"{stem}_events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n0.2\t0.1\tstimulus\t99\t20\n"
    )
    (root / "participants.tsv").write_text("participant_id\tgroup\nsub-01\tcontrol\n")
    (root / "notes.txt").write_text("Unlisted source material is retained.\n")
    (root / "datacite.yml").write_text("title: Example\n")

    def download(**kwargs):
        shutil.copytree(root, kwargs["target_dir"], dirs_exist_ok=True)

    monkeypatch.setitem(sys.modules, "openneuro", SimpleNamespace(download=download))
    return root, signal, metadata, events, channels


def fetch(tmp_path, **kwargs):
    """Fetch the fixture into its own temporary cache."""
    return fetch_openneuro_dataset(
        "ds008082",
        subjects="01",
        version="1.0.0",
        cache_dir=tmp_path / "cache",
        return_report=True,
        **kwargs,
    )


def test_prepare_without_second_signal_copy(dataset, tmp_path, monkeypatch):
    """Correct derived metadata while retaining native bytes and source tables."""
    source, signal, metadata, events, channels = dataset
    root, report = fetch(tmp_path)
    assert (report["dataset"], report["version"], report["subjects"]) == (
        "ds008082",
        "1.0.0",
        ["01"],
    )
    relative = signal.relative_to(source)
    assert (root / relative).read_bytes() == signal.read_bytes()
    assert len(list((tmp_path / "cache").rglob("*.bdf"))) == 1
    assert (root / events.relative_to(source)).read_bytes() == events.read_bytes()
    assert (root / channels.relative_to(source)).read_bytes() == channels.read_bytes()
    assert (root / "notes.txt").read_bytes() == (source / "notes.txt").read_bytes()
    assert (root / "datacite.yml").read_bytes() == (
        source / "datacite.yml"
    ).read_bytes()
    assert (root / ".bidsignore").read_text() == "datacite.yml\n"
    prepared = json.loads((root / metadata.relative_to(source)).read_text())
    assert prepared["RecordingDuration"] == 1
    assert prepared["EEGChannelCount"] == 1
    assert prepared["CustomAcquisitionField"] == "retained"
    assert json.loads(metadata.read_text())["RecordingDuration"] == 8
    record = report["recordings"][0]
    assert record["original_metadata"]["RecordingDuration"] == 8
    assert record["events"] == 1
    assert record["samples"] == 100
    raw = read_raw_bids(get_bids_path_from_fname(root / relative), verbose=False)
    assert not raw.preload
    assert raw.get_channel_types() == ["eeg", "ecg", "stim"]
    raw.close()
    monkeypatch.setitem(sys.modules, "openneuro", None)
    cached, cached_report = fetch(tmp_path, download=False)
    assert cached == root and cached_report == report
    assert (
        report["downloaded_files"][relative.as_posix()]["sha256"]
        == hashlib.sha256(signal.read_bytes()).hexdigest()
    )
    description = json.loads((root / "dataset_description.json").read_text())
    assert "DatasetDOI" not in description
    assert description["SourceDatasets"] == [{"DOI": "10.1234/example"}]


def test_verified_corrections_are_separate_cache_decisions(dataset, tmp_path):
    """Retain evidence and never reuse a cache prepared with different choices."""
    source, signal, metadata, _, _ = dataset
    root, _ = fetch(tmp_path)
    corrections = {
        signal.relative_to(source).as_posix(): {
            "source": "Verified acquisition record, revision 2",
            "metadata": {"PowerLineFrequency": 60, "EEGReference": "common"},
            "channel_types": {"ECG": "EMG"},
        }
    }
    corrected, report = fetch(tmp_path, metadata_corrections=corrections)
    assert corrected != root
    values = json.loads((corrected / metadata.relative_to(source)).read_text())
    assert values["PowerLineFrequency"] == 60
    assert values["EEGReference"] == "common"
    assert values["EMGChannelCount"] == 1 and values["ECGChannelCount"] == 0
    assert any(
        item["source"] == "Verified acquisition record, revision 2"
        for item in report["recordings"][0]["corrections"]
    )


@pytest.mark.parametrize("problem", ["frequency", "events", "channels"])
def test_failed_preparation_has_report_and_no_completed_cache(
    dataset, tmp_path, problem
):
    """Report blocking conflicts without publishing partially corrected data."""
    _, _, metadata, events, channels = dataset
    if problem == "frequency":
        values = json.loads(metadata.read_text())
        values["SamplingFrequency"] = 200
        metadata.write_text(json.dumps(values))
    elif problem == "events":
        events.write_text("onset\tduration\tsample\n0.2\t0.1\t90\n")
    else:
        channels.write_text(channels.read_text().replace("Cz", "WrongName"))
    with pytest.raises(BIDSPreparationError) as caught:
        fetch(tmp_path)
    assert caught.value.report["summary"]["failed"] == 1
    assert caught.value.report["recordings"][0]["status"] == "failed"
    assert not list((tmp_path / "cache").rglob("complete.json"))
    assert not list((tmp_path / "cache").rglob("*.bdf"))


def test_discontinuities_and_missing_roles_reported(dataset, tmp_path):
    """Preserve unknown facts and discontinuity timing without guessing roles."""
    source, _, metadata, events, channels = dataset
    channels.unlink()
    events.write_text("onset\tduration\ttrial_type\n0.2\t0\tdiscontinuity\n")
    with pytest.warns(UserWarning, match="finding"):
        root, report = fetch(tmp_path)
    codes = {item["code"] for item in report["recordings"][0]["issues"]}
    assert {"discontinuities", "missing_channel_roles"} <= codes
    assert (
        json.loads((root / metadata.relative_to(source)).read_text())["EEGChannelCount"]
        == 8
    )
    assert (root / events.relative_to(source)).read_bytes() == events.read_bytes()


def test_report_integrity_checked(dataset, tmp_path):
    """A modified correction report invalidates the prepared cache."""
    root, _ = fetch(tmp_path)
    (root.parent / "preparation.json").write_text("{}")
    with pytest.raises(ValueError, match="preparation report has changed"):
        fetch(tmp_path, download=False)


@pytest.mark.parametrize("problem", ["declared_count", "partial_tail", "discontinuous"])
def test_native_record_layout_rejected_before_preparation(dataset, tmp_path, problem):
    """Do not cache native layouts whose continuous sample grid is unsupported."""
    _, signal, _, _, _ = dataset
    content = bytearray(signal.read_bytes())
    if problem == "declared_count":
        content[236:244] = b"2       "
    elif problem == "partial_tail":
        content += b"\x00"
    else:
        content[192:236] = b"BDF+D".ljust(44)
    signal.write_bytes(content)
    with pytest.raises(BIDSPreparationError) as caught:
        fetch(tmp_path)
    assert caught.value.report["summary"]["failed"] == 1
    assert not list((tmp_path / "cache").rglob("complete.json"))


@pytest.mark.parametrize("old_policy", [2, 3])
def test_older_preparation_policy_is_not_reused(dataset, tmp_path, old_policy):
    """Earlier layout/event/inheritance checks cannot satisfy the current policy."""
    root, _ = fetch(tmp_path)
    entry = root.parent
    manifest = json.loads((entry / "complete.json").read_text())
    manifest["preparation"]["version"] = old_policy
    key = {"subjects": ["01"], "preparation": manifest["preparation"]}
    old_entry = entry.with_name(
        hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    )
    entry.rename(old_entry)
    (old_entry / "complete.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="No cached OpenNeuro"):
        fetch(tmp_path, download=False)
    new_root, _ = fetch(tmp_path)
    assert new_root.parent != old_entry
    assert (old_entry / "complete.json").is_file()


@pytest.mark.parametrize(
    "onset,sample,alternative",
    [(0.2, "20", "99"), (0.2, "n/a", "99"), (0.2, "20", "20.5"), (0.205, "20", "21")],
)
def test_every_event_sample_column_is_validated(
    dataset, tmp_path, onset, sample, alternative
):
    """Neither a second sample column nor a missing first value bypasses checks."""
    _, _, _, events, _ = dataset
    events.write_text(
        f"onset\tduration\tsample\tonset_sample\n{onset}\t0\t{sample}\t{alternative}\n"
    )
    with pytest.raises(BIDSPreparationError) as caught:
        fetch(tmp_path)
    assert (
        "invalid timing or sample position"
        in caught.value.report["recordings"][0]["error"]
    )
    assert not list((tmp_path / "cache").rglob("complete.json"))


@pytest.mark.parametrize(
    "sample,alternative", [("20", "20"), ("n/a", "20"), ("20", "n/a")]
)
def test_agreeing_event_sample_columns_are_preserved(
    dataset, tmp_path, sample, alternative
):
    """Missing aliases and matching positions remain valid without table edits."""
    source, _, _, events, _ = dataset
    events.write_text(
        f"onset\tduration\tsample\tonset_sample\n0.204\t0\t{sample}\t{alternative}\n"
    )
    root, _ = fetch(tmp_path)
    assert (root / events.relative_to(source)).read_bytes() == events.read_bytes()


@pytest.mark.parametrize("root_only", [False, True])
def test_inherited_metadata_is_materialized_with_origins(dataset, tmp_path, root_only):
    """Expose inherited fields to downstream MNE-BIDS and retain their sources."""
    source, _, metadata, _, _ = dataset
    values = json.loads(metadata.read_text())
    inherited = {"Manufacturer": "Inherited vendor", "PowerLineFrequency": 50}
    if root_only:
        inherited.update(values)
        metadata.unlink()
    else:
        values.pop("PowerLineFrequency")
        metadata.write_text(json.dumps(values))
    ancestor = source / "task-rest_eeg.json"
    ancestor.write_text(json.dumps(inherited))
    before = ancestor.read_bytes()
    root, report = fetch(tmp_path)
    output = root / metadata.relative_to(source)
    assert json.loads(output.read_text())["Manufacturer"] == "Inherited vendor"
    assert json.loads(output.read_text())["PowerLineFrequency"] == 50
    raw = read_raw_bids(
        get_bids_path_from_fname(output.with_suffix(".bdf")), verbose=False
    )
    assert raw.info["line_freq"] == 50
    raw.close()
    record = report["recordings"][0]
    assert record["metadata_origins"]["PowerLineFrequency"] == "task-rest_eeg.json"
    assert record["metadata_sources"][0] == "task-rest_eeg.json"
    assert record["actions"][0]["path"] == metadata.relative_to(source).as_posix()
    assert not any(
        issue["code"] == "unknown_mains_frequency" for issue in record["issues"]
    )
    assert ancestor.read_bytes() == before
    assert (root / ancestor.name).read_bytes() == before


def test_ambiguous_inherited_metadata_is_not_cached(dataset, tmp_path):
    """Ambiguous source selection cannot become a completed prepared entry."""
    source, _, _, _, _ = dataset
    for name in ["eeg.json", "task-rest_eeg.json"]:
        (source / name).write_text('{"PowerLineFrequency": 50}')
    with pytest.raises(BIDSPreparationError) as caught:
        fetch(tmp_path)
    assert "Multiple applicable" in caught.value.report["recordings"][0]["error"]
    assert not list((tmp_path / "cache").rglob("complete.json"))


@pytest.mark.parametrize(
    "correction",
    [
        {"metadata": {"PowerLineFrequency": 50}},
        {"source": "notes", "metadata": {"SamplingFrequency": 200}},
        {"source": "notes", "metadata": {"PowerLineFrequency": True}},
        {"source": "notes", "channel_types": {"Cz": "invented"}},
    ],
)
def test_invalid_corrections_fail_before_download(tmp_path, monkeypatch, correction):
    """Require explicit evidence and reject attempts to overwrite signal timing."""

    def download(**kwargs):
        pytest.fail("Invalid correction triggered a download")

    monkeypatch.setitem(sys.modules, "openneuro", SimpleNamespace(download=download))
    with pytest.raises(ValueError):
        fetch(
            tmp_path,
            metadata_corrections={"sub-01/eeg/sub-01_task-rest_eeg.bdf": correction},
        )
