"""Exercise source inspection and selected reading with tiny generated BDF files."""

import hashlib
import json
from copy import deepcopy

import edfio
import mne
import numpy as np
import pytest

from aeon_neuro.datasets import (
    EEGDatasetInfo,
    inspect_bids_eeg,
    open_bids_eeg_recording,
)
from aeon_neuro.datasets import _bids_input as adapter


@pytest.fixture
def bids(tmp_path, request):
    """Use inherited EEG metadata, explicit subjects and distinguishable samples."""
    (tmp_path / "dataset_description.json").write_text(
        '{"Name":"Generated EEG","BIDSVersion":"1.9.0"}'
    )
    (tmp_path / "task-imagery_eeg.json").write_text(
        '{"SamplingFrequency":32,"RecordingDuration":8}'
    )
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tcohort\nsub-01\tcontrol\nsub-02\tpatient\n"
    )
    extension = getattr(request, "param", "bdf")
    signal_type, writer = (
        (edfio.BdfSignal, edfio.Bdf)
        if extension == "bdf"
        else (edfio.EdfSignal, edfio.Edf)
    )
    paths = []
    for subject in ("01", "02"):
        stem = tmp_path / f"sub-{subject}/eeg/sub-{subject}_task-imagery"
        stem.parent.mkdir(parents=True)
        path = stem.with_name(stem.name + f"_eeg.{extension}")
        values = np.random.default_rng(int(subject)).normal(size=(2, 256))
        writer(
            [
                signal_type(row, 32, label=name, physical_dimension="uV")
                for row, name in zip(values, ("C3", "C4"))
            ]
        ).write(path)
        stem.with_name(stem.name + "_channels.tsv").write_text(
            "name\ttype\tunits\nC3\tEEG\tuV\nC4\tEEG\tuV\n"
        )
        # Keep BOM reader coverage independent of the host's default encoding.
        stem.with_name(stem.name + "_events.tsv").write_text(
            "\ufeffonset\tduration\ttrial_type\tsample\n"
            "1\t2\tleft\t32\n4\t2\tright\t128\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tmp_path, paths


def test_inspection_and_metadata_view_never_read_or_hash_signals(bids, monkeypatch):
    """Headers/tables suffice and the saved catalogue survives unavailable signals."""
    root, paths = bids
    original_hash = adapter._hash

    def metadata_hash(path):
        assert path.suffix != ".bdf", "Signal hash during inspection"
        return original_hash(path)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Sample decoding before explicit materialization")

    monkeypatch.setattr(adapter, "_hash", metadata_hash)
    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected_read)
    report = inspect_bids_eeg(root)
    assert report["ready"] and len(report["recordings"]) == 2
    first = report["recordings"][0]
    assert first["participant.cohort"] == "control"
    assert (
        first["metadata_origins"]["fields"]["SamplingFrequency"]
        == "task-imagery_eeg.json"
    )
    assert first["session"] is None
    assert report["events"][0]["sample"] == "32"
    assert all(issue["severity"] == "warning" for issue in report["issues"])
    for path in paths:
        path.unlink()
    saved = json.loads(json.dumps(report))
    info = EEGDatasetInfo(saved)
    assert len(info) == 2 and len(info.events) == 4
    assert info.to_dict() == saved


def test_selection_reads_only_chosen_source_and_exact_samples(bids, monkeypatch):
    """Opening a selected source preserves samples and owns only its handle."""
    root, paths = bids
    report = inspect_bids_eeg(root)
    selected = report["recordings"][-1]
    selected["expected_sha256"] = adapter._hash(paths[-1])
    opened = []
    original = adapter.load_eeg_to_mne

    def watched(path, **kwargs):
        opened.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(adapter, "load_eeg_to_mne", watched)
    with open_bids_eeg_recording(root, selected) as raw:
        values = raw.get_data(picks=["C4", "C3"], start=144, stop=176)
    assert opened == [paths[-1]]
    with original(paths[-1], preload=False, verbose=False) as raw:
        np.testing.assert_array_equal(
            values, raw.get_data(picks=["C4", "C3"], start=144, stop=176)
        )
    subset = inspect_bids_eeg(root, subjects=["01"])
    assert subset["source_ids"] == [paths[0].relative_to(root).as_posix()]


@pytest.mark.parametrize(
    "fault",
    ["channels", "rate", "sample_alias", "ambiguous", "sidecar_tie", "unsupported"],
)
def test_blockers_return_diagnostics_without_partial_catalogue(bids, fault):
    """A broken selected recording cannot disappear into an apparently valid subset."""
    root, paths = bids
    stem = paths[-1].name.removesuffix("_eeg.bdf")
    if fault == "channels":
        paths[-1].with_name(stem + "_channels.tsv").write_text("name\nC4\nC3\n")
    elif fault == "rate":
        paths[-1].with_suffix(".json").write_text('{"SamplingFrequency":64}')
    elif fault == "sample_alias":
        paths[-1].with_name(stem + "_events.tsv").write_text(
            "onset\tduration\tsample\tonset_sample\n1\t2\t32\t33\n"
        )
    elif fault == "ambiguous":
        (root / "eeg.json").write_text('{"SamplingFrequency":32}')
    elif fault == "sidecar_tie":
        paths[-1].with_name(stem + "_channels.tsv").unlink()
        paths[-1].rename(
            paths[-1].with_name("sub-02_task-imagery_acq-a_run-01_eeg.bdf")
        )
        for name in [
            "sub-02_task-imagery_acq-a_channels.tsv",
            "sub-02_task-imagery_run-01_channels.tsv",
        ]:
            paths[-1].with_name(name).write_text("name\ttype\nC3\tEEG\nC4\tEEG\n")
    else:
        paths[-1].rename(paths[-1].with_suffix(".vhdr"))
    report = inspect_bids_eeg(root)
    assert not report["ready"]
    assert report["recordings"] is None and report["events"] is None
    assert len(report["source_ids"]) == 2
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert errors and errors[0]["message"]


def test_optional_tables_remain_unknown_instead_of_inferred(bids):
    """Recording targets can still use known headers when no event table exists."""
    root, paths = bids
    for path in paths:
        path.with_name(path.name.removesuffix("_eeg.bdf") + "_events.tsv").unlink()
    (root / "participants.tsv").unlink()
    report = inspect_bids_eeg(root)
    assert report["ready"] and report["events"] == []
    assert {issue["code"] for issue in report["issues"]} >= {
        "events_missing",
        "participants_missing",
        "quality_unreviewed",
    }
    assert "participant.cohort" not in report["recordings"][0]


@pytest.mark.parametrize("changed", ["signal", "metadata", "checksum", "version"])
def test_stale_or_untrusted_sources_fail_before_decode(bids, monkeypatch, changed):
    """Saved file snapshots and optional content checks precede the reader."""
    root, paths = bids
    descriptor = deepcopy(inspect_bids_eeg(root)["recordings"][0])
    if changed == "signal":
        with paths[0].open("ab") as stream:
            stream.write(b"x")
    elif changed == "metadata":
        (root / "participants.tsv").write_text(
            "participant_id\tcohort\nsub-01\tchanged\n"
        )
    elif changed == "checksum":
        descriptor["expected_sha256"] = hashlib.sha256(b"wrong source").hexdigest()
    else:
        descriptor["bids_source"]["version"] = 2

    def unexpected(*args, **kwargs):
        raise AssertionError("Stale source reached the decoder")

    monkeypatch.setattr(adapter, "load_eeg_to_mne", unexpected)
    with pytest.raises(ValueError):
        open_bids_eeg_recording(root, descriptor)


def test_no_matching_sources_is_an_explicit_blocker(bids):
    """A wrong selection is distinguishable from an empty valid dataset."""
    report = inspect_bids_eeg(bids[0], subjects=["99"])
    assert not report["ready"] and report["source_ids"] == []
    assert report["issues"][0]["code"] == "selection_failed"


def test_partial_filter_mismatch_blocks_before_header_inspection(bids, monkeypatch):
    """The adapter shares the existing loader's strict selection boundary."""

    def unexpected(*args, **kwargs):
        raise AssertionError("Partial selection reached a source header")

    monkeypatch.setattr(adapter, "load_eeg_to_mne", unexpected)
    report = inspect_bids_eeg(bids[0], subjects=["01", "typo"])
    assert not report["ready"] and report["recordings"] is None
    assert "typo" in report["issues"][0]["message"]


@pytest.mark.parametrize("bids", ["edf"], indirect=True)
def test_edf_sources_use_the_same_metadata_and_reader_boundary(bids):
    """Both advertised single-file encodings use the shared source opener."""
    root, _ = bids
    report = inspect_bids_eeg(root, subjects=["01"])
    assert report["ready"] and report["source_ids"][0].endswith(".edf")
    with open_bids_eeg_recording(root, report["recordings"][0]) as raw:
        assert raw.get_data(start=48, stop=80).shape == (2, 32)


def test_changed_header_closes_the_newly_owned_handle(bids, monkeypatch):
    """A descriptor/header mismatch must not leak its source handle."""
    root, paths = bids
    descriptor = inspect_bids_eeg(root)["recordings"][0]
    descriptor["n_samples"] += 1
    raw = adapter.load_eeg_to_mne(paths[0], preload=False, verbose=False)
    closed = []
    close = raw.close

    def watched_close():
        closed.append(True)
        close()

    monkeypatch.setattr(raw, "close", watched_close)
    monkeypatch.setattr(adapter, "load_eeg_to_mne", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="header differs"):
        open_bids_eeg_recording(root, descriptor)
    assert closed == [True]


def test_dataset_sidecars_retain_complete_population_and_selected_scope(bids):
    """Selecting signals must not rewrite the population described by the source."""
    root, paths = bids
    (root / "README.md").write_text(
        "# Generated study\nCaf\u00e9 metadata.\n", encoding="utf-8"
    )
    dictionary = {"cohort": {"Levels": {"control": "Comparison group"}}}
    (root / "participants.json").write_text(json.dumps(dictionary), encoding="utf-8")
    report = inspect_bids_eeg(root, subjects=["02"])
    assert report["version"] == 2 and report["ready"]
    dataset = report["dataset"]
    assert dataset["description"] == json.loads(
        (root / "dataset_description.json").read_text()
    )
    assert dataset["readme"] == "# Generated study\nCaf\u00e9 metadata.\n"
    assert dataset["participants"] == {
        "columns": ["participant_id", "cohort"],
        "rows": [
            {"participant_id": "sub-01", "cohort": "control"},
            {"participant_id": "sub-02", "cohort": "patient"},
        ],
    }
    assert dataset["participants_json"] == dictionary
    assert dataset["selected_participants"] == ["sub-02"]
    assert dataset["paths"]["readme"] == "README.md"
    descriptor = report["recordings"][0]
    assert descriptor["filepath"] == paths[1].relative_to(root).as_posix()
    assert descriptor["participant.cohort"] == "patient"
    for name in [
        "dataset_description.json",
        "README.md",
        "participants.tsv",
        "participants.json",
    ]:
        assert (
            descriptor["bids_source"]["files"][name]["sha256"]
            == hashlib.sha256((root / name).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("column", [None, "sample", "onset_sample"])
def test_normalized_event_samples_preserve_original_columns(bids, column):
    """Export exact zero-based integers even if the source omitted sample indices."""
    root, paths = bids
    event_path = paths[0].with_name("sub-01_task-imagery_events.tsv")
    header = "onset\tduration\ttrial_type" + (f"\t{column}" if column else "")
    values = "1\tn/a\tleft" + ("\t32" if column else "")
    event_path.write_text(header + "\n" + values + "\n", encoding="utf-8")
    before = event_path.read_bytes()
    report = inspect_bids_eeg(root, subjects=["01"])
    assert report["ready"]
    row = report["events"][0]
    assert type(row["bids_onset_sample"]) is int
    assert row["bids_onset_sample"] == 32
    assert row["onset"] == "1" and row["duration"] == "n/a"
    if column:
        assert row[column] == "32"
    assert event_path.read_bytes() == before


@pytest.mark.parametrize(
    "fault", ["json", "duplicate_participant", "readmes", "reserved_sample"]
)
def test_invalid_dataset_metadata_and_reserved_sample_block_inspection(bids, fault):
    """Never choose an attribution file or overwrite a source column silently."""
    root, paths = bids
    if fault == "json":
        (root / "participants.json").write_text("[]", encoding="utf-8")
    elif fault == "duplicate_participant":
        with (root / "participants.tsv").open("a", encoding="utf-8") as stream:
            stream.write("sub-02\tother\n")
    elif fault == "readmes":
        (root / "README").write_text("one", encoding="utf-8")
        (root / "README.md").write_text("two", encoding="utf-8")
    else:
        paths[0].with_name("sub-01_task-imagery_events.tsv").write_text(
            "onset\tduration\tbids_onset_sample\n1\t2\t32\n", encoding="utf-8"
        )
    report = inspect_bids_eeg(root, subjects=["01"])
    assert not report["ready"]
    assert report["recordings"] is None and report["events"] is None
    assert report["source_ids"] == [paths[0].relative_to(root).as_posix()]


@pytest.fixture
def sidecars(bids):
    """Place geometry at subject scope and acquisition facts in scans.tsv."""
    root, paths = bids
    (root / "README.md").write_text("Generated study\n", encoding="utf-8")
    (root / "participants.json").write_text(
        '{"cohort":{"Description":"Group"}}', encoding="utf-8"
    )
    for path in paths:
        subject_dir = path.parent.parent
        subject = subject_dir.name
        (subject_dir / f"{subject}_electrodes.tsv").write_text(
            "name\tx\ty\tz\nC4\tn/a\t2\t3\nC3\t4\t5\t6\nREF\t7\t8\t9\n",
            encoding="utf-8",
        )
        (subject_dir / f"{subject}_coordsystem.json").write_text(
            '{"EEGCoordinateSystem":"Other","EEGCoordinateUnits":"mm",'
            '"EEGCoordinateSystemDescription":"Source frame"}',
            encoding="utf-8",
        )
        (subject_dir / f"{subject}_scans.tsv").write_text(
            "filename\tacq_time\tquality\n"
            "eeg/other_eeg.bdf\tn/a\tother\n"
            f"eeg/{path.name}\t2020-01-01T12:00:00\tn/a\n",
            encoding="utf-8",
        )
    return root, paths


def test_recording_sidecars_preserve_source_geometry_and_scan_identity(
    sidecars, monkeypatch
):
    """Geometry rows are source facts, not a channel permutation or a new montage."""
    root, paths = sidecars

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Sample read during sidecar inspection")

    monkeypatch.setattr(mne.io.BaseRaw, "get_data", unexpected_read)
    original_hash = adapter._hash

    def metadata_hash(path):
        assert path.suffix not in (".bdf", ".edf")
        return original_hash(path)

    monkeypatch.setattr(adapter, "_hash", metadata_hash)
    report = inspect_bids_eeg(root)
    assert report["ready"]
    first = report["recordings"][0]
    assert first["channel_names"] == ["C3", "C4"]
    assert [row["name"] for row in first["electrodes"]["rows"]] == ["C4", "C3", "REF"]
    assert first["electrodes"]["rows"][0]["x"] == "n/a"
    assert first["coordsystem"]["EEGCoordinateUnits"] == "mm"
    assert first["scan"]["row_index"] == 1
    assert first["scan"]["row"]["filename"] == f"eeg/{paths[0].name}"
    assert first["scan"]["row"]["quality"] == "n/a"
    assert first["sidecar_paths"]["coordsystem"] == "sub-01/sub-01_coordsystem.json"
    assert first["sidecar_paths"]["scans"] == "sub-01/sub-01_scans.tsv"
    for path in first["sidecar_paths"].values():
        assert (
            first["bids_source"]["files"][path]["sha256"]
            == hashlib.sha256((root / path).read_bytes()).hexdigest()
        )
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "participants.json",
        "sub-01/sub-01_electrodes.tsv",
        "sub-01/sub-01_coordsystem.json",
        "sub-01/sub-01_scans.tsv",
    ],
)
def test_new_sidecar_changes_rejected_before_opening_signal(
    sidecars, monkeypatch, name
):
    """Every newly consumed metadata file participates in source replay checks."""
    root, _ = sidecars
    descriptor = inspect_bids_eeg(root)["recordings"][0]
    with (root / name).open("a", encoding="utf-8") as stream:
        stream.write("\n")

    def unexpected(*args, **kwargs):
        raise AssertionError("Changed metadata reached signal opening")

    monkeypatch.setattr(adapter, "load_eeg_to_mne", unexpected)
    with pytest.raises(ValueError, match="Inspected source changed"):
        open_bids_eeg_recording(root, descriptor)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_scan",
        "missing_filename",
        "invalid_coordinates",
        "ambiguous_geometry",
        "electrode_columns",
    ],
)
def test_malformed_or_ambiguous_recording_sidecars_block_catalogue(sidecars, fault):
    """A recording with unresolved sidecar identity cannot disappear silently."""
    root, paths = sidecars
    directory = paths[0].parent.parent
    if fault == "duplicate_scan":
        with (directory / "sub-01_scans.tsv").open("a", encoding="utf-8") as stream:
            stream.write(f"eeg/{paths[0].name}\tn/a\tduplicate\n")
    elif fault == "missing_filename":
        (directory / "sub-01_scans.tsv").write_text("acq_time\nn/a\n", encoding="utf-8")
    elif fault == "invalid_coordinates":
        (directory / "sub-01_coordsystem.json").write_text("[]", encoding="utf-8")
    elif fault == "electrode_columns":
        (directory / "sub-01_electrodes.tsv").write_text(
            "name\tx\nC3\t1\n", encoding="utf-8"
        )
    else:
        data = (directory / "sub-01_electrodes.tsv").read_text(encoding="utf-8")
        (directory / "sub-01_electrodes.tsv").unlink()
        for space in ("A", "B"):
            (directory / f"sub-01_space-{space}_electrodes.tsv").write_text(
                data, encoding="utf-8"
            )
    report = inspect_bids_eeg(root)
    assert not report["ready"]
    assert report["recordings"] is None and report["events"] is None
    assert any(
        issue["recording_id"].endswith(paths[0].name)
        for issue in report["issues"]
        if issue["severity"] == "error"
    )


def test_absent_optional_metadata_stays_unknown(bids):
    """Absent geometry does not become inferred sensor positions or quality."""
    root, _ = bids
    report = inspect_bids_eeg(root)
    assert report["ready"]
    assert report["dataset"]["readme"] is None
    assert report["dataset"]["participants_json"] is None
    first = report["recordings"][0]
    assert (
        first["electrodes"] is None
        and first["coordsystem"] is None
        and first["scan"] is None
    )
    assert first["sidecar_paths"]["scans"] is None


def test_missing_coordinate_units_and_scan_row_remain_explicit(sidecars):
    """Incomplete optional facts stay available with diagnostics for consumers."""
    root, paths = sidecars
    directory = paths[0].parent.parent
    (directory / "sub-01_coordsystem.json").write_text("{}", encoding="utf-8")
    (directory / "sub-01_scans.tsv").write_text(
        "filename\tacq_time\neeg/other_eeg.bdf\tn/a\n", encoding="utf-8"
    )
    report = inspect_bids_eeg(root, subjects=["01"])
    assert report["ready"]
    first = report["recordings"][0]
    assert first["coordsystem"] == {} and first["scan"] is None
    assert {issue["code"] for issue in report["issues"]} >= {
        "coordinate_units_missing",
        "scan_row_missing",
    }


def test_changed_metadata_during_read_is_rejected(bids, monkeypatch):
    """A saved checksum must not describe different bytes from the parsed values."""
    root, _ = bids
    original = adapter._table

    def changed(path):
        result = original(path)
        if path.name == "participants.tsv":
            with path.open("a", encoding="utf-8") as stream:
                stream.write("sub-03\tnew\n")
        return result

    monkeypatch.setattr(adapter, "_table", changed)
    report = inspect_bids_eeg(root)
    assert not report["ready"] and report["dataset"] is None
    assert "Metadata changed during inspection" in report["issues"][-1]["message"]


def test_coordinate_system_matches_the_electrode_space(sidecars):
    """A single electrode space must not acquire another space's units/frame."""
    root, paths = sidecars
    directory = paths[0].parent.parent
    (directory / "sub-01_electrodes.tsv").rename(
        directory / "sub-01_space-A_electrodes.tsv"
    )
    (directory / "sub-01_coordsystem.json").rename(
        directory / "sub-01_space-A_coordsystem.json"
    )
    (directory / "sub-01_space-B_coordsystem.json").write_text(
        '{"EEGCoordinateUnits":"m"}', encoding="utf-8"
    )
    report = inspect_bids_eeg(root, subjects=["01"])
    assert report["ready"]
    first = report["recordings"][0]
    assert first["coordsystem"]["EEGCoordinateUnits"] == "mm"
    assert first["sidecar_paths"]["coordsystem"].endswith("space-A_coordsystem.json")
    assert "sub-01/sub-01_space-B_coordsystem.json" not in first["bids_source"]["files"]


def test_scans_are_scoped_to_the_recording_session(sidecars):
    """A subject-level manifest cannot substitute another acquisition's row."""
    root, paths = sidecars
    subject = paths[0].parent.parent
    session = subject / "ses-01"
    session.mkdir()
    paths[0].parent.rename(session / "eeg")
    for path in (session / "eeg").iterdir():
        path.rename(path.with_name(path.name.replace("sub-01_", "sub-01_ses-01_")))
    (session / "sub-01_ses-01_scans.tsv").write_text(
        "filename\tacq_time\neeg/sub-01_ses-01_task-imagery_eeg.bdf\t"
        "2020-02-01T12:00:00\n",
        encoding="utf-8",
    )
    report = inspect_bids_eeg(root, subjects=["01"])
    assert report["ready"]
    first = report["recordings"][0]
    assert first["scan"]["row"]["acq_time"] == "2020-02-01T12:00:00"
    assert first["sidecar_paths"]["scans"] == "sub-01/ses-01/sub-01_ses-01_scans.tsv"


def test_readme_preserves_line_endings_and_metadata_is_parsed_once(bids, monkeypatch):
    """The dataset record is shared by scope, not reparsed for each recording."""
    root, _ = bids
    (root / "README.txt").write_bytes(b"First line\r\nSecond line\r\n")
    original = adapter._table
    calls = []

    def watched(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(adapter, "_table", watched)
    report = inspect_bids_eeg(root)
    assert report["ready"]
    assert report["dataset"]["readme"] == "First line\r\nSecond line\r\n"
    assert calls.count("participants.tsv") == 1


def test_existing_version_one_source_descriptor_still_opens(bids):
    """The new report format does not change the existing source-snapshot reader."""
    root, _ = bids
    descriptor = inspect_bids_eeg(root, subjects=["01"])["recordings"][0]
    for key in ("electrodes", "coordsystem", "scan", "sidecar_paths"):
        descriptor.pop(key)
    assert descriptor["bids_source"]["version"] == 1
    with open_bids_eeg_recording(root, json.loads(json.dumps(descriptor))) as raw:
        assert raw.n_times == descriptor["n_samples"]
        assert not raw.preload
