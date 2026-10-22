"""Regressions for BIDS tables and metadata that cannot round-trip through Raw."""

import hashlib
import json

import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_array_equal

from aeon_neuro.utils.bids import convert_eeg_to_bids


@pytest.fixture
def recording_factory(tmp_path):
    """Create small independent BIDS recordings with custom source metadata."""
    pybv = pytest.importorskip("pybv")
    root = tmp_path / "source"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Metadata regression",
                "BIDSVersion": "1.9.0",
                "License": "CC0",
                "DatasetDOI": "doi:10.1234/example",
                "Authors": ["Example Author"],
            }
        )
    )
    (root / "participants.tsv").write_text(
        "participant_id\tsex\tgroup\tplayer1_age\n"
        "sub-01\tF\tCovid\t22\nsub-02\tM\tControl\t23\n"
    )
    (root / "participants.json").write_text(
        json.dumps(
            {
                "group": {"Description": "Published participant group"},
                "player1_age": {"Description": "First player age", "Units": "year"},
            }
        )
    )

    def make(subject="01", run="01", behavioral=False):
        source = BIDSPath(
            root=root,
            subject=subject,
            task="test",
            run=run,
            datatype="eeg",
            suffix="eeg",
            extension=".vhdr",
        )
        source.directory.mkdir(parents=True, exist_ok=True)
        pybv.write_brainvision(
            data=np.arange(300).reshape(3, 100) * 1e-7,
            sfreq=100,
            ch_names=["Cz", "Pz", "ECG"],
            folder_out=source.directory,
            fname_base=source.fpath.stem,
        )
        source.copy().update(extension=".json").fpath.write_text(
            json.dumps(
                {
                    "TaskName": "test",
                    "SamplingFrequency": 100,
                    "PowerLineFrequency": 50,
                    "RecordingDuration": 1.25,
                    "EEGChannelCount": 2,
                    "ECGChannelCount": 1,
                    "EEGReference": "common",
                    "EEGGround": "declared-ground",
                }
            )
        )
        source.copy().update(suffix="channels", extension=".tsv").fpath.write_text(
            "name\ttype\tunits\tstatus\nCz\tEEG\tuV\tgood\n"
            "Pz\tEEG\tuV\tbad\nECG\tECG\tuV\tgood\n"
        )
        events = (
            "onset\tduration\tonset_sample\ttrial_num\tplayer1_resp\tplayer2_resp\n"
            "0.2\t0.1\t20\t1\t1\t3\n0.5\t0.1\t50\t2\t2\t1\n"
            if behavioral
            else "onset\tduration\tsample\ttrial_type\tvalue\n"
            "0.2\t0.1\t20\tphotic_stimulation\t71\n"
            "0.5\t0.1\t50\tphotic_stimulation\t5\n"
        )
        source.copy().update(suffix="events", extension=".tsv").fpath.write_text(events)
        source.copy().update(suffix="events", extension=".json").fpath.write_text(
            json.dumps(
                {
                    "player1_resp": {
                        "Levels": {"1": "rock", "2": "paper", "3": "scissors"}
                    }
                }
            )
        )
        # This imported template montage has no fiducials. Re-exporting it through
        # MNE-BIDS alone can change its frame and generate empty landmark arrays.
        coordinate = source.copy().update(task=None, run=None)
        coordinate.update(suffix="electrodes", extension=".tsv").fpath.write_text(
            "name\tx\ty\tz\nCz\t0\t0\t90\nPz\t0\t-50\t80\nECG\tn/a\tn/a\tn/a\n"
        )
        coordinate.update(suffix="coordsystem", extension=".json").fpath.write_text(
            json.dumps({"EEGCoordinateSystem": "MNI305", "EEGCoordinateUnits": "mm"})
        )
        return source

    return make


def _hashes(root):
    """Snapshot a small generated fixture to detect unintended writes."""
    return {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("behavioral", [False, True])
def test_bids_source_preserves_metadata(recording_factory, tmp_path, behavioral):
    """Retain event semantics, participant fields and template coordinates."""
    source = recording_factory(behavioral=behavioral)
    original_files = _hashes(source.root)
    original = read_raw_bids(source)
    destination = source.copy().update(root=tmp_path / "output", extension=None)

    with pytest.warns(UserWarning, match="RecordingDuration"):
        written = convert_eeg_to_bids(source, destination)
    restored = read_raw_bids(written)

    assert written.root == destination.root
    assert_array_equal(restored.get_data(), original.get_data())
    assert restored.get_channel_types() == ["eeg", "eeg", "ecg"]
    assert restored.info["bads"] == ["Pz"]
    assert restored.info["line_freq"] == 50
    assert_array_equal(
        restored.annotations.description, original.annotations.description
    )
    assert_array_equal(restored.annotations.onset, original.annotations.onset)
    assert list(restored.annotations.extras) == list(original.annotations.extras)
    for suffix, extension in (
        ("events", ".tsv"),
        ("events", ".json"),
        ("channels", ".tsv"),
        ("electrodes", ".tsv"),
        ("coordsystem", ".json"),
    ):
        assert source.find_matching_sidecar(suffix, extension).read_bytes() == (
            written.find_matching_sidecar(suffix, extension).read_bytes()
        )
    metadata = json.loads(written.copy().update(extension=".json").fpath.read_text())
    assert metadata["RecordingDuration"] == 1
    assert metadata["EEGReference"] == "common"
    assert metadata["EEGGround"] == "declared-ground"
    assert (written.root / "participants.tsv").read_text() == (
        "participant_id\tsex\tgroup\tplayer1_age\nsub-01\tF\tCovid\t22\n"
    )
    assert (written.root / "participants.json").read_bytes() == (
        source.root / "participants.json"
    ).read_bytes()
    description = json.loads((written.root / "dataset_description.json").read_text())
    assert "DatasetDOI" not in description
    assert description["SourceDatasets"] == [{"DOI": "doi:10.1234/example"}]
    assert description["Authors"] == ["Example Author"]
    assert _hashes(source.root) == original_files


def test_bids_source_appends_without_losing_rows(recording_factory, tmp_path):
    """Appending subjects and runs retains participant labels and prior scans."""
    sources = [
        recording_factory(),
        recording_factory(run="02"),
        recording_factory(subject="02"),
    ]
    output = tmp_path / "output"
    for source in sources:
        convert_eeg_to_bids(source, source.copy().update(root=output))
    assert (output / "participants.tsv").read_bytes() == (
        sources[0].root / "participants.tsv"
    ).read_bytes()
    scans = (output / "sub-01/sub-01_scans.tsv").read_text()
    assert "run-01" in scans and "run-02" in scans
    before = _hashes(output)
    with pytest.raises(FileExistsError):
        convert_eeg_to_bids(sources[0], sources[0].copy().update(root=output))
    assert _hashes(output) == before


@pytest.mark.parametrize("problem", ["sample", "bounds", "rate", "participant"])
def test_bids_source_rejects_inconsistent_metadata(
    recording_factory, tmp_path, problem
):
    """Inconsistent source timing, signal rate or identity fails before writing."""
    source = recording_factory()
    if problem in ("sample", "bounds"):
        source.copy().update(suffix="events", extension=".tsv").fpath.write_text(
            "onset\tduration\tsample\n"
            + ("0.2\t0.1\t21\n" if problem == "sample" else "1\t0\t100\n")
        )
    elif problem == "rate":
        path = source.copy().update(extension=".json").fpath
        metadata = json.loads(path.read_text())
        metadata["SamplingFrequency"] = 101
        path.write_text(json.dumps(metadata))
    else:
        (source.root / "participants.tsv").write_text(
            "participant_id\tgroup\nsub-02\tControl\n"
        )
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        convert_eeg_to_bids(source, source.copy().update(root=output))
    assert not output.exists()


def test_bids_source_rejects_unsafe_destinations(recording_factory, tmp_path):
    """Source data, identifiers and incompatible destination metadata are protected."""
    source = recording_factory()
    before = _hashes(source.root)
    for destination in (
        source,
        source.copy().update(root=source.root / "output"),
        source.copy().update(root=tmp_path / "output", subject="02"),
    ):
        with pytest.raises(ValueError):
            convert_eeg_to_bids(source, destination)
    output = tmp_path / "output"
    destination = source.copy().update(root=output)
    convert_eeg_to_bids(source, destination)
    before_output = _hashes(output)
    second = recording_factory(subject="02")
    dictionary = source.root / "participants.json"
    dictionary.write_text('{"group": {"Description": "Conflicting meaning"}}')
    with pytest.raises(FileExistsError, match="metadata"):
        convert_eeg_to_bids(second, second.copy().update(root=output))
    assert _hashes(output) == before_output
    dictionary.write_bytes(
        json.dumps(
            {
                "group": {"Description": "Published participant group"},
                "player1_age": {"Description": "First player age", "Units": "year"},
            }
        ).encode()
    )
    assert all(_hashes(source.root)[p] == digest for p, digest in before.items())


def test_bids_source_writer_failure_leaves_destination_untouched(
    recording_factory, tmp_path, monkeypatch
):
    """A failed backend cannot publish a partially prepared dataset."""
    source = recording_factory()
    output = tmp_path / "output"

    def fail(raw, bids_path):
        bids_path.root.mkdir(exist_ok=True)
        (bids_path.root / "partial.txt").write_text("partial")
        raise RuntimeError("writer failed")

    monkeypatch.setattr("aeon_neuro.utils._bids_metadata.write_raw_bids", fail)
    with pytest.raises(RuntimeError, match="writer failed"):
        convert_eeg_to_bids(source, source.copy().update(root=output))
    assert not output.exists()
