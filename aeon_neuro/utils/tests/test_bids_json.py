"""Test EEG JSON inheritance independently of native signal readers."""

import json

import pytest
from mne_bids import BIDSPath

from aeon_neuro.utils._bids_json import _read_eeg_json


@pytest.fixture
def source(tmp_path):
    """Provide a recording identity without requiring a signal payload."""
    path = BIDSPath(
        root=tmp_path,
        subject="01",
        session="01",
        task="rest",
        run="01",
        datatype="eeg",
        suffix="eeg",
        extension=".edf",
    )
    path.directory.mkdir(parents=True)
    return path


def write(path, values):
    """Write metadata at an existing fixture directory level."""
    path.write_text(json.dumps(values))
    return path.read_bytes()


def test_inheritance_and_origins(source):
    """Retain inherited keys, override whole values and preserve source files."""
    root = source.root / "task-rest_eeg.json"
    subject = source.root / "sub-01/sub-01_task-rest_eeg.json"
    session = source.root / "sub-01/ses-01/sub-01_ses-01_task-rest_eeg.json"
    recording = source.copy().update(extension=".json").fpath
    originals = {
        root: write(
            root,
            {
                "PowerLineFrequency": 50,
                "Manufacturer": "Example",
                "SoftwareFilters": {"old": "root"},
            },
        ),
        subject: write(subject, {"EEGReference": "common"}),
        session: write(session, {"PowerLineFrequency": 60}),
        recording: write(
            recording, {"TaskName": "rest", "SoftwareFilters": {"new": "recording"}}
        ),
    }
    metadata, provenance = _read_eeg_json(source)
    assert metadata == {
        "PowerLineFrequency": 60,
        "Manufacturer": "Example",
        "EEGReference": "common",
        "TaskName": "rest",
        "SoftwareFilters": {"new": "recording"},
    }
    assert provenance["sources"] == [
        path.relative_to(source.root).as_posix() for path in originals
    ]
    assert provenance["fields"]["Manufacturer"] == "task-rest_eeg.json"
    assert (
        provenance["fields"]["PowerLineFrequency"]
        == session.relative_to(source.root).as_posix()
    )
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("generic_name", ["eeg.json", "task-rest_eeg.json"])
def test_root_metadata_and_unrelated_files(source, generic_name):
    """Exclude other tasks, subjects, runs and descendant directories."""
    write(source.root / generic_name, {"PowerLineFrequency": 50})
    for name in [
        "task-other_eeg.json",
        "sub-02_task-rest_eeg.json",
        "task-rest_run-02_eeg.json",
    ]:
        (source.root / name).write_text("invalid unrelated JSON")
    extra = source.directory / "nested"
    extra.mkdir()
    write(extra / "task-rest_eeg.json", {"PowerLineFrequency": 99})
    metadata, provenance = _read_eeg_json(source)
    assert metadata == {"PowerLineFrequency": 50}
    assert provenance["sources"] == [generic_name]


def test_same_level_ambiguity_is_rejected(source):
    """Filename specificity cannot justify choosing between ambiguous sidecars."""
    write(source.root / "eeg.json", {"PowerLineFrequency": 50})
    write(source.root / "task-rest_eeg.json", {"PowerLineFrequency": 60})
    with pytest.raises(ValueError, match="Multiple applicable"):
        _read_eeg_json(source)


@pytest.mark.parametrize(
    "content", ["invalid JSON", "[]", '{"PowerLineFrequency": NaN}']
)
def test_invalid_applicable_json_is_rejected(source, content):
    """Do not manufacture an effective value from malformed metadata."""
    (source.root / "task-rest_eeg.json").write_text(content)
    with pytest.raises(ValueError):
        _read_eeg_json(source)


def test_missing_metadata_is_rejected(source):
    """Reject recordings without any applicable EEG JSON metadata."""
    with pytest.raises(ValueError, match="No applicable"):
        _read_eeg_json(source)
