"""Check companion discovery against formats already understood by MNE."""

import re
from pathlib import Path
from shutil import copyfile

import mne
import numpy as np
import pytest

from aeon_neuro.utils.data_io import load_eeg_to_mne


@pytest.fixture
def brainvision_files(tmp_path):
    """Copy the bundled recording, retaining its linked data and marker files."""
    original = (
        Path(__file__).resolve().parents[3]
        / "example_raw_eeg/basic_classification_task/sub-01/ses-01/eeg"
        / "sub-01_ses-01_task-task_run-01_eeg.vhdr"
    )
    for suffix in (".vhdr", ".vmrk", ".eeg"):
        copyfile(
            original.with_suffix(suffix), tmp_path / original.with_suffix(suffix).name
        )
    return tmp_path / original.name


@pytest.mark.parametrize("variant", ["codepage", "neurone_section"])
@pytest.mark.parametrize("companion", [".eeg", ".vmrk"])
def test_companion_matches_declared_header_metadata(
    brainvision_files, variant, companion
):
    """A supported header encoding/section spelling must work from each entry file."""
    header = brainvision_files
    text = header.read_text(encoding="utf-8-sig")
    if variant == "codepage":
        # CP932 filenames cannot be recovered by assuming Windows-1252 bytes.
        for suffix in (".eeg", ".vmrk"):
            source = header.with_suffix(suffix)
            target = header.parent / ("記録" + suffix)
            source.rename(target)
            text = text.replace(source.name, target.name)
        marker = header.parent / "記録.vmrk"
        marker_text = marker.read_text(encoding="utf-8-sig")
        marker_text = marker_text.replace(header.with_suffix(".eeg").name, "記録.eeg")
        marker.write_text(marker_text, encoding="utf-8")
        text = re.sub(r"(?m)^Codepage=.*$", "Codepage=cp932", text)
        header.write_bytes(text.replace("µ", "u").encode("cp932"))
        source = header.parent / ("記録" + companion)
    else:
        text = text.replace("[Common Infos]", "[Common infos]")
        header.write_text(text, encoding="utf-8")
        source = header.with_suffix(companion)

    expected = mne.io.read_raw_brainvision(header, verbose=False)
    actual = load_eeg_to_mne(source, verbose=False)
    assert not actual.preload
    assert actual.ch_names == expected.ch_names
    assert actual.info["sfreq"] == expected.info["sfreq"]
    np.testing.assert_array_equal(actual.get_data(stop=32), expected.get_data(stop=32))
    np.testing.assert_array_equal(actual.annotations.onset, expected.annotations.onset)
    np.testing.assert_array_equal(
        actual.annotations.description, expected.annotations.description
    )


def test_unrelated_bad_header_does_not_block_discovery(brainvision_files):
    """A malformed sibling must not break an otherwise readable companion route."""
    (brainvision_files.parent / "000-unrelated.vhdr").write_bytes(b"\x81not a header")
    actual = load_eeg_to_mne(brainvision_files.with_suffix(".eeg"), verbose=False)
    assert actual.n_times > 0


def test_ambiguous_companion_still_requires_explicit_header(brainvision_files):
    """Encoding support must not silently choose one of two matching headers."""
    copyfile(brainvision_files, brainvision_files.parent / "duplicate.vhdr")
    with pytest.raises(ValueError, match="Multiple BrainVision headers"):
        load_eeg_to_mne(brainvision_files.with_suffix(".eeg"))
