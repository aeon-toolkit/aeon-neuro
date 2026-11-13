"""Validate restored identities, source grids and collection declarations."""

import json
from copy import deepcopy

import numpy as np
import pytest

from aeon_neuro.datasets import EEGDatasetInfo


def source_state():
    """Use a small, fully specified recording without any signal IO."""
    return EEGDatasetInfo(
        recordings=[
            dict(
                recording_id="r0",
                participant_id="01",
                channel_names=["C3"],
                sampling_frequency=8.0,
                n_samples=8,
            )
        ]
    ).to_dict()


def current_state():
    """Return a current case linked to the known original recording."""
    state = source_state()
    state["view"] = dict(
        version=1,
        kind="cases",
        rows=[
            dict(
                case_id="c0",
                recording_id="r0",
                participant_id="01",
                group_id="g0",
                source_support=[0, 8],
            )
        ],
        layouts=[
            dict(
                channel_names=["C3"],
                sampling_frequency=8.0,
                units=["V"],
                n_samples=8,
                source_origin_sample=0.0,
                source_sample_step=1.0,
            )
        ],
    )
    return state


@pytest.mark.parametrize("subject", ["02", 1, "1"])
def test_restore_rejects_subject_conflicting_with_known_source(subject):
    """The original participant identity cannot silently change in case state."""
    state = current_state()
    state["view"]["rows"][0]["participant_id"] = subject
    with pytest.raises(ValueError):
        EEGDatasetInfo(state)


@pytest.mark.parametrize(
    "step,length", [(np.int8(4), 33), (np.int16(1024), 33), (np.int64(2**62), 3)]
)
def test_numpy_step_cannot_wrap_past_support_validation(step, length):
    """Exact Python arithmetic establishes that the final point is outside."""
    state = current_state()
    last = (length - 1) * int(step)
    assert last >= 8
    state["view"]["layouts"][0].update(
        source_sample_step=step, n_samples=length, sampling_frequency=8.0 / int(step)
    )
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        EEGDatasetInfo(state)


@pytest.mark.parametrize(
    "issues",
    [None, {}, "warning", [None], [{}], [{"severity": "warning", "message": "note"}]],
)
def test_malformed_diagnostics_fail_during_restoration(issues):
    """Reject malformed structural diagnostics before selection can crash."""
    state = source_state()
    state["issues"] = issues
    with pytest.raises(ValueError):
        EEGDatasetInfo(state)


@pytest.mark.parametrize(
    "location", ["explicit_source", "current_subject", "current_group"]
)
def test_supported_numpy_identifiers_have_portable_state(location):
    """Integral identifiers should behave like their ordinary Python values."""
    if location == "explicit_source":
        record = source_state()["recordings"][0]
        record["participant_id"] = np.int64(1)
        info = EEGDatasetInfo(recordings=[record])
    else:
        state = current_state()
        key = "participant_id" if location == "current_subject" else "group_id"
        state["view"]["rows"][0][key] = np.int64(1)
        if key == "participant_id":
            state["recordings"][0]["participant_id"] = 1
            state["dataset"]["selected_participants"] = [1]
        info = EEGDatasetInfo(state)
    serialized = json.dumps(info.to_dict())
    assert EEGDatasetInfo(json.loads(serialized)).to_dict() == info.to_dict()


@pytest.mark.parametrize(
    "shape,dtype,names,units",
    [
        ((0, 1, 8), complex, ["C3"], ["V"]),
        ((0, 1, 8), bool, ["C3"], ["V"]),
        ((0, 1, 8), object, ["C3"], ["V"]),
        ((0, 2, 8), float, ["C3"], ["V"]),
        ((0, 0, 8), float, ["C3"], ["V"]),
        ((0, 1, 0), float, ["C3"], ["V"]),
        ((0, 1, 8), float, ["C3"], []),
        ((0, 1, 8), float, ["C3"], [False]),
    ],
)
def test_empty_dense_input_still_validates_declared_shape_and_type(
    shape, dtype, names, units
):
    """Zero cases should not erase incompatible declared channel/type facts."""
    with pytest.raises(ValueError):
        EEGDatasetInfo.from_arrays(
            np.empty(shape, dtype=dtype), sfreq=8, channel_names=names, units=units
        )


@pytest.mark.parametrize("identity", [42, ""])
def test_recording_reference_type_is_checked_without_source_descriptors(identity):
    """Original source IDs are nonempty strings or absent, even without context."""
    state = current_state()
    state["recordings"] = []
    state["source_ids"] = []
    state["dataset"]["selected_participants"] = []
    state["view"]["rows"][0]["recording_id"] = identity
    with pytest.raises(ValueError):
        EEGDatasetInfo(state)


@pytest.mark.parametrize(
    "original,current",
    [
        ("01", "01"),
        ("01", None),
        (None, "02"),
        (None, None),
        (1, np.int64(1)),
        ("1", "1"),
    ],
)
def test_compatible_or_unknown_subjects_allow_independent_groups(original, current):
    """Subject consistency must not infer unknowns or constrain declared groups."""
    state = current_state()
    state["recordings"][0]["participant_id"] = original
    state["dataset"]["selected_participants"] = [] if original is None else [original]
    state["view"]["rows"][0].update(participant_id=current, group_id="custom-group")
    info = EEGDatasetInfo(state)
    assert info.participant_ids == (current,)
    assert info.groups.tolist() == ["custom-group"]
    assert info.recordings[0].metadata["participant_id"] == original


def test_subject_consistency_keeps_integer_and_string_identities_distinct():
    """A textual representation does not identify the same declared subject."""
    state = current_state()
    state["recordings"][0]["participant_id"] = 1
    state["dataset"]["selected_participants"] = [1]
    state["view"]["rows"][0]["participant_id"] = "1"
    with pytest.raises(ValueError, match="participant disagrees"):
        EEGDatasetInfo(state)


@pytest.mark.parametrize("step,length", [(np.int8(2), 4), (np.float32(0.5), 16)])
def test_numpy_layout_values_are_portable_without_mutating_input(step, length):
    """Supported NumPy layout scalars retain their values as Python numbers."""
    state = current_state()
    state["view"]["rows"][0]["source_support"] = [np.int64(0), np.int64(8)]
    state["view"]["layouts"][0].update(
        source_origin_sample=np.int64(0),
        source_sample_step=step,
        n_samples=np.int64(length),
        sampling_frequency=np.float64(8.0 / float(step)),
    )
    before = deepcopy(state)
    info = EEGDatasetInfo(state)
    exported = json.loads(json.dumps(info.to_dict()))
    layout = exported["view"]["layouts"][0]
    assert layout["source_sample_step"] == float(step)
    assert layout["n_samples"] == length
    assert exported["view"]["rows"][0]["source_support"] == [0, 8]
    EEGDatasetInfo(exported).validate_layout(np.zeros((1, 1, length)))
    assert state == before
    assert isinstance(state["view"]["layouts"][0]["source_sample_step"], np.generic)


@pytest.mark.parametrize("length", [3, 10**400])
def test_nonfinite_derived_grid_fails_without_allocating_samples(length):
    """Finite operands cannot establish a valid grid when arithmetic overflows."""
    state = current_state()
    state["view"]["layouts"][0].update(
        n_samples=length, source_sample_step=1e308, sampling_frequency=8e-308
    )
    with pytest.raises(ValueError, match="grid must be finite"):
        EEGDatasetInfo(state)


@pytest.mark.parametrize(
    "key,value",
    [
        ("severity", None),
        ("severity", " "),
        ("message", None),
        ("message", 42),
        ("recording_id", 1),
        ("recording_id", " "),
    ],
)
def test_diagnostic_field_types_are_validated(key, value):
    """Only structural diagnostic fields are constrained, not extensions."""
    state = source_state()
    issue = dict(recording_id=None, severity="warning", message="note")
    issue[key] = value
    state["issues"] = [issue]
    with pytest.raises(ValueError):
        EEGDatasetInfo(state)


@pytest.mark.parametrize("issues", [None, [{"severity": "error"}]])
def test_diagnostic_validation_precedes_readiness_reporting(issues):
    """Unready reports still reject invalid diagnostics with a useful error."""
    state = source_state()
    state.update(ready=False, issues=issues)
    with pytest.raises(ValueError):
        EEGDatasetInfo(state)


def test_unready_metadata_keeps_its_valid_error_message():
    """Valid source errors remain available in the readiness exception."""
    state = source_state()
    state.update(
        ready=False,
        issues=[
            dict(recording_id=None, severity="error", message="missing source facts")
        ],
    )
    with pytest.raises(ValueError, match="missing source facts"):
        EEGDatasetInfo(state)


def test_diagnostic_extensions_and_scopes_survive_selection():
    """Dataset diagnostics and custom annotations keep their detached values."""
    state = source_state()
    state["issues"] = [
        dict(
            recording_id=identity, severity="notice", message="", custom={"values": [1]}
        )
        for identity in (None, "r0", "external")
    ]
    info = EEGDatasetInfo(state)
    assert info.to_dict() == state
    assert info[[]].to_dict()["issues"] == state["issues"][:1]
    selected = info[[0]].to_dict()
    assert selected["issues"] == state["issues"][:2]
    selected["issues"][0]["custom"]["values"].append(2)
    assert info.to_dict() == state


@pytest.mark.parametrize("dense", [True, False])
def test_valid_empty_collections_and_selections_still_work(dense):
    """Empty case axes remain useful without permitting empty signal axes."""
    X = np.empty((0, 1, 8)) if dense else []
    info = EEGDatasetInfo.from_arrays(X, sfreq=8, channel_names=["C3"], units=[None])
    selected, subset = info.select(X, [])
    assert len(selected) == len(subset) == 0
    subset.validate_layout(selected)


@pytest.mark.parametrize("dtype", [complex, bool, object])
def test_empty_layout_validation_rejects_nonnumeric_collections(dtype):
    """A valid empty subset does not make unrelated input dtypes acceptable."""
    info = EEGDatasetInfo.from_arrays(
        np.zeros((1, 1, 8)), sfreq=8, channel_names=["C3"]
    )[[]]
    with pytest.raises(ValueError):
        info.validate_layout(np.empty((0, 1, 8), dtype=dtype))


@pytest.mark.parametrize("identity", [None, "external-recording"])
def test_optional_source_context_keeps_valid_recording_references(identity):
    """Reference type validation does not invent required source descriptors."""
    state = current_state()
    state.update(recordings=[], source_ids=[])
    state["dataset"]["selected_participants"] = []
    state["view"]["rows"][0]["recording_id"] = identity
    info = EEGDatasetInfo(state)
    assert info.to_dict() == state


def test_numpy_subject_scope_and_rows_normalize_without_changing_source_input():
    """Known IDs are portable while flexible source annotations remain intact."""
    state = current_state()
    state["recordings"][0]["participant_id"] = np.int64(1)
    state["dataset"]["selected_participants"] = [np.int64(1)]
    state["view"]["rows"][0].update(participant_id=np.int64(1), group_id=np.int64(2))
    state["recordings"][0]["custom"] = {"values": [1]}
    info = EEGDatasetInfo(state)
    exported = json.loads(json.dumps(info.to_dict()))
    assert type(info.participant_ids[0]) is int
    assert type(info.groups[0]) is int
    assert exported["dataset"]["selected_participants"] == [1]
    assert exported["recordings"][0]["custom"] == {"values": [1]}
    assert isinstance(state["recordings"][0]["participant_id"], np.integer)
    assert isinstance(state["dataset"]["selected_participants"][0], np.integer)
