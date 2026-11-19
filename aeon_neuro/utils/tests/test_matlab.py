"""Exercise calibrated MATLAB ingestion and BIDS round-trips with real files."""

import csv
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import savemat

from aeon_neuro.utils.bids import convert_eeg_to_bids
from aeon_neuro.utils.matlab import MatlabSchema, load_matlab_recording

pytest.importorskip("pymatreader", minversion="1.3.2")


def _save(path, values, version="5"):
    """Create genuine MATLAB files, including MATLAB-compatible HDF5 metadata."""
    if version == "7.3":
        storage = pytest.importorskip("hdf5storage")
        storage.savemat(str(path), values, fmt="7.3", store_python_metadata=False)
    else:
        savemat(path, values)
    return path


def _schema(**options):
    """Use different physiological units to catch scalar calibration mistakes."""
    defaults = dict(
        signal_key="signals",
        sfreq=100,
        channel_names=["Cz", "EOG"],
        channel_types=["eeg", "eog"],
        units=["uV", "mV"],
    )
    defaults.update(options)
    return MatlabSchema(**defaults)


def _load_signals(tmp_path, signals=None, **schema_options):
    """Create a MAT file for tests of the shared signal schema."""
    if signals is None:
        signals = np.zeros((100, 2))
    path = _save(tmp_path / "signals.mat", {"signals": signals})
    return load_matlab_recording(path, schema=_schema(**schema_options))


def _tremmel_values(*, terminal=False):
    """Build short n-back/arithmetic logs with every distinct field populated."""
    count = 6000
    clock = 100 + np.arange(count) / 200
    signal = np.sin(np.arange(count)[:, None] / 20) * np.arange(1, 31)
    rows = [[100.0001, -99]]
    for i, onset in enumerate([100.5, 103.5]):
        rows.extend(
            [
                [onset, i],
                [onset + 1e-6, i + 1],
                [onset + 2e-6, 7],
                [onset + 1, -1 if i else 0],
            ]
        )
    rows.append([107, -99])
    for i, onset in enumerate([107.5, 125 if terminal else 116.5]):
        response = 130.003 if terminal and i else onset + 6
        rows.extend(
            [[onset, i], [onset + 1e-6, 11], [onset + 2e-6, 22], [response, 33]]
        )
    return {"data": np.column_stack([clock, signal]), "experiment": np.asarray(rows)}


def _iva_values():
    """Represent published integer counts, nested metadata and withheld labels."""
    return {
        "cnt": (np.arange(2000).reshape(1000, 2) - 1000).astype(np.int16),
        "nfo": {"fs": 100, "clab": np.array(["C3", "C4"], dtype=object)},
        "mrk": {
            "pos": [1, 101, 501],
            "y": [1, 2, np.nan],
            "className": np.array(["right", "foot"], dtype=object),
        },
    }


def _table(written, suffix="events"):
    """Read a BIDS table without annotation or numeric coercion."""
    path = written.copy().update(suffix=suffix, extension=".tsv").fpath
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write(recording, root, *, subject="01"):
    """Exercise the public converter with the actual BrainVision backend."""
    pytest.importorskip("pybv")
    return convert_eeg_to_bids(
        recording,
        BIDSPath(root=root, subject=subject, task="test"),
        format="BrainVision",
        allow_preload=True,
    )


@pytest.mark.parametrize("version", ["5", "7.3"])
@pytest.mark.parametrize("axis", [0, 1])
def test_matlab_schema_units_axes_and_roundtrip(tmp_path, version, axis):
    """Retain physical samples, source indices, duration and behavioural fields."""
    source = np.arange(800).reshape(400, 2) - 300
    schema = _schema(
        sample_axis=axis,
        scale=[2, 0.5],
        event_parser=lambda mat: [
            {
                "onset": sample,
                "duration": duration,
                "trial_type": label,
                "response": response,
            }
            for sample, duration, label, response in zip(
                mat["starts"], [0.25, 0], ["left", "right"], [2, -1]
            )
        ],
        event_time_unit="samples",
        event_index_base=1,
        event_descriptions={"response": "Original response, -1 means missing."},
        metadata={"EEGReference": "Cz", "PowerLineFrequency": 50},
    )
    path = _save(
        tmp_path / "generic.mat",
        {"signals": source if axis == 0 else source.T, "starts": [1, 400]},
        version,
    )
    recording = load_matlab_recording(path, schema=schema, verbose=False)
    expected = source.T * np.array([2e-6, 0.5e-3])[:, None]
    assert_allclose(recording.raw.get_data(), expected, rtol=1e-15)
    assert [e["sample"] for e in recording.events] == [0, 399]
    assert [e["source_sample"] for e in recording.events] == [1, 400]
    assert (
        recording.provenance["SourceSHA256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    written = _write(recording, tmp_path / "bids")
    restored = read_raw_bids(written, verbose=False)
    assert_allclose(restored.get_data(), expected, rtol=2e-7, atol=1e-12)
    assert restored.get_channel_types() == ["eeg", "eog"]
    assert restored.info["line_freq"] == 50
    assert_array_equal(restored.annotations.description, ["left", "right"])
    assert_allclose(restored.annotations.duration, [0.25, 0])
    rows = _table(written)
    assert [r["response"] for r in rows] == ["2", "-1"]
    assert rows[0]["source_sample"] == "1"
    assert float(rows[1]["onset"]) == 3.99
    sidecar = json.loads(written.copy().update(extension=".json").fpath.read_text())
    assert sidecar["EEGReference"] == "Cz"
    assert sidecar["Manufacturer"] == "n/a"
    assert (
        sidecar["MatlabSource"]["SourceSHA256"] == recording.provenance["SourceSHA256"]
    )
    assert not len(recording.raw.annotations)
    channels = _table(written, "channels")
    assert {c["low_cutoff"] for c in channels} == {"n/a"}
    assert {c["high_cutoff"] for c in channels} == {"n/a"}
    assert {c["status"] for c in channels} == {"n/a"}


@pytest.mark.parametrize("version", ["5", "7.3"])
def test_matlab_tremmel_events_and_roundtrip(tmp_path, version):
    """Decode all four records and retain an out-of-range terminal response."""
    values = _tremmel_values(terminal=True)
    path = _save(tmp_path / "tremmel.mat", values, version)
    with pytest.warns(UserWarning):
        recording = load_matlab_recording(
            path, adapter="tremmel2024", units="µV", verbose=False
        )
    assert recording.raw.ch_names[:4] == ["ECG", "HEOG", "VEOG", "FP1"]
    assert recording.raw.ch_names[-1] == "Inear"
    assert recording.raw.get_channel_types() == ["ecg", "eog", "eog"] + ["eeg"] * 27
    assert_allclose(recording.raw.get_data(), values["data"][:, 1:].T * 1e-6)
    assert len(recording.events) == 4
    first, last = recording.events[0], recording.events[-1]
    assert first["nback_level"] == 1 and first["stimulus"] == 7
    assert first["response"] == 0 and first["response_time"] == 1
    assert recording.events[1]["response_time"] == "n/a"
    assert last["operand1"] == 11 and last["operand2"] == 22
    assert last["response_timestamp"] == 130.003
    assert last["duration"] == 8.5
    assert last["recording_covers_event"] is False
    assert (
        recording.provenance["AdapterContext"]["ExperimentStarts"][1]["Timestamp"]
        == 107
    )
    assert any(d["code"] == "incomplete_event_windows" for d in recording.diagnostics)
    written = _write(recording, tmp_path / "bids")
    table = _table(written)
    assert len(table) == 4
    assert float(table[-1]["duration"]) == 8.5
    assert float(table[-1]["response_timestamp"]) == 130.003
    dictionary = json.loads(
        written.copy().update(suffix="events", extension=".json").fpath.read_text()
    )
    assert dictionary["response_time"]["Units"] == "s"
    # MNE annotations may crop incomplete terminal durations; the table above
    # is intentionally authoritative. No sample is padded to hide this fact.
    restored = read_raw_bids(written, verbose=False)
    assert restored.n_times == 6000
    assert recording.raw.info["highpass"] == 0.5
    assert recording.raw.info["lowpass"] == 100
    channels = _table(written, "channels")
    assert {c["low_cutoff"] for c in channels} == {"0.5"}
    assert {c["high_cutoff"] for c in channels} == {"100.0"}
    assert_allclose(
        restored.get_data(), recording.raw.get_data(), rtol=2e-7, atol=1e-12
    )


@pytest.mark.parametrize("version", ["5", "7.3"])
def test_matlab_iva_calibration_missing_labels_and_roundtrip(tmp_path, version):
    """Use 0.1 microvolt counts and preserve unlabeled cues without inventing y."""
    values = _iva_values()
    path = _save(tmp_path / "iva.mat", values, version)
    recording = load_matlab_recording(
        path, adapter="bci_competition_iii_iva", verbose=False
    )
    assert_allclose(recording.raw.get_data(), values["cnt"].T * 1e-7)
    assert [r["sample"] for r in recording.events] == [0, 100, 500]
    assert [r["trial_type"] for r in recording.events] == ["right", "foot", "unknown"]
    assert recording.events[-1]["value"] == "n/a"
    written = _write(recording, tmp_path / "bids")
    restored = read_raw_bids(written, verbose=False)
    assert_allclose(restored.get_data(), values["cnt"].T * 1e-7, rtol=2e-7, atol=1e-12)
    assert_array_equal(restored.annotations.description, ["right", "foot", "unknown"])
    assert_allclose(restored.annotations.duration, [3.5] * 3)
    assert _table(written)[-1]["value"] == "n/a"


@pytest.mark.parametrize("shape,axis", [((50, 1), 0), ((1, 50), 1)])
def test_matlab_single_channel_shape(tmp_path, shape, axis):
    """Restore singleton channel dimensions from file metadata after decoding."""
    values = np.arange(50).reshape(shape)
    path = _save(tmp_path / "one.mat", {"signals": values})
    recording = load_matlab_recording(
        path,
        schema=_schema(
            channel_names=["Cz"], channel_types="eeg", units="nV", sample_axis=axis
        ),
    )
    assert_allclose(recording.raw.get_data(), np.arange(50)[None, :] * 1e-9)


@pytest.mark.parametrize("shape", [(2, 100, 2), (1, 100, 2)])
def test_matlab_reject_epochs_including_squeezed_axis(tmp_path, shape):
    """Reject an original epoch axis even if the decoder would squeeze it."""
    path = _save(tmp_path / "epochs.mat", {"signals": np.zeros(shape)})
    with pytest.raises(ValueError, match="epochs"):
        load_matlab_recording(path, schema=_schema())


@pytest.mark.parametrize(
    "options,message",
    [
        ({"units": None}, "physical units"),
        ({"units": "counts"}, "units"),
        ({"units": ["uV"]}, "one value"),
        ({"scale": 0}, "positive"),
        ({"scale": True}, "numeric calibration"),
        ({"scale": [1, np.inf]}, "positive"),
        ({"highpass": -1}, "filter cutoffs"),
        ({"lowpass": np.nan}, "filter cutoffs"),
        ({"highpass": 30, "lowpass": 10}, "exceed lowpass"),
        ({"sfreq": 0}, "sfreq"),
        ({"sfreq": np.nan}, "sfreq"),
        ({"sfreq": True}, "sfreq"),
        ({"sample_axis": True}, "sample_axis"),
        ({"sample_axis": 0.0}, "sample_axis"),
        ({"sample_axis": 2}, "sample_axis"),
        ({"channel_names": ["Cz", "Cz"]}, "uniquely"),
        ({"channel_names": ["Cz"]}, "uniquely"),
        ({"channel_names": "Cz"}, "complete channel names"),
        ({"channel_types": "stim"}, "types"),
        ({"signal_columns": [0, 0]}, "signal_columns"),
        ({"signal_columns": [0, 2]}, "signal_columns"),
        ({"timestamp_column": 0}, "separate"),
        ({"timestamps_key": "absent"}, "missing"),
        ({"timing_tolerance": 0.5}, "timing_tolerance"),
        ({"timestamp_policy": "guess"}, "timestamp_policy"),
        ({"timestamp_policy": "fit"}, "requires source"),
        ({"event_time_unit": "minutes"}, "combination"),
        ({"event_index_base": 1}, "combination"),
        ({"event_index_base": 1.0, "event_time_unit": "samples"}, "combination"),
        ({"event_time_unit": "timestamps"}, "same clock"),
    ],
)
def test_matlab_invalid_schema(tmp_path, options, message):
    """Fail explicitly on ambiguous or physically invalid mappings."""
    with pytest.raises(ValueError, match=message):
        _load_signals(tmp_path, np.zeros((20, 2)), **options)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j])
def test_matlab_invalid_signals(tmp_path, bad):
    """Reject invalid signal values before unit conversion or export."""
    values = np.zeros((10, 2), dtype=complex if bad == 1j else float)
    values[3, 0] = bad
    with pytest.raises(ValueError, match="finite real"):
        _load_signals(tmp_path, values)


@pytest.mark.parametrize("axis", [0, 1])
def test_matlab_missing_values_in_auxiliary_column(tmp_path, axis):
    """Allow missing auxiliary data unless the schema uses it for timing."""
    eeg = np.arange(20)
    values = np.column_stack([eeg, np.full(eeg.size, np.nan)])
    path = _save(tmp_path / "aux.mat", {"signals": values if axis == 0 else values.T})
    schema = _schema(
        sample_axis=axis,
        signal_columns=[0],
        channel_names=["Cz"],
        channel_types="eeg",
        units="uV",
    )
    recording = load_matlab_recording(path, schema=schema)
    assert_allclose(recording.raw.get_data(), eeg[None, :] * 1e-6)
    assert recording.provenance["UnselectedColumnsZeroBased"] == [1]

    with pytest.raises(ValueError, match="Timestamps must be a finite vector"):
        load_matlab_recording(path, schema=replace(schema, timestamp_column=1))


@pytest.mark.parametrize(
    "kind", ["gap", "reset", "drift", "jitter", "wrong_length", "nan"]
)
def test_matlab_timestamp_failures(tmp_path, kind):
    """Prevent continuous Raw from concealing gaps or an incorrect sample grid."""
    stamps = 10 + np.arange(100) / 100
    if kind == "gap":
        stamps[50:] += 0.1
    elif kind == "reset":
        stamps[50:] -= 1
    elif kind == "drift":
        stamps = 10 + np.arange(100) / 110
    elif kind == "jitter":
        stamps[30] += 0.004
    elif kind == "wrong_length":
        stamps = stamps[:-1]
    else:
        stamps[50] = np.nan
    path = _save(
        tmp_path / "clock.mat", {"signals": np.zeros((100, 2)), "clock": stamps}
    )
    with pytest.raises(ValueError, match="Timestamp"):
        load_matlab_recording(
            path, schema=_schema(timestamps_key="clock", timestamp_policy="fit")
        )


def test_matlab_explicit_clock_fit(tmp_path):
    """Use a declared clock policy without changing samples or source timestamps."""
    stamps = 12345 + np.arange(2000) / 100.05
    stamps[1:-1:2] += 0.001
    path = _save(
        tmp_path / "clock.mat", {"signals": np.ones((2000, 2)), "clock": stamps}
    )
    schema = _schema(timestamps_key="clock")
    with pytest.raises(ValueError, match="regular grid"):
        load_matlab_recording(path, schema=schema)
    schema = replace(
        schema,
        timestamp_policy="fit",
        event_time_unit="timestamps",
        event_parser=lambda _: [
            {"onset": stamps[20], "duration": 0, "trial_type": "cue"}
        ],
    )
    recording = load_matlab_recording(path, schema=schema)
    assert_allclose(recording.raw.info["sfreq"], 100.05)
    assert recording.events[0]["sample"] == 20
    assert recording.events[0]["source_timestamp"] == stamps[20]
    assert recording.provenance["Timing"]["SignalInterpolation"] is False


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"onset": -0.01}, "outside"),
        ({"onset": 1.0}, "outside"),
        ({"onset": np.nan}, "finite"),
        ({"duration": -1}, "duration"),
        ({"duration": np.inf}, "duration"),
        ({"trial_type": ""}, "trial_type"),
        ({"trial_type": "cue\tbad"}, "tabs"),
        ({"sample": 4}, "reserved"),
        ({"new_column": 4}, "description"),
        ({"bad column": 4}, "column names"),
        ({"response": {"bad": 1}}, "scalar"),
    ],
)
def test_matlab_invalid_events(tmp_path, changes, message):
    """Reject errors rather than lose events or broadcast behavioural columns."""
    row = {"onset": 0.1, "duration": 0, "trial_type": "cue"}
    row.update(changes)
    with pytest.raises(ValueError, match=message):
        _load_signals(tmp_path, event_parser=lambda _: [row])


@pytest.mark.parametrize("base", [0, 1])
def test_matlab_sample_index_bases(tmp_path, base):
    """Preserve both inclusive sample bounds and reject fractional indices."""
    path = _save(tmp_path / "samples.mat", {"signals": np.zeros((10, 2))})
    schema = _schema(
        event_time_unit="samples",
        event_index_base=base,
        event_parser=lambda _: [
            {"onset": base + p, "duration": 0, "trial_type": "cue"} for p in [0, 9]
        ],
    )
    recording = load_matlab_recording(path, schema=schema)
    assert [row["sample"] for row in recording.events] == [0, 9]
    with pytest.raises(ValueError, match="integers"):
        load_matlab_recording(
            path,
            schema=replace(
                schema,
                event_parser=lambda _: [
                    {"onset": 1.5, "duration": 0, "trial_type": "cue"}
                ],
            ),
        )


def test_matlab_custom_adapter(tmp_path):
    """Allow local adapters without editing a global registry or executing MAT."""
    path = _save(tmp_path / "custom.mat", {"signals": np.zeros((100, 2)), "rate": 250})

    def adapter(mat, *, units):
        return _schema(sfreq=mat["rate"], units=units)

    recording = load_matlab_recording(path, adapter=adapter, units="uV")
    assert recording.raw.info["sfreq"] == 250
    assert recording.provenance["Adapter"] == "adapter"


@pytest.mark.parametrize("unit", ["timestamps", "seconds"])
@pytest.mark.parametrize(
    "count,origin,rate,declared,policy,jitter",
    [
        (400, 7558.80258, 200.005404, 200, "fit", 0),
        (42, 100, 200.0055, 200, "fit", 0),
        (100, 100, 100, 100, "nominal", 0.001),
        (100, 100, 100, 100, "nominal", -0.001),
    ],
)
def test_matlab_timestamp_endpoint_events(
    tmp_path, unit, count, origin, rate, declared, policy, jitter
):
    """Keep endpoint events despite accepted jitter or fitted-clock roundoff."""
    clock = origin + np.arange(count) / rate
    clock[-1] += jitter
    path = _save(
        tmp_path / "clock.mat",
        {
            "signals": np.zeros((count, 2)),
            "clock": clock,
        },
    )
    coordinates = clock[[0, -1]] - (clock[0] if unit == "seconds" else 0)
    schema = _schema(
        sfreq=declared,
        timestamps_key="clock",
        timestamp_policy=policy,
        event_time_unit=unit,
        event_parser=lambda _: [
            {"onset": stamp, "duration": 0, "trial_type": "edge"}
            for stamp in coordinates
        ],
    )
    recording = load_matlab_recording(path, schema=schema)
    assert [r["sample"] for r in recording.events] == [0, count - 1]
    written = _write(recording, tmp_path / "bids")
    restored = read_raw_bids(written, verbose=False)
    assert restored.n_times == count
    assert [int(r["sample"]) for r in _table(written)] == [0, count - 1]
    source_column = "source_onset" if unit == "seconds" else "source_timestamp"
    assert_allclose([r[source_column] for r in recording.events], coordinates)
    assert_allclose(
        restored.annotations.onset,
        [r["onset"] for r in recording.events],
        atol=1e-6,
    )
    # Do not let the accepted jitter or nearest-sample rounding admit events
    # that actually precede or follow the source signal.
    for outside in [coordinates[0] - 0.001, coordinates[-1] + 0.001]:
        with pytest.raises(ValueError, match="outside the recorded sample span"):
            load_matlab_recording(
                path,
                schema=replace(
                    schema,
                    event_parser=lambda _, onset=outside: [
                        {"onset": onset, "duration": 0, "trial_type": "outside"}
                    ],
                ),
            )


def test_matlab_seconds_endpoint_roundoff(tmp_path):
    """Permit machine roundoff without allowing another fraction of a sample."""
    endpoint = 99 / 100
    row = {"onset": np.nextafter(endpoint, np.inf), "duration": 0, "trial_type": "edge"}
    recording = _load_signals(tmp_path, event_parser=lambda _: [row])
    assert recording.events[0]["sample"] == 99
    row["onset"] = endpoint + 0.001
    with pytest.raises(ValueError, match="outside the recorded sample span"):
        _load_signals(tmp_path, event_parser=lambda _: [row])


def test_matlab_unknown_duration_and_simultaneous_events(tmp_path):
    """Keep overlapping event rows and unknown duration in the full BIDS table."""
    recording = _load_signals(
        tmp_path,
        event_parser=lambda _: [
            {"onset": 0.1, "duration": "n/a", "trial_type": "first"},
            {"onset": 0.1, "duration": 0.25, "trial_type": "second"},
        ],
    )
    written = _write(recording, tmp_path / "bids")
    rows = _table(written)
    assert [r["trial_type"] for r in rows] == ["first", "second"]
    assert [r["duration"] for r in rows] == ["n/a", "0.25"]
    assert [r["sample"] for r in rows] == ["10", "10"]


@pytest.mark.parametrize(
    "original,edited,coverage",
    [(0.1, 0.5, "False"), (0.5, 0.2, "True"), (0.5, "n/a", "n/a")],
)
def test_matlab_export_recomputes_event_coverage(tmp_path, original, edited, coverage):
    """Derive both exported coverage fields from the edited event duration."""
    recording = _load_signals(
        tmp_path,
        event_parser=lambda _: [
            {"onset": 0.1, "duration": 0, "trial_type": "start"},
            {"onset": 0.8, "duration": original, "trial_type": "cue"},
        ],
    )
    recording.events[1]["duration"] = edited
    written = _write(recording, tmp_path / "bids")

    rows = _table(written)
    assert rows[1]["duration"] == str(edited)
    assert [r["recording_covers_event"] for r in rows] == ["True", coverage]
    sidecar = json.loads(written.copy().update(extension=".json").fpath.read_text())
    diagnostics = sidecar["ConversionDiagnostics"]
    incomplete = [d for d in diagnostics if d["code"] == "incomplete_event_windows"]
    if coverage == "False":
        assert len(incomplete) == 1
        assert incomplete[0]["rows_zero_based"] == [1]
    else:
        assert incomplete == []
    assert any(d["code"] == "unknown_reference" for d in diagnostics)


def test_matlab_no_events_and_unknown_line_frequency(tmp_path):
    """Allow an explicit recording without markers or invented acquisition facts."""
    path = _save(tmp_path / "rest.mat", {"signals": np.zeros((100, 2))})
    recording = load_matlab_recording(
        path, schema=_schema(), metadata={"PowerLineFrequency": "n/a"}
    )
    assert recording.events == []
    assert recording.raw.info["line_freq"] is None
    written = _write(recording, tmp_path / "bids")
    assert not written.copy().update(suffix="events", extension=".tsv").fpath.exists()
    assert read_raw_bids(written, verbose=False).info["line_freq"] is None


def test_matlab_tremmel_missing_units_and_malformed_log(tmp_path):
    """Require calibration evidence and complete, ordered four-record groups."""
    values = _tremmel_values()
    path = _save(tmp_path / "tremmel.mat", values)
    with pytest.raises(ValueError, match="does not specify physical units"):
        load_matlab_recording(path, adapter="tremmel2024")
    values["experiment"] = values["experiment"][:-1]
    _save(path, values)
    with pytest.raises(ValueError, match="groups of four"):
        load_matlab_recording(path, adapter="tremmel2024", units="uV")


@pytest.mark.parametrize(
    "variant",
    [
        "fractional_index",
        "bad_label",
        "wrong_counts",
        "ambiguous_info",
        "units_override",
    ],
)
def test_matlab_invalid_iva(tmp_path, variant):
    """Fail on modified source encodings and ambiguous competition metadata."""
    values = _iva_values()
    if variant == "fractional_index":
        values["mrk"]["pos"][0] = 1.5
    elif variant == "bad_label":
        values["mrk"]["y"][0] = 3
    elif variant == "wrong_counts":
        values["cnt"] = values["cnt"].astype(float)
    elif variant == "ambiguous_info":
        values["info"] = values["nfo"]
    path = _save(tmp_path / "iva.mat", values)
    with pytest.raises(ValueError):
        load_matlab_recording(
            path,
            adapter="bci_competition_iii_iva",
            units="V" if variant == "units_override" else None,
        )


def test_matlab_export_protects_existing_and_appends(tmp_path):
    """Stage complete exports and retain existing participants on serial append."""
    path = _save(tmp_path / "iva.mat", _iva_values())
    recording = load_matlab_recording(path, adapter="bci_competition_iii_iva")
    first = _write(recording, tmp_path / "bids")
    before = {
        str(p.relative_to(first.root)): p.read_bytes()
        for p in first.root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        _write(recording, first.root)
    after = {
        str(p.relative_to(first.root)): p.read_bytes()
        for p in first.root.rglob("*")
        if p.is_file()
    }
    assert before == after
    _write(recording, first.root, subject="02")
    participants = (first.root / "participants.tsv").read_text()
    assert "sub-01" in participants and "sub-02" in participants
    assert _table(first)[-1]["value"] == "n/a"


def test_matlab_export_failure_is_not_published(tmp_path, monkeypatch):
    """A native writer failure must not expose incomplete converted recordings."""
    path = _save(tmp_path / "iva.mat", _iva_values())
    recording = load_matlab_recording(path, adapter="bci_competition_iii_iva")

    def broken(*args, **kwargs):
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr("aeon_neuro.utils.bids.write_raw_bids", broken)
    with pytest.raises(RuntimeError, match="simulated"):
        _write(recording, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "mutation", ["sample", "duration", "description", "crop", "metadata", "frequency"]
)
def test_matlab_revalidate_before_export(tmp_path, mutation):
    """Do not publish tables that became inconsistent after the source was read."""
    path = _save(tmp_path / "iva.mat", _iva_values())
    recording = load_matlab_recording(path, adapter="bci_competition_iii_iva")
    if mutation == "sample":
        recording.events[0]["sample"] = 1
    elif mutation == "duration":
        recording.events[0]["duration"] = -1
    elif mutation == "description":
        recording.event_descriptions.pop("trial_type")
    elif mutation == "crop":
        recording.raw.crop(tmin=1)
    elif mutation == "metadata":
        recording.metadata["SamplingFrequency"] = 1000
    else:
        recording.raw.info["line_freq"] = 50
    with pytest.raises(ValueError):
        _write(recording, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("known_filters", [False, True])
@pytest.mark.parametrize(
    "processing",
    ["bandpass", "partial_bandpass", "notch", "reference", "scale", "filter_settings"],
)
def test_matlab_reject_processed_signals(tmp_path, known_filters, processing):
    """Reject processing even when MNE's global filter cutoffs stay unchanged."""
    recording = _load_signals(
        tmp_path,
        np.random.default_rng(42).normal(size=(2000, 2)),
        channel_names=["C3", "C4"],
        channel_types="eeg",
        units="uV",
        highpass=0.5 if known_filters else None,
        lowpass=45 if known_filters else None,
    )
    raw = recording.raw
    if processing == "bandpass":
        raw.filter(1, 30, method="iir", verbose=False)
    elif processing == "partial_bandpass":
        raw.filter(1, 30, picks=["C3"], method="iir", verbose=False)
    elif processing == "notch":
        raw.notch_filter(20, method="iir", verbose=False)
    elif processing == "reference":
        raw.set_eeg_reference("average", verbose=False)
    elif processing == "scale":
        raw.apply_function(lambda data: data * 2, verbose=False)
    else:
        with raw.info._unlock():
            raw.info["highpass"] = 1.0
    with pytest.raises(ValueError, match="signals or filter settings changed"):
        _write(recording, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_matlab_preprocess_copy_and_mark_bad_channels(tmp_path):
    """Keep original export usable when a separate copy is filtered for analysis."""
    recording = _load_signals(
        tmp_path,
        np.random.default_rng(42).normal(size=(2000, 2)),
        highpass=0.5,
        lowpass=45,
    )
    expected = recording.raw.get_data()
    analysis = recording.raw.copy().filter(1, 30, method="iir", verbose=False)
    assert not np.array_equal(analysis.get_data(), expected)
    recording.raw.info["bads"] = ["Cz"]
    written = _write(recording, tmp_path / "bids")
    restored = read_raw_bids(written, verbose=False)
    assert_allclose(restored.get_data(), expected, rtol=2e-7, atol=1e-12)
    assert restored.info["bads"] == ["Cz"]
    channels = _table(written, "channels")
    assert {c["low_cutoff"] for c in channels} == {"0.5"}
    assert {c["high_cutoff"] for c in channels} == {"45.0"}


@pytest.mark.parametrize("count,rate", [(150, 100), (1500, 200.0055)])
def test_matlab_brainvision_preserves_fractional_duration(tmp_path, count, rate):
    """Retain every sample without padding fractional-second recordings."""
    recording = _load_signals(
        tmp_path,
        np.random.default_rng(42).normal(size=(count, 2)),
        sfreq=rate,
        event_time_unit="samples",
        event_parser=lambda _: [
            {"onset": count - 1, "duration": 0, "trial_type": "end"}
        ],
    )
    written = _write(recording, tmp_path / "bids")
    restored = read_raw_bids(written, verbose=False)
    assert restored.n_times == count
    assert_allclose(restored.info["sfreq"], rate, rtol=1e-12)
    assert_allclose(
        restored.get_data(), recording.raw.get_data(), rtol=2e-7, atol=1e-12
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"format": "BrainVision"},
        {"format": "BrainVision", "allow_preload": True, "overwrite": True},
        {"events": np.array([[0, 0, 1]])},
        {"event_id": {"cue": 1}},
    ]
    + [
        {"format": encoding, "allow_preload": True}
        for encoding in ("auto", "EDF", "BDF", "EEGLAB", "FIF")
    ],
)
def test_matlab_export_requires_explicit_options(tmp_path, monkeypatch, options):
    """Reject invalid options and unsupported formats before reaching the writer."""
    path = _save(tmp_path / "iva.mat", _iva_values())
    recording = load_matlab_recording(path, adapter="bci_competition_iii_iva")

    def unexpected_writer(*args, **kwargs):
        pytest.fail("Invalid export options reached the signal writer.")

    monkeypatch.setattr("aeon_neuro.utils.bids.write_raw_bids", unexpected_writer)
    match = (
        "format='BrainVision'"
        if options.get("format", "BrainVision") != "BrainVision"
        else None
    )
    with pytest.raises(ValueError, match=match):
        convert_eeg_to_bids(
            recording,
            BIDSPath(root=tmp_path / "output", subject="01", task="mi"),
            **options,
        )
    assert not (tmp_path / "output").exists()
