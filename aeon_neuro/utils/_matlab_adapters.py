"""Explicit layouts for continuous Tremmel and BCI Competition III IVa data."""

import warnings

import numpy as np

from aeon_neuro.utils.matlab import MatlabSchema

_TREMMEL_CHANNELS = [
    "ECG",
    "HEOG",
    "VEOG",
    "FP1",
    "F7",
    "Fz",
    "F3",
    "FT9",
    "T7",
    "C3",
    "Cz",
    "CP1",
    "TP9",
    "P7",
    "P3",
    "Pz",
    "O1",
    "FP2",
    "F8",
    "FT10",
    "F4",
    "C4",
    "T8",
    "CP2",
    "TP10",
    "P4",
    "P8",
    "O2",
    "Oz",
    "Inear",
]


def _tremmel2024(mat, *, units):
    """Interpret the released workload matrices, retaining every trial group."""
    if units is None:
        raise ValueError(
            "The Tremmel README does not specify physical units; supply units "
            "explicitly for all 30 signals."
        )
    data = np.asarray(mat.get("data"))
    log = np.asarray(mat.get("experiment"))
    if data.ndim != 2 or data.shape[1] != 31:
        raise ValueError(
            "Tremmel data must have 31 columns (timestamps plus 30 signals)."
        )
    if (
        log.ndim != 2
        or log.shape[1] != 2
        or log.dtype.kind not in "fiu"
        or not np.isfinite(log).all()
    ):
        raise ValueError("Tremmel experiment must be a finite two-column log.")
    if np.any(np.diff(log[:, 0]) < 0) or np.any(log[:, 1] != np.rint(log[:, 1])):
        raise ValueError(
            "Tremmel log needs ordered timestamps and integer experiment values."
        )
    starts = np.flatnonzero(log[:, 1] == -99)
    if len(starts) != 2 or starts[0] != 0:
        raise ValueError(
            "Expected two Tremmel experiment starts, with the first at log row 1."
        )
    rows, findings, counts = [], [], []
    for task_number, (start, stop) in enumerate(zip(starts, [starts[1], len(log)])):
        block = log[start + 1 : stop]
        if not len(block) or len(block) % 4:
            raise ValueError(
                "Tremmel trials require complete groups of four log records."
            )
        trials = block.reshape(-1, 4, 2)
        counts.append(len(trials))
        task = "nback" if task_number == 0 else "arithmetic"
        for number, trial in enumerate(trials):
            trial_id, first, second, response = trial[:, 1].astype(int)
            if trial_id != number % 18:
                raise ValueError(
                    "Tremmel trial IDs must repeat 0 through 17 in source order."
                )
            if task_number == 0 and (
                not 0 <= first <= 3
                or not 0 <= second <= 9
                or response not in (-1, 0, 1)
            ):
                raise ValueError("Invalid n-back condition, stimulus or response code.")
            if task_number == 1 and (first <= 0 or second <= 0 or response < -1):
                raise ValueError("Invalid arithmetic operands or response code.")
            row = {
                "onset": float(trial[0, 0]),
                "duration": 2.5 if task_number == 0 else 8.5,
                "trial_type": task,
                "trial_id": int(trial_id),
                "block_index": number // 18 + 1,
                "source_log_row": int(start + 2 + number * 4),
                "response": int(response),
                "response_time": (
                    float(trial[3, 0] - trial[0, 0]) if response != -1 else "n/a"
                ),
                "condition_timestamp": float(trial[1, 0]),
                "stimulus_timestamp": float(trial[2, 0]),
                "response_timestamp": float(trial[3, 0]),
                "nback_level": int(first) if task_number == 0 else "n/a",
                "stimulus": int(second) if task_number == 0 else "n/a",
                "operand1": int(first) if task_number == 1 else "n/a",
                "operand2": int(second) if task_number == 1 else "n/a",
            }
            rows.append(row)
        if len(trials) != 144:
            findings.append(
                {
                    "code": "unexpected_trial_count",
                    "task": task,
                    "observed": len(trials),
                    "published": 144,
                    "message": (
                        f"{task} has {len(trials)} trial groups, versus 144 in "
                        "the publication; all are retained."
                    ),
                }
            )
    outside = np.flatnonzero((log[:, 0] < data[0, 0]) | (log[:, 0] > data[-1, 0]))
    if len(outside):
        findings.append(
            {
                "code": "log_outside_signal",
                "source_rows_one_based": (outside + 1).tolist(),
                "message": (
                    "Some log timestamps fall outside the signal span; start markers "
                    "and behavioural timestamps are retained as source metadata."
                ),
            }
        )
    for finding in findings:
        warnings.warn(finding["message"], UserWarning, stacklevel=3)
    descriptions = {
        "trial_id": "Source trial ID, 0 to 17, repeated for each block.",
        "block_index": (
            "One-based block inferred from the validated 18-trial sequence, "
            "separately within each task."
        ),
        "source_log_row": (
            "One-based experiment matrix row of the trial-start record; the next "
            "three rows provide condition, stimulus and response."
        ),
        "response": (
            "Source response: n-back 0=target, 1=non-target; arithmetic=entered sum. "
            "-1 means no response."
        ),
        "response_time": {
            "Description": "Response timestamp minus trial onset; n/a for no response.",
            "Units": "s",
        },
        "nback_level": "Source n-back difficulty (0 to 3); n/a for arithmetic.",
        "stimulus": "Displayed n-back digit (0 to 9); n/a for arithmetic.",
        "operand1": "First arithmetic operand; n/a for n-back.",
        "operand2": "Second arithmetic operand; n/a for n-back.",
    }
    for column, meaning in [
        ("condition_timestamp", "n-back level or first operand"),
        ("stimulus_timestamp", "digit or second operand"),
        ("response_timestamp", "response, including timeout"),
    ]:
        descriptions[column] = {
            "Description": f"Original LSL timestamp of {meaning} log record.",
            "Units": "s",
        }
    return MatlabSchema(
        signal_key="data",
        sfreq=200,
        channel_names=_TREMMEL_CHANNELS,
        channel_types=["ecg", "eog", "eog"] + ["eeg"] * 27,
        units=units,
        highpass=0.5,
        lowpass=100,
        signal_columns=tuple(range(1, 31)),
        timestamp_column=0,
        timestamp_policy="fit",
        event_parser=lambda _: rows,
        event_time_unit="timestamps",
        event_descriptions=descriptions,
        metadata={
            "TaskDescription": (
                "N-back and mental arithmetic workload tasks in one continuous "
                "released recording."
            ),
            "PowerLineFrequency": 50,
            "SoftwareFilters": {
                "Reported bandpass": {"LowCutoffHz": 0.5, "HighCutoffHz": 100},
                "Reported notch": {"FrequencyHz": 50},
            },
        },
        provenance={
            "SourceReference": "https://doi.org/10.5258/SOTON/D3057",
            "SourceProcessing": (
                "Dataset README reports 200 Hz, 0.5-100 Hz bandpass and 50 Hz notch. "
                "Scalp channels have not been re-referenced for analysis. Acquisition"
                " reference is unspecified. No further processing is applied by this "
                "adapter."
            ),
            "UnitsEvidence": (
                "Explicit caller-supplied calibration; units are absent from the "
                "dataset README."
            ),
            "ExperimentStarts": [
                {
                    "SourceRowOneBased": int(i + 1),
                    "Timestamp": float(log[i, 0]),
                    "Value": -99,
                }
                for i in starts
            ],
            "TrialCounts": dict(zip(["nback", "arithmetic"], counts)),
            "Diagnostics": findings,
        },
    )


def _bci_competition_iii_iva(mat, *, units):
    """Read the documented continuous INT16/counts IVa release and its markers."""
    if units is not None:
        raise ValueError(
            "IVa calibration is fixed at 0.1 uV/count; use a custom schema for "
            "modified data."
        )
    info_keys = [key for key in ("nfo", "info") if key in mat]
    if len(info_keys) != 1 or not isinstance(mat.get("mrk"), dict):
        raise ValueError("IVa requires mrk and exactly one of the nfo/info structures.")
    info, markers = mat[info_keys[0]], mat["mrk"]
    if (
        not isinstance(info, dict)
        or not {"fs", "clab"} <= info.keys()
        or not {"pos", "y"} <= markers.keys()
    ):
        raise ValueError("IVa needs fs/clab metadata and pos/y markers.")
    signal = np.asarray(mat.get("cnt"))
    if signal.ndim != 2 or signal.dtype != np.dtype("int16"):
        raise ValueError(
            "IVa cnt must be a continuous INT16 matrix with documented 0.1 "
            "uV/count calibration."
        )
    positions = np.atleast_1d(markers["pos"])
    labels = np.atleast_1d(markers["y"])
    if (
        positions.ndim != 1
        or labels.ndim != 1
        or len(labels) != len(positions)
        or labels.dtype.kind not in "fiu"
        or np.isinf(labels).any()
        or not np.isin(labels[np.isfinite(labels)], [1, 2]).all()
    ):
        raise ValueError("IVa pos/y must be aligned vectors with labels 1, 2 or NaN.")
    classes = markers.get("className", info.get("classes", ["right", "foot"]))
    classes = list(np.atleast_1d(classes))
    if (
        len(classes) != 2
        or any(not isinstance(c, str) or not c.strip() for c in classes)
        or len(set(classes)) != 2
        or "unknown" in classes
    ):
        raise ValueError("IVa class names must identify two distinct labelled classes.")
    rows = [
        {
            "onset": p,
            "duration": 3.5,
            "trial_type": "unknown" if np.isnan(label) else classes[int(label) - 1],
            "value": label,
        }
        for p, label in zip(positions, labels)
    ]
    return MatlabSchema(
        signal_key="cnt",
        sfreq=float(info["fs"]),
        channel_names=list(np.atleast_1d(info["clab"])),
        channel_types="eeg",
        units="uV",
        scale=0.1,
        event_parser=lambda _: rows,
        event_time_unit="samples",
        event_index_base=1,
        metadata={
            "Manufacturer": "Brain Products",
            "ManufacturersModelName": "BrainAmp",
            "EEGPlacementScheme": "extended 10-20",
            "TaskDescription": (
                "BCI Competition III IVa right-hand and foot motor imagery; unknown "
                "preserves withheld source labels."
            ),
        },
        provenance={
            "SourceReference": "https://www.bbci.de/competition/iii/desc_IVa.html",
            "SourceProcessing": (
                "Published acquisition bandpass 0.05-200 Hz, 1000 Hz digitisation; an"
                " alternative 100 Hz downsampled release exists. Output uses the "
                "file's declared fs; no filtering or resampling is added."
            ),
            "CalibrationEvidence": "Published INT16 counts: 0.1 microvolt per count.",
            "ElectrodePositions": (
                "2D projected positions are not converted into physical 3D "
                "coordinates."
            ),
        },
    )


_ADAPTERS = {
    "tremmel2024": _tremmel2024,
    "bci_competition_iii_iva": _bci_competition_iii_iva,
}
