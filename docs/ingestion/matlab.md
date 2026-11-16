# Continuous MATLAB recordings

`load_matlab_recording` uses an explicit schema or adapter to read continuous,
two-dimensional voltage signals from pre-7.3 or HDF5-based MATLAB 7.3 files.
MATLAB itself is not required. Original dimensions are checked before restoring
squeezed channel axes; epochs and discontinuities are unsupported. Recordings
are loaded into memory.

Install the reader and BrainVision exporter:

```sh
pip install 'aeon-neuro[eeg_io]'
```

## Tremmel workload data

The adapter follows the [dataset README](https://doi.org/10.5258/SOTON/D3057).
Confirm units for all 30 physiological columns; the README does not specify them.

```python
from mne_bids import BIDSPath
from aeon_neuro.utils.matlab import MatlabSchema, load_matlab_recording
from aeon_neuro.utils.bids import convert_eeg_to_bids

recording = load_matlab_recording(
    "source/subj001.mat", adapter="tremmel2024", units="uV",
)
written = convert_eeg_to_bids(
    recording,
    BIDSPath(root="workload-bids", subject="001", task="workload"),
    format="BrainVision",
    allow_preload=True,
)
```

The LSL timestamp column is excluded from signals. Source channel order is
retained: ECG, two EOG, 26 scalp EEG and in-ear EEG channels. Both tasks remain
in one continuous recording. Supply subject/session/run identifiers in `BIDSPath`.

Each four-record trial group becomes an event row preserving trial/block IDs,
n-back conditions, stimuli, arithmetic operands, responses and all four source
timestamps. `source_log_row` traces the original group. Missing responses retain
`-1` with `response_time=n/a`; experiment starts are stored in
`MatlabSource.AdapterContext` in the EEG JSON. Diagnostics report out-of-span
behavioural timestamps and unexpected counts, including 162 n-back trials in
some subjects versus the published 144. Every group is retained.

The adapter fits an endpoint-anchored clock near 200 Hz without interpolating
samples: at most 1% rate disagreement and 0.25-sample residual, with nonmonotonic
timestamps or gaps over 1.5 nominal periods rejected. Provenance records clock
decisions and the source SHA-256; original per-sample timestamps remain in MAT.
The reported 0.5–100 Hz bandpass and 50 Hz notch describe existing processing.
These are the released processed matrices, not reconstructed acquisition data.

## BCI Competition III IVa

```python
recording = load_matlab_recording(
    "source/data_set_IVa_aa.mat", adapter="bci_competition_iii_iva",
)
```

Export with the same converter, supplying an appropriate destination `BIDSPath`.
The [IVa adapter](https://www.bbci.de/competition/iii/desc_IVa.html) reads INT16
`cnt` at 0.1 µV/count, one-based `mrk.pos`, labels `mrk.y`, and `nfo.fs`/`nfo.clab`.
`info` is accepted instead of `nfo`; providing both fails. Class names use
`mrk.className`, then `nfo.classes`, then the published right/foot order.
Withheld labels remain `value=n/a`, `trial_type=unknown`; cue duration is 3.5 s.
The file's sampling rate is retained, including the 100 Hz release. Projected
2D electrode positions are not converted to physical 3D coordinates. This is
separate from aeon-neuro's `MotorImagery` benchmark, which contains epoched ECoG.

## Another MATLAB layout

Describe a top-level matrix with `MatlabSchema`. The event parser receives the
decoded dictionary, including nested structures, and returns scalar rows with
`onset`, `duration` in seconds (or `"n/a"`), and `trial_type`. Describe additional
columns through `event_descriptions`.

```python
def parse_events(mat):
    return [
        {"onset": sample, "duration": 2.0, "trial_type": label}
        for sample, label in zip(mat["markers"]["pos"], mat["markers"]["label"])
    ]

schema = MatlabSchema(
    signal_key="signals",
    sfreq=250,
    channel_names=["C3", "C4", "EOG"],
    channel_types=["eeg", "eeg", "eog"],
    units="uV",
    sample_axis=0,
    event_parser=parse_events,
    event_time_unit="samples",
    event_index_base=1,
)
recording = load_matlab_recording("source/custom.mat", schema=schema)
```

The schema also documents per-channel units/scales, column selection, timestamp
columns/vectors, and nominal or fitted clocks. Seconds or absolute timestamps
can replace sample indices. Bounds use the source span, allowing only numerical
roundoff outside it; rounded onsets retain source coordinates and `timing_error`.
`highpass`/`lowpass` describe existing filters. Supply verified acquisition fields
with `metadata={"EEGReference": "..."}`. Unknown reference, cutoffs, manufacturer
and channel quality remain `n/a`; coordinates and wall-clock dates are not inferred.

Reusable adapters are caller-supplied Python functions
`adapter(decoded_mat, *, units)` returning a `MatlabSchema`. Add source-contract
documentation and generated pre-7.3/7.3 fixtures with calibration, event and BIDS
read-back checks, following `test_matlab.py`.

## Export and validation

Pass the complete `MatlabRecording` to preserve events, metadata and provenance.
Its `raw` holds volts and has no annotations. MATLAB export requires
`format="BrainVision"`, `allow_preload=True`, and `overwrite=False`; event overrides
are rejected. Keep samples and filter settings unchanged, and preprocess
`recording.raw.copy()` for analysis. Marking bad channels is supported.

Output is staged before publication; existing recordings and conflicting shared
metadata are protected. Sequential appends require agreeing metadata. Publishing
is not a filesystem transaction: inspect interrupted moves and avoid concurrent
appends. BrainVision float32 encoding introduces small numerical rounding.

Read `events.tsv` as the authoritative table: durations are never clipped and
signals are never padded. Incomplete windows retain their duration with
`recording_covers_event=False` and a diagnostic, although MNE annotations may
truncate them on read-back. Native markers use generated category codes; original
codes remain in `value`. Classification windows, exclusions and splits are downstream.

Review diagnostics and scientific metadata, then run the official BIDS validator:

```sh
deno run -ERWN jsr:@bids/validator workload-bids
```
