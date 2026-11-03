# Define a classification problem on loaded recordings

The [format reference](problem-format.md) defines every supported problem/binding
field, required/default rule and runtime input constraint. This guide introduces
the format through usage examples.

To define problems before signal files are downloaded, start with
[metadata-only planning and deferred materialization](problem-planning.md).
The definition and binding vocabulary below is shared by both routes; this page
describes the eager convenience function for arrays already in memory.

`load_classification_problem` adds a small experimental definition layer over
the existing case builders. Start with recording arrays and aligned metadata,
including the outputs of `load_bids_data`. Supply the choices that the recordings
do not determine: class meanings, one-case timing, grouping and optional channels.

The result is the existing `ClassificationDataset`, ready for explicit assignment
validation and an ordinary classifier. This first format supports fixed windows
and fixed event-relative epochs. It adds no dependencies and does not discover
files, preprocess signals or choose a train/test split.

## A problem and its dataset binding

Keep the prediction question separate from physical source columns and codes.
The same problem can use a different binding on another suitable dataset; the
same recordings can support another question with a different problem identity.
Bindings must reference the exact problem ID and version.

This example defines half-second post-cue cases from loaded recordings:

```python
from aeon_neuro.datasets import load_classification_problem

problem = {
    "format_version": 1,
    "id": "left-right-imagery",
    "version": "1",
    "question": "Classify imagined left versus right movement from post-cue EEG",
    "class_labels": ["left_hand", "right_hand"],
    "channels": ["C3", "C4"],
    "segmentation": {"mode": "event_epochs", "tmin": 0.0, "tmax": 0.5},
}
binding = {
    "format_version": 1,
    "problem": {"id": "left-right-imagery", "version": "1"},
    "target": {
        "source": "event", "column": "condition",
        "mapping": {"11": "left_hand", "22": "right_hand"},
    },
    "group": {"source": "recording", "column": "subject"},
    "case_fields": {
        "participant_id": {"source": "recording", "column": "subject"},
        "session": {"source": "recording", "column": "session"},
    },
    "events": {"onset": "cue_seconds", "id": "trial_id"},
}
dataset = load_classification_problem(
    recordings, metadata, problem, binding, events=event_rows
)
```

Each recording is a `(channel, time)` NumPy array. Metadata has a unique
`recording_id` or `filepath`, ordered `channel_names` and `sampling_frequency`.
The extra `subject` and `session` columns above are explicit binding inputs.
Keep known units, reference and prior processing in recording metadata; the
loader preserves that information without inferring or harmonising it.

Each event supplies `recording_id`, the bound onset and a recording-local event
ID, plus the target and any bound case fields. Numeric source codes such as `11`
are compared by their string spelling to JSON mapping keys. Source `11` and
`"11"` therefore use the same mapping, while `"011"` is distinct. Output classes
are nonempty strings; the mapping must cover exactly the declared vocabulary.
Observed cases need not contain every class, so check class coverage when
validating an evaluation. `class_labels` records the declared vocabulary order;
use the fitted classifier's `classes_` for probability-column alignment.

All selectors explicitly name `source="recording"` or `source="event"` and a
`column`. Recording fields are read from the matched metadata entry, never an
event field of the same name. `case_fields` copies explicit identity/provenance
fields onto cases for downstream split checks. They must be nonmissing string
or integer values and cannot overwrite generated bounds, labels or identities.
Participant/session/trial roles are not inferred from filenames or group names.

## Dictionaries and JSON files

Both documents accept ordinary dictionaries or UTF-8 JSON filenames:

```python
dataset = load_classification_problem(
    recordings, metadata,
    "imagery.problem.json", "imagery.binding.json",
    events=event_rows,
)
```

Unknown fields, unsupported versions, duplicate JSON keys, nonfinite settings,
missing columns and invalid mappings raise. There is no executable Python,
dynamic import or remote schema retrieval in the documents. Format 1 is an
initial experimental contract, not compatibility with other recipe formats.

## Fixed windows over recording targets

For a cohort-labelled dataset, use a recording target and fixed-window rule:

```python
problem = {
    "format_version": 1, "id": "cohort-windows", "version": "1",
    "question": "Classify the declared cohort using one-second EEG windows",
    "class_labels": ["control", "condition"],
    "segmentation": {"mode": "fixed_windows", "window_size": 1.0},
}
binding = {
    "format_version": 1,
    "problem": {"id": "cohort-windows", "version": "1"},
    "target": {
        "source": "recording", "column": "cohort",
        "mapping": {"control": "control", "condition": "condition"},
    },
    "group": {"source": "recording", "column": "subject"},
    "case_fields": {
        "participant_id": {"source": "recording", "column": "subject"},
    },
}
dataset = load_classification_problem(recordings, metadata, problem, binding)
```

`fixed_windows` requires `window_size` in seconds. Its optional fields match
`make_fixed_length_cases`: `step_size` defaults to `window_size`, `start` to
zero and `stop` to the recording end. A requested stop is capped at the source
end; trailing partial windows are omitted. Events and event selectors are
rejected in this mode. Window source intervals and case IDs follow the existing
helper, and windows retain recording order followed by time order.

## Epoch timing and exclusions

`event_epochs` requires `tmin` and `tmax` in seconds relative to the bound onset.
The interval is `[onset + tmin, onset + tmax)`, matching `make_event_cases`.
All onsets and offsets must lie on the supplied recording's sample grid.
Numeric TSV onset strings are accepted; no silent rounding is performed.
Onsets are relative to the first sample of the supplied array. Resolve MNE
`first_samp`, acquisition-absolute samples, crops and one-based event conventions
before supplying them. The loader does not infer a coordinate conversion.

`on_out_of_bounds` defaults to `"raise"`. Set it to `"exclude"` in the epoch
segmentation to retain out-of-bounds candidates with reason `outside_recording`.
Epochs are never clipped or padded. Event IDs must be unique per recording,
including excluded candidates. Their structured recording/event identity is
stable when event rows are reordered. `source_event_row` is the zero-based input
row position; `source_event` preserves the original complete row. Event output
order follows the supplied table rather than sorted IDs.

Both target bindings default to `on_unmapped="raise"`. Explicit `"exclude"`
retains candidates with unrecognised target codes and reason `unmapped_target`.
Missing columns and invalid values still raise. Source adapters may also supply
an event's boolean `included=False` and a nonempty `exclusion_reason`, for
example after checking an independent quality annotation. Such rows can omit
the target but must retain valid timing, group and bound identities. This first
version consumes those eligibility decisions; it does not execute a quality
detector or a declarative annotation-query language. All-excluded construction
raises through the existing builder.

## Resolved records and evaluation

`dataset.metadata["problem"]` retains independent effective documents and their
SHA256 hashes. Effective documents include the defaults described above; actual
channels, rate, included cases and exclusions remain in the standard dataset
fields. Input arrays, metadata, events and definitions are not modified.

These hashes identify configuration, not recording contents or a completed
experiment. Case bounds describe the supplied arrays; preserve canonical source
identities and full filter/sequence support separately when those differ.
The loader does not authenticate source files or validate earlier processing.
It also does not promise lazy access or a process-memory limit: recordings are
already loaded, and the ordinary case builder copies the extracted arrays.

Use [assignment validation](classification-problems.md) before fitting. Explicit
groups identify the intended grouping unit; they do not themselves generate or
enforce an evaluation protocol. Learned transformations belong within training
partitions. Successful construction cannot establish scientific validity or
equivalence with another dataset, paper or MOABB paradigm.

Run `python examples/problem_loading.py` for a complete generated example. It
loads JSON definitions, constructs two event targets on the same intervals and
a cohort-window problem, validates a participant-held-out split and fits an aeon
dummy classifier. It downloads no data and makes no EEG performance claim.
