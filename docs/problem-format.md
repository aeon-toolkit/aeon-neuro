# Define classification problems on loaded recordings

A **problem** describes the prediction question and the construction of one
case. A **binding** connects that problem to explicit recording/event fields.
`load_classification_problem` applies them to arrays already in memory, including
outputs from `load_bids_data`. The version-1 format is experimental. It adds no
dependencies and does not discover files, preprocess signals or choose a split.

For BIDS sources loaded with `load_bids_dataset`, the experimental
[object-based builder](bids-dataset-info.md#make-a-classification-problem)
accepts the same authored documents as `make_classification_problem(X, info,
problem, binding)`, supplying events and preserving source/history provenance.

The same case rules also support [metadata-only planning and selected loading](problem-planning.md)
before signals are available. Both plain catalogues and shared BIDS information
objects can supply the planning inputs.

## Shared information and provenance

`make_classification_problem(X, info, problem, binding)` uses the shared
`EEGDatasetInfo` object, including current recording layouts and events. It adds
resolved definition/binding hashes to the base source/history/construction
provenance under `dataset.metadata["provenance"]["problem"]`. The plain-input
`load_classification_problem` keeps its separate `metadata["problem"]` contract.
Both routes reuse the existing case builders and retain explicit scientific
choices. See [shared EEG information](eeg-information.md) for source/case
selection, processing, optional estimator metadata and the schema-independent
case API. Authored documents are an additional convenience, not a prerequisite
for case information.

## Complete event problem and binding

This runnable example makes half-second post-cue cases from generated signals:

```python
import numpy as np
from aeon_neuro.datasets import load_classification_problem

recordings = np.arange(64, dtype=float).reshape(2, 2, 16)
metadata = [
    {"recording_id": f"R{i}", "subject": f"P{i}",
     "channel_names": ["C3", "C4"], "sampling_frequency": 4}
    for i in range(2)
]
event_rows = [
    {"recording_id": f"R{i}", "cue_seconds": onset,
     "trial_id": trial, "condition": code}
    for i in range(2)
    for onset, trial, code in [(0.5, "T1", 11), (2.5, "T2", 22)]
]
problem = {
    "format_version": 1, "id": "left-right-imagery", "version": "1",
    "question": "Classify declared left versus right imagery from post-cue EEG",
    "class_labels": ["left_hand", "right_hand"], "channels": ["C3", "C4"],
    "segmentation": {"mode": "event_epochs", "tmin": 0.0, "tmax": 0.5},
}
binding = {
    "format_version": 1,
    "problem": {"id": "left-right-imagery", "version": "1"},
    "target": {"source": "event", "column": "condition",
               "mapping": {"11": "left_hand", "22": "right_hand"}},
    "group": {"source": "recording", "column": "subject"},
    "case_fields": {"participant_id": {"source": "recording", "column": "subject"}},
    "events": {"onset": "cue_seconds", "id": "trial_id"},
}
dataset = load_classification_problem(
    recordings, metadata, problem, binding, events=event_rows
)
assert dataset.X.shape == (4, 2, 2)
assert dataset.y.tolist() == ["left_hand", "right_hand"] * 2
```

The documents can also be UTF-8 JSON filenames. Run
`python examples/problem_loading.py` for two targets on the same intervals,
a cohort problem and a participant-held-out assignment check. All inputs are
generated; the example downloads no data and makes no EEG performance claim.

## Fixed windows over recording targets

For a cohort question, supply an explicit cohort attribute and change the rule:

```python
cohort_metadata = [
    {**row, "cohort": cohort}
    for row, cohort in zip(metadata, ["control", "condition"], strict=True)
]
cohort_problem = {
    "format_version": 1, "id": "cohort-windows", "version": "1",
    "question": "Classify the declared cohort using one-second EEG windows",
    "class_labels": ["control", "condition"],
    "segmentation": {"mode": "fixed_windows", "window_size": 1.0},
}
cohort_binding = {
    "format_version": 1,
    "problem": {"id": "cohort-windows", "version": "1"},
    "target": {"source": "recording", "column": "cohort",
               "mapping": {"control": "control", "condition": "condition"}},
    "group": {"source": "recording", "column": "subject"},
}
cohort = load_classification_problem(
    recordings, cohort_metadata, cohort_problem, cohort_binding
)
assert cohort.X.shape == (8, 2, 4)
```

The result is the existing `ClassificationDataset`: `X`, `y`, `groups`, `cases`,
`exclusions` and metadata. Class meanings, source eligibility and evaluation
groups remain explicit. Use [assignment validation](classification-problems.md)
before fitting; grouping fields alone do not enforce an evaluation protocol.

The following tables are the complete version-1 field reference.

## Document rules and versions

Both documents require integer `format_version: 1`. This identifies the format;
the problem's string `id` and `version` identify a particular authored problem.
The binding must reference that exact pair. No version migration or implicit
latest version is applied.

All fields allowed in each fixed object are listed below. Unknown fields,
missing required fields, unsupported modes, duplicate JSON keys and nonfinite
numbers are rejected. `target.mapping` and `case_fields` have explicitly variable
keys. There is no `$schema` property or remote schema retrieval in this format;
validation is implemented in Python, without a standalone JSON Schema file.

Object-field errors name the affected object and list all missing and unsupported
keys in sorted order. A typo can appear in both lists. A non-dictionary nested
value instead reports the required object type.

The tables use JSON types. Python dictionaries are normalized through JSON;
use ordinary JSON-compatible values. Text must contain a non-whitespace
character. Its spelling, case and surrounding whitespace are preserved rather
than trimmed or case-folded. Boolean values do not stand in for integer
identifiers or numeric timing settings.

**Required** means the field must be supplied. **Default** describes an omitted
optional field. Explicit `null` is permitted only where stated. Defaults are
retained in the effective documents returned with the constructed dataset.

## Problem document

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `problem.format_version` | Required integer; exactly `1`. | Format identifier, distinct from the problem's own version. |
| `problem.id` | Required nonempty string. | Authored problem identity. No slug pattern or filename inference is imposed. |
| `problem.version` | Required nonempty string. | Authored problem revision; the binding must match it exactly. |
| `problem.question` | Required nonempty string. | Human-readable prediction question. Its scientific meaning is not validated. |
| `problem.class_labels` | Required array of at least two unique nonempty strings. | Declared class vocabulary. Every name must appear among the binding's mapping values, even if a particular construction observes fewer classes. |
| `problem.channels` | Optional nonempty array of unique nonempty strings; default is each recording's full channel list. | Explicit source channel names in output order. Requested names must exist in included sources. This selects/reorders channels without renaming, interpolation or resampling. |
| `problem.segmentation` | Required object. | Exactly one of the following two mode-specific forms. |

Declared class order is retained as metadata. It is not an estimator's
probability-column order; use the fitted estimator's `classes_` for that purpose.
Observed-class coverage is checked separately when validating an evaluation.

### Fixed-window segmentation

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `problem.segmentation.mode` | Required string; exactly `"fixed_windows"`. | Construct complete windows with recording-level targets and groups. |
| `problem.segmentation.window_size` | Required finite number. | Window duration in seconds; must span at least one sample on each source grid. |
| `problem.segmentation.step_size` | Optional finite number or `null`; default equals `window_size`. | Step between window starts, spanning at least one sample. Explicit `null` also uses `window_size` during construction. Smaller steps allow overlapping cases. |
| `problem.segmentation.start` | Optional finite number; default `0.0`. | Nonnegative start in seconds from the first supplied sample. |
| `problem.segmentation.stop` | Optional finite number or `null`; default `null`. | Exclusive end in seconds. A number must exceed `start` and is capped at the recording end; `null` uses the actual end. |

All numeric times must lie on each recording's sample grid. Window starts follow
recording order and then increasing sample position. Only complete windows are
generated: a short recording or partial tail contributes no shortened/padded
case and no exclusion row. At least one included case must remain overall.

This mode accepts only `source="recording"` selectors. `binding.events` and the
loader's `events` argument must be absent. It does not interpret annotation
states or expose an event-level eligibility policy.

### Event-epoch segmentation

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `problem.segmentation.mode` | Required string; exactly `"event_epochs"`. | Construct one candidate per supplied event row. |
| `problem.segmentation.tmin` | Required finite number. | Inclusive start offset in seconds from the bound event onset; negative values are allowed. |
| `problem.segmentation.tmax` | Required finite number, greater than `tmin`. | Exclusive end offset from that onset. The event's `duration` does not supply an automatic epoch length. |
| `problem.segmentation.on_out_of_bounds` | Optional `"raise"` or `"exclude"`; default `"raise"`. | Raise for an included candidate outside the recording, or retain it as an exclusion with reason `"outside_recording"`. |

For each event, the interval is `[onset + tmin, onset + tmax)`. Offsets and onsets
must lie on the source grid; negative offsets and wholly pre-event intervals
are permitted when their bounds are valid. Epochs are never clipped or padded.
Both event- and recording-level selectors are available in this mode.

## Binding document

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `binding.format_version` | Required integer; exactly `1`. | Binding format identifier. |
| `binding.problem` | Required object containing exactly `id` and `version`. | Its values must equal `problem.id` and `problem.version`. No extra properties are accepted. |
| `binding.problem.id` | Required nonempty string. | Exact value of `problem.id`. |
| `binding.problem.version` | Required nonempty string. | Exact value of `problem.version`. |
| `binding.target` | Required object. | Select and map a source target as described below. |
| `binding.group` | Required selector object. | Select the explicit grouping identity for each case. Group names do not infer or enforce an evaluation protocol. |
| `binding.case_fields` | Optional object; default `{}`. | Map chosen output case-field names to selectors, for example participant/session identities used by split validation. |
| `binding.events` | Required in event-epoch mode; forbidden in fixed-window mode. | Object naming the onset and recording-local event-ID columns. |

### Source selectors and identities

`binding.group` and every value in `binding.case_fields` use exactly these keys:

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `source` | Required `"recording"` or `"event"`. | Read the matched recording descriptor or current event row. Fixed-window mode permits only `"recording"`. |
| `column` | Required nonempty string. | Exact dictionary key to read from the named source. A missing key raises. |

There is no fallback to another source when column names overlap. A group or
custom case-field value must be a nonempty string or integer, excluding booleans
and the exact missing-value string `"n/a"`. Groups for included cases must be
uniformly strings or uniformly integers. Fixed-window enumeration also requires
uniformly typed recording groups before applying target exclusions. Custom case
fields are validated individually; they need not share one type across different
fields.

`binding.case_fields` keys must be nonempty strings. These generated names are
reserved and cannot be used as keys:

```text
case_id, recording_id, start_sample, stop_sample, onset, label, group_id,
included, exclusion_reason, channel_names, output_channel_names, row,
source_file, source_sampling_frequency, source_event, source_event_row, event_id
```

Participant, session and trial roles are not inferred from filenames. Copy
the identities needed for an evaluation through `case_fields` explicitly.

Optional BIDS entities such as `session` can be `None` when absent. A selector
used for a group or custom identity then raises. Omit an optional `case_fields`
entry when that identity is unavailable, or choose another valid grouping key.
Do not turn `None` into a string to invent an identity.

### Target mapping

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `binding.target.source` | Required `"recording"` or `"event"`. | Same source rule and mode restriction as a selector. |
| `binding.target.column` | Required nonempty string. | Exact source field containing the target code. |
| `binding.target.mapping` | Required nonempty object. | Nonempty string keys map source codes to class-name strings. Its set of values must equal `problem.class_labels`; several codes may map to one class. |
| `binding.target.on_unmapped` | Optional `"raise"` or `"exclude"`; default `"raise"`. | Raise on an unrecognised code, or retain its candidates with reason `"unmapped_target"`. |

Source target codes must be nonempty strings or integers, excluding booleans.
Codes are compared using their string spelling: integer `11` and string `"11"`
use the same mapping key, while `"011"` is distinct. Dictionary keys that collide
during JSON normalization are rejected. Missing target columns and invalid code
types still raise under `on_unmapped="exclude"`.

Pandas can represent integer codes with blanks as floats, so `11.0` is rejected
by the same rule. Convert a genuinely numeric code column explicitly with the
nullable `Int64` dtype before `to_dict("records")`. Fractional values raise during
conversion. Keep textual codes such as `"011"` as strings when padding matters.
Missing targets still need a policy. For a protocol that excludes them:

```python
import pandas as pd

event_table = pd.DataFrame(event_rows)
event_table["condition"] = event_table["condition"].astype("Int64")
missing = event_table["condition"].isna()
converted_rows = event_table.to_dict("records")
for row, is_missing in zip(converted_rows, missing, strict=True):
    if is_missing:
        row.update(included=False, exclusion_reason="missing_target")
```

Pass `converted_rows` to the loader. Excluded rows retain their identities and
timing, while other missing fields and malformed values still raise.

Fixed-window candidates inherit their recording target. An explicitly unmapped
recording retains its generated windows as exclusions. An event already excluded
by a supplied eligibility decision does not need its target column, but still
needs its timing, event identity, grouping and bound custom identities.

### Event-column binding

`binding.events` contains exactly these two fields:

| Field | Type and presence | Meaning and constraints |
| --- | --- | --- |
| `binding.events.onset` | Required nonempty string. | Event column containing recording-relative cue/onset seconds. |
| `binding.events.id` | Required nonempty string. | Event column containing a nonempty string or integer ID, unique within the recording, including excluded events. Boolean IDs and `"n/a"` are rejected. |

## Recording and event inputs

The following are runtime inputs to the loader, outside the two authored
documents. Recording arrays are nonempty real numeric NumPy arrays in
`(channel, sample)` order, supplied as a collection or stacked recording array.
Metadata entries must align with that collection.

| Recording metadata field | Requirement and interpretation |
| --- | --- |
| `recording_id` or `filepath` | A unique nonempty string. An explicit `recording_id` takes precedence over `filepath`; it must be valid when supplied. |
| `channel_names` | Nonempty unique string names in actual array-channel order, with exactly one name per channel. |
| `sampling_frequency` | Finite positive number in Hz. |
| `n_samples` | Optional integer matching the array length. The loader derives and retains it when absent. |
| Fields used by binding selectors | Required wherever the selected target/group/case field needs them. |
| Other metadata | Retained, including known units, previous processing and original-source identities; these facts are not inferred or authenticated. |

Event-epoch construction requires an iterable of event dictionaries in desired
output order, such as `table.to_dict("records")`. A TSV filename is not read
automatically. Each row supplies:

| Event field | Requirement and interpretation |
| --- | --- |
| `recording_id` | Required string matching one supplied recording. |
| Column named by `binding.events.onset` | Finite numeric seconds, also accepting numeric TSV strings. Boolean, missing, nonfinite and off-grid values raise. |
| Column named by `binding.events.id` | Required recording-local identity with the rules above. |
| Columns used by event selectors | Required for target/group/custom fields, except the target of an already excluded event. |
| `included` | Optional boolean; default `true`. This is a supplied eligibility decision, not a quality detector. |
| `exclusion_reason` | Optional string; default `""`. An excluded event requires a non-whitespace reason; an included event cannot have a reason. |
| Other event columns | Retained in the original `source_event` record, without automatically overriding generated case fields. |

Onsets are measured from the first sample of the supplied array. Acquisition
offsets such as MNE `first_samp`, cropped arrays and one-based source indices
must be resolved before calling the loader. Other columns named `sample`,
`onset_sample` or `duration` are not automatically interpreted or cross-checked
by this binding layer.

Grid validation permits an absolute floating-point residual of `1e-7` samples.
It does not intentionally quantize arbitrary off-grid times. Different included
sources must yield identical output channel names/order and sampling rate;
there is no implicit harmonization. Extracted samples must be finite. Native
values are retained, with no inferred unit conversion.

## Constructed cases and effective configuration

The result is a `ClassificationDataset`. `X`, `y`, `groups` and `cases` have the
same included-case order. Equal-shaped arrays stack in `(case, channel, time)`
order. The two timing modes use fixed case lengths at a common output rate;
per-event duration does not select a different case length. Cases from fixed
windows follow recording/time order; event cases follow input row order.

Every case or exclusion retains `case_id`, `recording_id`, `start_sample`,
`stop_sample`, `label`, `group_id`, `included`, `exclusion_reason`,
`source_sampling_frequency` and `source_file`, plus bound custom fields. Included
cases additionally have zero-based output `row`, `channel_names` and
`output_channel_names`. Event cases also retain `event_id`, numeric `onset`, the
zero-based original input `source_event_row`, and an independent `source_event`.
Exclusions have no output row; a label can be `null` when no target was resolved.

Bounds are zero-based and stop-exclusive on the supplied recording. Event IDs
are encoded with their recording ID while preserving string/integer distinctions;
reordering source rows does not change those case IDs. Fixed-window IDs use the
recording and sample bounds. Case IDs do not include the prediction question,
class mapping or signal contents; an event's case ID can persist when its epoch
changes. Interpret identities together with bounds and effective configuration.

An explicit source eligibility exclusion keeps its supplied reason. Unmapped
targets use `unmapped_target`; an otherwise included out-of-bounds epoch uses
`outside_recording` when permitted. Already excluded events retain their reason
even if their interval is outside the recording. Duplicate identities and other
malformed inputs still raise. If all candidates are excluded or no complete
windows exist, construction raises instead of returning an empty dataset.

`dataset.metadata` retains recording metadata, output channels/rate and sample
coordinate conventions. Its `problem` entry contains:

| Field | Meaning |
| --- | --- |
| `definition` | Independent effective problem document, including defaults. |
| `binding` | Independent effective binding document, including defaults. |
| `definition_sha256` | SHA256 of the effective problem serialized as sorted-key JSON. |
| `binding_sha256` | SHA256 of the effective binding serialized as sorted-key JSON. |

Explicit `step_size: null` is retained as `null` in that document even though
construction uses the window duration; omission resolves to the numeric window
duration. These hashes identify effective document representation, not guaranteed
semantic equivalence, source content or completed execution. Inputs and outputs
do not share mutable case/configuration state.

## Scope of validation

This format constructs cases on already loaded arrays. It does not discover or
download sources, execute a quality rule, filter/resample data, choose partitions,
fit a classifier or compute metrics. Authored structural validity does not
establish class meaning or scientific suitability. Source aliases, prior
processing and complete preprocessing support must remain explicit when adding
an evaluation. Use [assignment validation](classification-problems.md) before
fitting; grouping metadata alone does not prove independence or prevent leakage.
