# Inspect local BIDS EEG sources

`inspect_bids_eeg` retains source facts from local EDF/BDF recordings and their
BIDS sidecars for inspection or later loading. Source inspection and
scientific choices stay separate: the adapter has no problem, target mapping,
epoch, quality policy or evaluation configuration to infer.

Entity discovery uses [MNE-BIDS](https://mne.tools/mne-bids/stable/generated/mne_bids.find_matching_paths.html).
Sidecars use its [BIDSPath matching](https://mne.tools/mne-bids/stable/generated/mne_bids.BIDSPath.html)
and aeon-neuro's existing EEG JSON inheritance resolver. Supported native
headers are opened through the existing ingestion reader, without calling
`get_data` or hashing full signal files. All source handles are closed.

For explicit whole-recording loading and paired signal/metadata selection, see
[loaded BIDS recordings](bids-dataset-info.md). Metadata-only inspection remains
available without allocating the signal arrays.

## Inspect and read the diagnostics

```python
from aeon_neuro.datasets import inspect_bids_eeg

report = inspect_bids_eeg("/data/nm000245", subjects=["1"])
for issue in report["issues"]:
    print(issue["severity"], issue["recording_id"], issue["message"])
```

The first supported encodings are EDF and BDF. Discovered BrainVision and
EEGLAB recordings produce a blocking diagnostic because their companion-file
snapshots are outside this increment. Other existing loaders are unchanged.
Filters `subjects`, `sessions`, `tasks`, `acqs` and `runs` use BIDS entity values
without prefixes. No matching recordings is an explicit error report.
Every supplied filter value must match within the combined selection. A partial
match such as `subjects=["1,", "2"]` is a blocking selection diagnostic, using
the same guard as `load_bids_data`, before any native header is opened.

The returned ordinary dictionary has:

| Field | Meaning |
| --- | --- |
| `version` | Inspection-report version, currently `2`; separate from the problem format and source-snapshot versions. |
| `ready` | Whether all selected sources passed this source inspection. It does not establish that a particular problem or evaluation is ready. |
| `source_ids` | Every selected recording's root-relative source path, in deterministic order. |
| `recordings` | Source descriptors when ready; `None` if any source has a blocker. |
| `events` | Original event columns with explicit recording and table-row IDs when ready; `None` if blocked. |
| `dataset` | Dataset-level description, README, complete participant table/dictionary, selected participant IDs and consumed-file identities. `None` if dataset metadata could not be inspected. |
| `issues` | Dictionaries containing `severity`, `code`, `recording_id` and an actionable `message`. |

Malformed or ambiguous metadata, channel-order/rate conflicts, invalid event
timing and unreadable or unsupported sources block the catalogue. A broken
recording cannot silently disappear while the remaining subset looks ready.
Review the issues or select an explicitly narrower scope before trying again.

Missing optional participant/channel/event tables produce warnings. Known header
facts remain available; missing participant attributes, session identities and
event rows are not invented. An event-epoch problem still needs actual events.
Published discontinuities and duration discrepancies are reported and retained,
without automatic clipping, interpolation or source corrections. This is not
a complete BIDS validator or an artifact detector.

Descriptors retain actual channel order, rate, exact sample count, first sample,
native channel types and FIFF unit codes. Sidecars do not rescale or retype the
decoded samples. Resolved EEG JSON and field origins are retained separately,
along with dataset attribution/conversion metadata when available. Participant
attributes use literal flat keys such as `participant.cohort`; these are ordinary
binding column names rather than dotted lookup expressions.

Event rows retain their original columns, adding `recording_id`, `bids_event_id`,
zero-based `bids_event_row` and integer `bids_onset_sample`. These IDs identify rows of the published table,
not recovered original acquisition trials. `onset` must be finite and on-grid;
any supplied `sample` and `onset_sample` must agree with it and one another.
`bids_onset_sample` is computed from onset seconds and the decoded rate, measured
from the recording's first stored sample. It is present even when the published
table has no sample column. The four added names are reserved: an existing source
column with one of those names blocks inspection rather than being overwritten.
Unknown `duration="n/a"` is preserved. Quality annotations require a separately
reviewed interpretation; their mere absence never means that trials are clean.

## Dataset and additional recording sidecars

Dataset-level files are read once for the selected scope. All TSV values remain
strings, including `n/a`; JSON values retain their types and arbitrary fields.
Tables have `columns` and `rows`, preserving column and row order. This report
does not require pandas or construct an in-memory signal/metadata container.

| Dataset field | Contents |
| --- | --- |
| `description` | Complete `dataset_description.json` object, or `None`. The earlier per-recording `dataset_metadata` field is retained too. |
| `readme` | UTF-8 text with original line endings, or `None`. Supports `README`, `README.md`, `README.rst` or `README.txt`; multiple variants block inspection. |
| `participants` | Complete `participants.tsv` table, including unselected participants, or `None`. Duplicate participant IDs block inspection. |
| `participants_json` | Complete participant-column dictionary, or `None`. |
| `selected_participants` | Sorted distinct `sub-` identifiers for the selected recordings, independently of the source participant table. |
| `paths` | Root-relative filenames for the four metadata fields, with `None` for absent files. |
| `files` | Size, nanosecond modification time and SHA256 for each consumed dataset metadata file. |

Each recording additionally provides:

| Recording field | Contents |
| --- | --- |
| `electrodes` | Applicable electrode table as `columns`/`rows`, or `None`. Preserves published values, missing coordinates and electrode order. Electrode rows may include references absent from signal channels. |
| `coordsystem` | Applicable coordinate-system JSON object, or `None`. Coordinates and units are neither converted nor inferred. |
| `scan` | The matching scans-table row as `row_index` and `row`, or `None`. The row index is zero-based in the source table. Duplicate matching rows block inspection. |
| `sidecar_paths` | Root-relative paths for channels, events, electrodes, coordsystem and scans, including `None` for absent files. |

```python
if report["ready"]:
    participants = report["dataset"]["participants"]
    for recording in report["recordings"]:
        print(recording["recording_id"], recording["sidecar_paths"])
        if recording["coordsystem"] is not None:
            print(recording["coordsystem"].get("EEGCoordinateUnits"))
```

Electrode/coordinate files use public MNE-BIDS sidecar matching, retaining strict
ambiguity errors. A single electrode `space-` can be resolved even when the raw
filename has no space entity; its coordinate system must match that space.
Multiple equally applicable geometries block inspection. Geometry rows do not
reorder signal channels or apply a montage. Missing coordinate units remain a
warning and do not establish that coordinates are usable for spatial operations.

The scans table belongs to the recording's subject/session. Its `filename`
joins the source path relative to that table's directory, using POSIX paths.
Original filename and acquisition-time text are retained. A missing matching
row is a warning, rather than an invented acquisition time or a row from another
session. This is structural inspection, not complete BIDS validation.

All consumed dataset and recording sidecars enter each recording's `bids_source`
snapshot. Changed README, dictionary, geometry or scans files therefore fail
before source opening. Metadata is checked around parsing to detect changes
during inspection. These checks are not a filesystem transaction; keep sources
immutable during use. Missing optional metadata is explicit through `None` and
diagnostics. Malformed tables/JSON block the catalogue.

Version 2 adds dataset/recording sidecars and reserves `bids_onset_sample`.
Earlier version-1 source snapshots remain readable by
`open_bids_eeg_recording`; their saved meaning is unchanged, without adopting
new sidecars. Reinspect to obtain the expanded report. Authored problem/binding
format version 1 is unchanged. The [loaded metadata view](bids-dataset-info.md)
reuses this report for paired selection and caller-reported history. Scientific
choices remain explicit in problem bindings.

## Open an explicitly selected source

```python
from aeon_neuro.datasets import open_bids_eeg_recording

if not report["ready"]:
    raise ValueError(report["issues"])

descriptor = report["recordings"][0]
with open_bids_eeg_recording("/data/nm000245", descriptor) as raw:
    values = raw.get_data(start=0, stop=min(128, raw.n_times))
```

Only the selected source is opened. Saved metadata-file SHA256 checks, source
size/modification time and decoded headers are checked before returning an owned
lazy Raw handle. The caller explicitly reads samples and closes the handle;
inspection itself remains metadata-only. The source files are unchanged.

Source size/time and header agreement are **not full content verification**.
For trusted release checks, add `expected_sha256` to each recording descriptor
before opening it. The opener verifies that signal checksum during source
opening, reading every byte of the selected file. Inspection does not
invent a release checksum or transfer missing files.

File snapshots are local stale-input checks, not signatures or a filesystem
transaction. Keep inspected inputs immutable during use. Metadata additions do
not redefine a saved catalogue: reinspect to adopt new source files or corrected
metadata. Different paths may alias one original recording; original-source
identities and full processing supports still need an explicit evaluation policy.

## Format changes remain a separate responsibility

The adapter consumes BIDS sources and produces source facts; it does not parse
or migrate problem documents. Its report/snapshot version and the authored
`format_version` identify different contracts. New problem types can use the
same inspected facts when those inputs remain sufficient. Other modalities or
companion-file layouts need appropriate source adapters.

Retain authored problem documents separately from source evidence. Reinspect
when adopting corrected source metadata or a different selection. The inspector
does not migrate problem formats or certify scientific suitability.
