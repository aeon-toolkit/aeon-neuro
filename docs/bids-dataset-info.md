# Loaded BIDS recordings and aligned metadata

`load_bids_dataset` loads local EDF/BDF recordings and returns their complete
source metadata through `EEGDatasetInfo`. It reuses the existing
[source inspector](bids-problem-input.md), snapshots and source opener. Loading
does not infer a target, change channel units/order, apply a montage or preprocess
signals. The existing `load_bids_data` interface and its format support are unchanged.

```python
from aeon_neuro.datasets import load_bids_dataset

X, info = load_bids_dataset("/data/study", subjects=["01", "02"])
print(info.source_ids, info.participant_ids)
for recording in info.recordings:
    print(recording.recording_id, recording.sfreq, recording.channel_names)
```

This explicit operation reads complete selected recordings into RAM. Equal
shapes, channel names/order and exact sampling rates produce a 3D NumPy array
in `(recording, channel, sample)` order. Otherwise the result is a list of 2D
arrays; no padding, reordering or resampling makes them stackable. Every owned
source handle is closed, including when a later recording fails to load.

For large sources, retain metadata-only inspection and selected case
materialization instead. `EEGDatasetInfo(inspect_bids_eeg(...))` creates a metadata
view without reading signals, but does not itself load or verify signal content.
The constructor requires a ready version-2 inspection report and checks selected
source/event identities and coordinates. Inspection failures remain explicit
errors, including their original blocking messages.

## Inspect source facts

The metadata views expose detached Python dictionaries and rows, preserving the
portable inspection representation. Optional detached pandas views are described in the [EEG information guide](eeg-information.md).
Unknown optional facts stay `None`; source rows and `n/a` values retain their meanings.

| View | Contents |
| --- | --- |
| `info.description`, `info.readme` | Complete source dataset description and README, retaining the README's original line endings. |
| `info.participants`, `info.participants_json` | Complete source participant table (`columns`/`rows`) and column dictionary, including unselected participants. |
| `info.dataset` | Dataset facts, file identities and selected participant scope. |
| `info.source_ids` | Source recording IDs in signal order. |
| `info.participant_ids` | One original `sub-` participant ID per recording, in signal order. Multiple recordings may share one participant. |
| `info.metadata`, `info.events` | Detached original acquisition descriptors/events. |
| `info.current_metadata`, `info.events_for_cases()` | Paired current recording descriptors and literal event targets for the plain case API, with the supplied current layout. |
| `info.recordings` | Tuple of `RecordingInfo` views in signal order. |
| `recording.entities` | Source subject/session/task/acquisition/run values, without prefixes. |
| `recording.sfreq`, `recording.channel_names` | Decoded rate and ordered source channel names. |
| `recording.channels`, `recording.eeg_json` | Original channel rows and resolved source EEG JSON. |
| `recording.events` | Source event rows, including exact `bids_onset_sample` coordinates where available. |
| `recording.electrodes`, `recording.coordsystem`, `recording.scan` | Optional source geometry and matched scans row. |

`recording.bad_channels()` returns only channels explicitly marked `bad`, raising
when statuses are absent or unrecognized. `recording.channels_of_type("EEG")`
uses declared channel-table types and raises when those facts are unavailable.
Neither method certifies clean EEG or chooses channels automatically. Full value
origins, diagnostics and source snapshots remain available in `info.metadata`
and the detached version-2 report returned by `info.to_dict()`.

Changing a returned dictionary/list does not mutate the stored metadata. Source
facts are not rewritten to describe arbitrary external array operations. For
example, a source `EEGReference` remains an acquisition fact after a caller
independently rereferences an array. Use explicit history records to document
processing without changing those source facts.

## Select signals and metadata together

```python
X_selected, selected_info = info.select(X, [1, 0])  # reverse source order
X_subject, subject_info = info.select(X, [True, False])
X_empty, empty_info = info.select(X, [])
```

`select` accepts a slice, unique nonnegative integer indices, or a boolean mask
matching the recording count. Slice semantics follow Python, including reverse
slices. Negative integer indices, duplicate recordings, mixed boolean/integer
selectors and incompatible shapes raise errors. A selection copies both the
signal arrays and metadata so later edits cannot change the parent. This can
temporarily require substantial memory. Dense arrays retain their recording
axis; list/tuple inputs return a list of separate arrays.

Events follow the requested recording order, preserving their original IDs and
within-recording row order. The selected participant scope and recording-specific
diagnostics are updated; the complete source participant table is retained.
An empty view is valid and serializable, although classification construction
requires eligible observations. It cannot be used as an empty training dataset.

`info.validate_layout(X)` checks recording counts, sample/channel dimensions and
numeric types. These checks cannot detect an external permutation of equal-shaped
arrays or same-shape value edits. Keep X and info together and use paired selection;
this API does not fingerprint arrays on every operation. Source metadata inspection
also does not certify unchanged signal content without a trusted content hash.

## Construct cases with shared information

Use the existing case helpers with the object as their metadata argument:

```python
from aeon_neuro.datasets import make_event_cases, build_classification_dataset

candidate_events = selected_info.events_for_cases(label_column="trial_type")
candidates = make_event_cases(
    X_selected, selected_info, candidate_events, tmin=0.0, tmax=1.0,
)
cases = build_classification_dataset(X_selected, selected_info, candidates)
print(cases.info.participant_ids, cases.info.channel_names)
```

The event convenience preserves literal source labels and explicit recording
groups. Map labels or edit eligibility in the returned rows when the scientific
question requires it. Fixed windows and whole recordings use explicit labels and
groups through their ordinary case helpers. No problem-definition document or
second event parser is required. Supported processing updates the recording grid
used by the helpers while preserving published event rows separately.

The result is the existing `ClassificationDataset`. Passing the shared object
adds one version-1 `metadata["provenance"]` record containing original `source`,
complete processing `history`, and the actual `construction` implementation and
package version. `dataset.info` derives detached current case identities,
subjects/groups, channels/rate/units and original source support from that record.
Explicit case participant identities take precedence over recording subjects.
Logical output-channel aliases retain their original source picks and units.

Plain descriptor/event inputs keep their existing behavior. They do not
implicitly acquire dataset-level source context or processing history. Source
snapshots and reported history do not authenticate arbitrary external array edits
or establish leakage-free processing. Validate actual evaluation assignments
separately with the existing [split validation](classification-problems.md).
When shared provenance is present, split validation also checks its original
source support; current nonoverlapping window bounds cannot hide dependencies
from earlier filtering or resampling.
