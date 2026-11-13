# EEG information

`EEGDatasetInfo` stores metadata aligned with EEG arrays. It keeps current row
layouts separate from optional original source facts. Construction reads no
files and does not certify the contents or interpretation of supplied data.
The interface is experimental.

```python
import numpy as np
from aeon_neuro.datasets import EEGDatasetInfo

X = np.zeros((4, 2, 128))
info = EEGDatasetInfo.from_arrays(
    X, sfreq=128, channel_names=["C3", "C4"], units=["V", "V"],
    subjects=["01", "01", "02", "02"],
    case_ids=["left-01", "right-01", "left-02", "right-02"],
)
selected_X, selected_info = info.select(X, [3, 0])
restored = EEGDatasetInfo(selected_info.to_dict())
restored.validate_layout(selected_X)
```

## Stored information

| Scope | Contents |
| --- | --- |
| Current row | Case identity, optional subject and group, optional original recording identity and source interval. |
| Current layout | Ordered channel names, sampling frequency, per-channel units, sample count and source grid mapping when known. |
| Optional dataset facts | Dataset description, README, complete participants table and participant dictionary. |
| Optional source recording | Identity, participant, source layout and supplied channel, event, acquisition and geometry metadata. |

`from_arrays` requires a rate and ordered channel names. Units and subjects may
remain unknown. Groups default to supplied subjects. Default case IDs identify
only the supplied collection; they do not establish shared physical sources.
Original source references and intervals are meaningful only when supplied.
Labels remain a separate `y` array.

Source metadata can be supplied directly, without a reader or inspection report:

```python
source_info = EEGDatasetInfo(
    dataset={"description": {"Name": "Example"}},
    recordings=[{
        "recording_id": "recording-01", "participant_id": "01",
        "channel_names": ["C3", "C4"], "sampling_frequency": 128,
        "n_samples": 1024,
    }],
)
```

`info.recordings` contains original `RecordingInfo` views. Supplied sidecar rows
are preserved rather than rewritten to describe external transformations.
Source participant IDs can be strings, integers or unknown (`None`). Unknowns
stay on their recording rows, while `dataset.selected_participants` in portable
state lists only known IDs. Selection recomputes that scope without inferring
identities or converting integer IDs to strings.

Unknown optional facts remain absent. Names alone make no electrode-position
claim. `info.participant_ids` and `info.groups` describe current rows;
`info.participants` retains the full supplied source population.

## Selection and portable state

`info.select(X, selector)` copies arrays and matching information together.
`info[selector]` selects information alone; apply the same selector to `X` and
`y`. Selectors accept slices, unique nonnegative integer indices and matching
boolean masks. Empty selections are valid. Copies can require substantial RAM.
Current layouts can describe unequal-length rows; uniform rate/channel/unit
accessors raise if rows disagree rather than selecting an arbitrary row.

`to_dict` returns detached version-2 state with an optional version-1 current
view. `EEGDatasetInfo(state)` validates and restores it. This encoding retains
source context separately from selected current rows. Returned dictionaries,
and rows are detached from stored state. Metadata contains no
signal samples, and layout checks cannot detect equal-shaped array swaps or
external value changes.

BIDS inspection/loading, estimator integration, processing and training-time
validation are separate consumers of this metadata contract.

## Optional information in classifiers

EEG classifiers accept keyword-only `info` in `fit`, `predict` and
`predict_proba`. `BaseEEGClassifier` validates and retains detached fitting
information while preserving aeon's reset and validation lifecycle. Constructor
rate/channel conflicts raise. Prediction information must match the fitted
rate, ordered names and known units. Call-specific prediction information is
removed after success or failure. When fitting uses plain arrays, known rate and
channel constructor settings are retained as fitting facts too. Changing a
constructor parameter after fitting does not redefine the layout of the trained
model; supplied prediction information must still agree with those retained facts.
Unknown fitting facts remain unknown. Pretrained CBraMod and LaBraM require
200 Hz whenever a rate is supplied, including prediction after plain-array
fitting. This fitted model requirement is independent of the unknown training
rate and does not change when constructor parameters are later edited.

```python
from aeon_neuro.classification.deep_learning import EEGNetClassifier

clf = EEGNetClassifier(n_epochs=1)
clf.fit(X, y, info=info)
probabilities = clf.predict_proba(X, info=info)
```

Plain arrays, cloning, sklearn pipelines and ordinary CV calls retain their
interfaces. Automatic metadata forwarding through pipelines/CV is separate.
Native aeon reexports keep their native interfaces. Information-derived channel
names alone imply no positions; spatial models or explicit constructor names
use the existing standard_1005 template policy, recorded in channel_geometry_.
Measured source electrodes are not silently converted to model geometry.


## BIDS sources and case construction

`inspect_bids_eeg` reads supported local EDF/BDF headers and sidecars without
reading signal samples. `load_bids_dataset` returns complete recordings and
aligned information through an explicit verified opener. The loader is separate
from the metadata container and closes all owned handles on success or failure.
Original events, channels, participants, acquisition fields and source geometry
remain available. Decoded EEG samples use volts; source sidecar units retain
their original meaning.

Ordinary recording/window/event helpers accept the information object and
`build_classification_dataset` returns aligned `dataset.info`. Labels, grouping,
channel choices and intervals remain explicit. No authored problem schema is
required. See the [source guide](bids-problem-input.md) and
[loading guide](bids-dataset-info.md) for conflicts, snapshot limits and examples.


The source-record dataclass also retains `sidecar_paths`, a mapping from sidecar
roles to source paths or explicit absence. An absent `channels` entry does not
hide supplied channel rows; an explicit `channels: None` records unavailability.

## Executed processing

`info.process(X, operation, **parameters)` returns detached arrays and updated
information together. It supports ordered channel selection, explicit scaling,
exact-grid crop, polyphase resampling, band-pass filtering and referencing.
Failures leave arrays and metadata unchanged. No operation fits a learned
transform; learned preprocessing still belongs inside training folds.
Operations update named layout fields and serialize the performed facts, including
reference or temporal-context fields omitted from a restored input. Original
source descriptors remain separate throughout the operation chain.

```python
X, info = info.process(X, "channels", names=["C4", "C3"])
X, info = info.process(X, "crop", tmin=0.25, tmax=0.75)
X, info = info.process(X, "resample", sfreq=64)
```

Current rate/channel/unit/grid fields change while original acquisition facts
remain intact. Executed history records implementation versions, parameters and
original application scopes. Selection preserves earlier scopes. Filtering and
resampling conservatively retain the complete contributing input row as source
support. Pointwise operations permit exact derived intervals. The ordinary
split validator checks known original-source support across partitions.

Resampling uses SciPy resample_poly with the exact decimal rate ratio, reflected
boundaries and ceil(input_count * target_rate / input_rate) output samples. The
sample grid does not change to fit the rounded count. Reduced numerator and
denominator above 100,000 raise before filter allocation. Source stats and
metadata history do not authenticate arbitrary external array edits.


## Dictionary structure

This is the reference for portable dictionaries. Internally, fixed metadata
structures use named dataclass fields:

| Record | Contents |
| --- | --- |
| `_DatasetFacts` | Dataset description, README, participant table/dictionary and selected source participants. |
| `_RecordingDescriptor` | Original recording identity, participant, source channels/rate/length, entities and optional source annotations. |
| `_CaseRow` | Current case, recording, participant and group identities with contributing sample support. |
| `_SignalLayout` | Current channels, sampling rate, units, sample count, reference and source-grid mapping. |
| `_CurrentView` | Version, kind and aligned lists of typed case rows and layouts. |

`EEGDatasetInfo._dataset` holds dataset facts. `_recordings` contains
`RecordingInfo` views, each with a typed `_source` descriptor and original event
rows. `_view` contains typed current rows/layouts, or `None` for source-only
information. Original events and diagnostics remain separate lists. Flexible
EEG JSON fields, table rows and unrecognized source extensions remain dictionaries.

Conversion to and from dictionaries happens at the portable-state and public
property boundaries. Each record tracks supplied keys so omitted optional fields
stay omitted; unrecognized fields round-trip without becoming fixed attributes.
The outer version-2 and current-view version-1 formats are unchanged. Records are
private mutable implementation details, not public immutable value objects.
Construction, properties and selection retain detached ownership. Use the public
properties and `to_dict()` rather than mutating private fields, which bypasses
alignment checks.

The following tables describe the core keys. Additional supplied source fields
are retained; this container does not validate every source annotation or make
arbitrary dictionaries JSON-compatible. No signal samples or target array `y`
are stored.

### Version-2 source state

| Key | Type | Meaning |
| --- | --- | --- |
| `version` | int | `2`, the outer state format. |
| `ready` | bool | Must be `True` for construction; describes supplied metadata readiness, not source authentication. |
| `issues` | list of dict, optional | Source diagnostics, empty for explicit construction. Omission is preserved through restoration and selection. Diagnostic rows use `severity`, `message` and `recording_id`; a `None` recording ID denotes dataset scope. |
| `dataset` | dict | Dataset facts described below. |
| `source_ids` | list of str | Original recording identities, exactly aligned with `recordings`. Unlike the public property, this is not derived from current cases. |
| `recordings` | list of dict | Original recording descriptors, empty for `from_arrays`. |
| `events` | list of dict | Original event rows referencing retained recording identities, empty when unavailable. |
| `view` | dict, optional | Version-1 current rows and layouts. Omitted for source-only information, rather than set to `None`. |

The `dataset` dictionary contains:

| Key | Type | Meaning |
| --- | --- | --- |
| `description` | dict or None | Original dataset description. |
| `readme` | str or None | Original README text. |
| `participants` | dict or None | Complete source participant table, using `columns` and `rows`. |
| `participants_json` | dict or None | Original participant-column dictionary. |
| `selected_participants` | list of str or int | Known participants in the retained `recordings`, sorted with integers before strings. Unknown IDs are omitted; `1` and `"1"` remain distinct. |

Selection of source-only information narrows the recording descriptors, events,
diagnostics and selected participant scope. Selection within a current view
narrows only its rows and layouts, retaining the full source context. Therefore
`dataset["selected_participants"]` need not equal the participants in a selected
case view. `participants` always retains the complete supplied source population.

### Original recording descriptors and events

Every recording descriptor requires the first five keys below. Optional fields
retain original source facts; they do not describe later case transformations.

| Key | Type | Meaning |
| --- | --- | --- |
| `recording_id` | str | Unique nonempty recording identity, root-relative for BIDS sources. |
| `participant_id` | str, int or None | Original participant identity, or unknown. |
| `channel_names` | list of str | Unique ordered source channel names. |
| `sampling_frequency` | float | Positive source rate in Hz. |
| `n_samples` | int | Positive original recording length in samples. |
| `subject`, `session`, `task`, `acq`, `run` | str or None, optional | Original BIDS entities. `RecordingInfo.entities` exposes `acq` as `acquisition`; missing entities return `None`. |
| `channel_metadata` | list of dict or None, optional | Original channel table rows, exposed as `channels`. Missing or `None` means unavailable. |
| `channel_units_fiff` | list of int, optional | Decoded unit codes aligned with source channels. Derived recording layouts map voltage/magnetic-field codes to `V`/`T`; unavailable or unrecognized codes give `None`. |
| `eeg_metadata` | dict, optional | Resolved original EEG JSON fields, exposed as `eeg_json`; omission gives `{}`. `EEGReference`, if supplied, seeds a source-only layout's reference. |
| `electrodes` | dict or None, optional | Original electrode table with `columns` and `rows`. |
| `coordsystem` | dict or None, optional | Original coordinate-system fields and units, without conversion. |
| `scan` | dict or None, optional | Matched scans entry with `row` (dict) and `row_index` (zero-based int). |

Source tables use `columns` (an ordered list of column names) and `rows` (a list
of dictionaries keyed by those names). Missing optional geometry or scan fields
are exposed as `None`. Source-file channel units can differ from decoded array
units; they must not be substituted for current layout units.

Each original event row contains these keys. Additional original event columns
are retained without being interpreted as labels.

| Key | Type | Meaning |
| --- | --- | --- |
| `recording_id` | str | Identity of a retained source recording. |
| `bids_event_id` | supplied identifier | Unique within its recording; the pair identifies the original event. |
| `onset` | float | Original onset in seconds relative to recording start. |
| `bids_onset_sample` | int or None | Exact source-grid sample for `onset`, or `None` for an off-grid onset. Construction checks consistency without rounding. |

### Version-1 current view

| Key | Type | Meaning |
| --- | --- | --- |
| `version` | int | `1`, independently versioned from the source state. |
| `kind` | str | `cases` or `recordings`. |
| `rows` | list of dict | Current aligned row identities and contributing support. |
| `layouts` | list of dict | One current layout per row, in exactly the same order. |

Each `rows` entry contains:

| Key | Type | Meaning |
| --- | --- | --- |
| `case_id` | str | Unique nonempty identity within the current view. |
| `recording_id` | str or None | Original source recording, or `None` for arrays without source context. |
| `participant_id` | str, int or None | Original subject for this current row, or unknown. |
| `group_id` | str, int or None | Declared current split group, defaulting to the supplied subject in `from_arrays`. |
| `source_support` | list of two int | Zero-based half-open contributing interval `[start, stop)` in source samples. Without source context, `from_arrays` uses the supplied row's `[0, n_samples)` interval; it establishes no physical source lineage. |

Each `layouts` entry contains:

| Key | Type | Meaning |
| --- | --- | --- |
| `channel_names` | list of str | Unique current channel names in array order. |
| `sampling_frequency` | float | Positive current rate in Hz. |
| `units` | list of str or None | One unit per current channel; individual entries may be `None`, never the list itself. |
| `n_samples` | int | Positive current row length. |
| `reference` | supplied value or None | Declared current reference, unknown for `from_arrays`. |
| `source_origin_sample` | float | Nonnegative source-grid coordinate of current sample zero. |
| `source_sample_step` | float | Positive source-grid increment per current sample. With a known source, equals source rate divided by current rate. |
| `temporal_context` | str, optional | `pointwise` or `whole_row`, describing whether contributing support is pointwise or spans the row. Constructors use `pointwise`; validation treats omission as `whole_row`. |
| `source_channel_names` | list of str, optional | Ordered original channels corresponding to current names. Omission uses `channel_names`; supplied mappings must align with the current channel count. |

The source coordinate for current sample `j` is
`source_origin_sample + j * source_sample_step`. Grid points must fall inside
the row's contributing support `[start, stop)`; a point at `stop` is outside it.
A source-only object has no stored current view;
`layouts` derives recording layouts on demand, while `cases` and `case_ids`
raise `ValueError`. Uniform `sfreq`, `channel_names` and `units` accessors also
raise for empty collections or differing row values. Unknown units appear as
`None` entries in a tuple, rather than a `None` tuple.
