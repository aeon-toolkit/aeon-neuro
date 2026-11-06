# EEG information from source recordings to estimators

`EEGDatasetInfo` holds source sidecars and the current layout of recording or
case arrays. Source acquisition facts remain available after channel selection,
cropping, scaling or resampling. All interfaces on this page are experimental.

## Load, construct and select cases

```python
from aeon_neuro.datasets import (
    load_bids_dataset, make_event_cases, build_classification_dataset,
)

recordings, source_info = load_bids_dataset("/data/study", subjects=["01", "02"])
events = source_info.events_for_cases(label_column="trial_type")
candidates = make_event_cases(recordings, source_info, events, tmin=0.0, tmax=1.0)
dataset = build_classification_dataset(recordings, source_info, candidates)
X, y, info = dataset.X, dataset.y, dataset.info
```

The ordinary case helpers keep scientific labels, grouping, intervals and channel
choices explicit. They accept the shared object as their metadata argument and
reuse the same case engine as plain descriptors. `dataset.info` returns detached
case information with source/history/construction provenance. No problem schema
is needed. Whole-recording and fixed-window helpers also accept the object.

| Property | Meaning |
| --- | --- |
| `info.case_ids`, `info.cases`, `info.case_table` | Current row identities and a detached list/pandas table of case facts. |
| `info.participant_ids`, `info.subject_of(i)` | One known original subject per current row; unknown identities stay `None`. |
| `info.groups` | Declared case groups, which can differ from subjects. |
| `info.sfreq`, `info.channel_names`, `info.units` | Current rate, ordered channels and decoded or explicitly supplied units. Heterogeneous layouts raise rather than choose an arbitrary row. |
| `info.layouts` | One current layout per row, including sample count and its original-source grid mapping. |
| `info.recordings`, `info.description`, `info.participants_frame` | Original recording views, dataset description and complete source participant population. |
| `info.history` | Preserved processing records and their original scopes. |

Decoded MNE EEG arrays use volts even when a BDF sidecar describes stored values
in microvolts. Original channel-table units remain unchanged in `info.recordings`.
Explicit case participant bindings take precedence over the recording subject;
the original BIDS entity remains in the source recording view. Unknown units
are never guessed. Original electrodes and their coordinate system
remain source facts; channel names do not establish measured positions.

Use paired selection for detached arrays and information:

```python
X_selected, selected_info = info.select(X, [3, 0])
```

`info[mask]` also returns detached information aligned with `X[mask]`. Select
`y` with the same indices. Integer selectors must be unique and nonnegative;
boolean masks must match the row count. Metadata validation cannot detect
external permutations or edits of equal-shaped arrays.

For arrays not loaded from BIDS, supply facts explicitly:

```python
import numpy as np
from aeon_neuro.datasets import EEGDatasetInfo

X = np.zeros((4, 2, 128))
info = EEGDatasetInfo.from_arrays(
    X, sfreq=128, channel_names=["C3", "C4"], units=["V", "V"],
    subjects=["01", "01", "02", "02"],
    case_ids=["01-left", "01-right", "02-left", "02-right"],
)
```

This creates no BIDS acquisition claims. Generated default case IDs identify only
the supplied collection. Subjects and units can remain explicitly unknown.

## Events and tables

`source_info.events_for_cases(label_column="trial_type")` supplies rows ready
for `make_event_cases`, using literal source labels and recording participant
IDs. `group_column` chooses another explicit recording field. Missing labels or
groups raise. Edit the returned event rows for explicit class mappings or event-based groups.
No convenience infers the scientific meaning of source codes.

For the plain case API, pair current descriptors with those events:

```python
from aeon_neuro.datasets import make_event_cases, build_classification_dataset

metadata = source_info.current_metadata
candidates = make_event_cases(
    recordings, metadata, source_info.events_for_cases(), tmin=0.0, tmax=1.0,
)
dataset = build_classification_dataset(recordings, metadata, candidates)
```

`current_metadata` follows supported recording processing and selection, including
cropped event origins and resampled grids. It returns detached descriptors.
Only channel names, rate and sample count reflect the current layout; other
fields remain source context. Original descriptors/events remain in `metadata`
and `events`. Already segmented case views raise instead of being reused as
recordings. Passing these descriptors alone retains the ordinary plain-input provenance.
Pass `source_info` itself as metadata to preserve the shared source/history record.

`recording.events_frame`, `recording.channels_frame` and
`info.participants_frame` provide detached pandas views. Portable dictionaries
and rows remain available for serialization. The existing bad-channel/type
accessors keep their explicit missing-information behavior.

## Execute and record processing

```python
X, info = info.process(X, "channels", names=["C4", "C3"])
X, info = info.process(X, "filter", l_freq=8.0, h_freq=30.0, method="iir")
X, info = info.process(X, "crop", tmin=0.25, tmax=0.75)
X, info = info.process(X, "resample", sfreq=64)
X, info = info.process(X, "scale", factor=1e6, units="uV")
```

Each successful call returns detached signals and information together. Failures
leave both inputs unchanged. The supported operations are:

| Operation | Parameters and behavior |
| --- | --- |
| `channels` | Explicit ordered `names`; preserves source geometry and reference. |
| `scale` | Finite nonzero `factor` and explicit output `units`. |
| `crop` | Optional `tmin`, `tmax`, `start_time`; reuses `TimeCrop` and its exact, half-open sample-grid checks. |
| `resample` | Positive target `sfreq`; reuses MNE polyphase resampling and records its actual output count. |
| `filter` | Explicit `l_freq`, `h_freq`, optional `method`; reuses `BandPassSeriesTransformer`. |
| `reference` | `channels="average"` or an explicit name list; records the contributing channels and requires equal known units. |

Operations run independently within each current row. Filtering an epoch and
filtering a continuous recording have different boundary support; choose the
stage deliberately. Current sample origins/steps change after cropping or
resampling. Case `source_support` retains a conservative half-open interval on
the original recording grid, including context that earlier processing may
have consumed. Filtering and resampling retain the complete contributing input row. Pointwise
channel selection, scaling, referencing and crop permit exact case intervals.
The shared split validator and inner classifier validation check this known
original support as well as the declared identities. A narrower verified support
map can permit more splits; dependent outputs must not be declared independent.

The operations fit no learned transform. Learned preprocessing still belongs
inside the training fold. `with_history` remains available for caller-reported
operations, with no claim that those operations ran or changed the layout.

Processed recordings can feed the ordinary case helpers. Event times are mapped
onto the current grid, while published events remain preserved separately.
A source column named `acquisition_event` blocks this derived route rather than
being overwritten. Already segmented case views are rejected as recording inputs.
Logical case channel aliases retain `source_channel_names`, so later selection
and estimator layout checks preserve the actual source picks and units.

## Fit and predict with optional information

```python
from aeon_neuro.classification.deep_learning import EEGNetClassifier

clf = EEGNetClassifier(
    n_epochs=20, validation_size=0.2, patience=5, random_state=42,
)
clf.fit(X_train, y_train, info=info_train)
probabilities = clf.predict_proba(X_test, info=info_test)
```

The EEG-specific classifiers inherit `BaseEEGClassifier`, which calls aeon's
base methods for resetting, collection/label validation, fitting and prediction.
Protected estimator methods retain their ordinary signatures and can access
`info_`, `sfreq_`, `channel_names_` and `subject_ids_`. Prediction methods can
access call-local `_prediction_info`, which is removed after success or failure.
Constructor parameters remain unchanged, including after cloning.

An explicit constructor rate or ordered channel list that disagrees with `info`
raises. Prediction information must match the fitted rate and ordered channels;
known incompatible units also raise. Omitting prediction information retains
ordinary shape validation but cannot verify facts absent from the call.

Plain `fit(X, y)` and `predict(X)` continue to work in sklearn pipelines,
`cross_val_score` and aeon entrypoints. Native classifiers imported from aeon
retain their own interfaces. Passing information through arbitrary pipeline/CV
routing is separate from the manual aligned selection shown here.

`BraindecodeClassifier` and EEGNet share optional early stopping through
`validation_size`; its default `None`
retains training on all supplied cases. With information, validation holds out
entire known subjects from the current fitting collection. Both partitions must
contain every fitting class. Missing subjects, too few subjects or a
class-confounded split raise without falling back to random cases. Without
information, validation uses stratified cases. For a within-person question,
explicitly set
`validation_strategy="stratified"` to retain metadata while stratifying cases.
`validation_strategy="subject"` requires known subjects even without the default
inference rule. Both strategies reject overlapping known original source support.
Stratification alone does not establish trial/group independence; retain an
explicit trial-aware outer policy and use independently constructed cases.
Array-only information has no verified original-source identity to check. The best validation-loss weights are restored. `training_indices_`,
`validation_indices_`, `validation_policy_`,
`validation_history_` and one-based `best_epoch_` record the realized split and
selection. The outer test set is never supplied to this process.

The merged generic Braindecode adapter uses the same base and validation loop
as its existing wrappers. Its default 200 epochs and incomplete training-batch
policy remain unchanged. Every validation/prediction row is included, and fitted
information and selected weights survive pickle restoration.

For ordinary models, information-derived channel names carry no position claim.
Explicit `channel_names`, and spatial models requiring positions such as DGCNN,
use the adapter's existing `standard_1005` template policy. These are template
coordinates, never measured BIDS electrodes. `channel_geometry_` records
`names_only` or `standard_1005` when the model accepts channel information.
Supply explicit constructor channel names to request template positions for an
ordinary model. Unknown template names raise; nonstandard information-derived
names remain usable by models that do not require positions.

Other consumers can call `BaseEEGClassifier.validation_split` inside `_fit`.
The shared base does not add early stopping to estimators that do not use it.

## Preserve provenance in saved outputs

`EEGDatasetInfo(info.to_dict())` restores detached portable state, including the
optional version-1 current view. Restoration validates identities, source support,
channel/rate/sample layouts and history before use. Original inspection reports
without the extension remain valid.

`info.to_ts_header()` formats that state as `.ts` comment text. Pass it as the
`header` argument to `aeon.datasets.save_to_ts_file`, which writes the structural
header and data. Comments preserve source context and history; they do not alter
the data format or prove unchanged array contents. The exported state includes
source tables and paths, so choose the source context appropriate for the output.


## Add classifiers and source types

A new EEG classifier inherits `BaseEEGClassifier` and implements the ordinary
`_fit(X, y)`, `_predict(X)` and optional `_predict_proba(X)` methods. Read fitted
`info_`, `sfreq_`, `channel_names_` and `subject_ids_` when needed. The shared base
checks metadata and calls aeon's lifecycle; models do not reread BIDS sidecars or
add information parameters to protected methods. New Braindecode architectures
can use the existing generic adapter, with explicit model input requirements.

Regular multichannel case arrays and explicit units/identities are usable through
`from_arrays` without BIDS. The native inspector and `RecordingInfo` remain EEG
and EDF/BDF-specific. Additional modalities need appropriate readers, channel
types, physical units and geometry policies; names alone must not imply EEG
positions. Pipeline/CV metadata routing remains a separate integration boundary.
Keep source acquisition adapters separate from the current case layout and
estimator contract when adding those capabilities.
