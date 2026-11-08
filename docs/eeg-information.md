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
Unknown fitting facts remain unknown.

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

## Executed processing

`info.process(X, operation, **parameters)` returns detached arrays and updated
information together. It supports ordered channel selection, explicit scaling,
exact-grid crop, polyphase resampling, band-pass filtering and referencing.
Failures leave arrays and metadata unchanged. No operation fits a learned
transform; learned preprocessing still belongs inside training folds.

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

## Optional inner validation and early stopping

BraindecodeClassifier and EEGNetClassifier accept `validation_size`, `patience`
and `validation_strategy`. The default validation_size=None retains training on
all supplied rows for every configured epoch and the merged training-batch policy.
With information, auto validation holds out whole known subjects; without it,
validation stratifies cases. Explicit stratified validation supports within-person
questions with retained metadata. Missing subjects, class-confounded partitions
and known source overlap raise without changing seeds or falling back silently.
Both partitions must contain every fitting class, even when stratification
succeeds. Label count and dimensionality must match the fitting collection.

```python
clf = EEGNetClassifier(
    n_epochs=20, validation_size=0.2, patience=5, random_state=42,
)
clf.fit(X_train, y_train, info=info_train)
```

Only the collection supplied to fit is split. Realized indices, validation policy,
loss history, best epoch and restored best weights remain available and survive
pickling. Default training avoids a redundant full input copy; fixed validation
tensors are built once per fit. Validation and prediction include every row.
Subject disjointness cannot override known original signal overlap, and
stratification alone does not establish independent trials/groups.
