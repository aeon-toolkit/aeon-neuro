# Validating classification assignments

For a dict/JSON definition over already loaded recordings, start with
[loading a classification problem](problem-format.md). Construction returns
the same case container used by the validators below.

Recordings and classification cases do not choose an evaluation protocol.
Testing new people, testing new sessions from known people and testing future
intervals require different explicit independence rules.

`validate_group_split` provides conservative independent-group validation.
`validate_classification_split` checks supplied partitions against a declared
policy without generating a split or fitting a model. Both consume the existing
`ClassificationDataset`; neither requires a recipe format or a classifier.

## A within-person session example

The following generated example assigns both classes from session T to training
and both classes from session E to testing. Reusing a participant is intentional.
For a new-person question, declare participant independence instead.

```python
import numpy as np

from aeon_neuro.datasets import (
    build_classification_dataset,
    make_fixed_length_cases,
    validate_classification_split,
)

recordings = np.arange(32, dtype=float).reshape(2, 2, 8)
metadata = [
    {
        "recording_id": session,
        "channel_names": ["C3", "C4"],
        "sampling_frequency": 4,
    }
    for session in ("T", "E")
]
cases = make_fixed_length_cases(
    recordings, metadata, labels=[0, 0], groups=["T", "E"], window_size=1
)
for index, case in enumerate(cases):
    case.update(
        label=index % 2,
        participant_id="P1",
        session=case["recording_id"],
    )
dataset = build_classification_dataset(recordings, metadata, cases)
assignments = validate_classification_split(
    dataset,
    {"train": [0, 1], "test": [2, 3]},
    disjoint_by=[("participant_id", "session")],
    source_fields=("recording_id", "start_sample", "stop_sample"),
    partition_values={"train": {"session": ["T"]}, "test": {"session": ["E"]}},
    require_all_classes=("train", "test"),
)
assert assignments["partitions"]["train"]["case_ids"] == [
    case["case_id"] for case in dataset.cases[:2]
]
```

The compound identity `(participant_id, session)` must not cross partitions.
Separate rules, such as `[("participant_id",), ("trial_id",)]`, enforce each
identity independently. Missing identities fail. Partition-value constraints
reject reversed T/E roles, while class coverage is checked only for the requested
partitions. The result records the effective policy, scope and ordered case-ID
assignments using ordinary Python values.

## Original-source overlap and inner validation

`source_fields` explicitly names the source identity and half-open sample bounds.
The example uses distinct original recordings. If different recording IDs refer
to copies or crops of one source, supply a canonical original-source field and
bounds on its original coordinate grid, for example
`("original_source_id", "original_start_sample", "original_stop_sample")`.
These fields can be retained in case records by the existing case builder.
Intervals from one original source may touch across partitions but must not
overlap. Different channel banks do not remove shared-source support.

The validator trusts these identity and coordinate assertions. It does not hash
files, recover unknown participant links, map transformed coordinates or infer
acquisition chronology. Include filter/sequence context in source support where
needed, or use stronger grouping that keeps that context together.

For inner validation, pass `eligible_indices` containing only the parent training
indices. The supplied partitions must cover exactly that scope, so adding an
outer test case fails. The caller must establish the correct parent scope and
fit learned transforms and model selection only within the appropriate training
partition. Validation does not inspect an estimator's internal CV or certify
preprocessing performed before case construction.

The returned assignment snapshot is an inspection record, not a persistent
certificate. Revalidate against the current dataset before fitting or reusing
assignments after any changes.
