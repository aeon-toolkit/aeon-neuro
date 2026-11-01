# Reproducible classification records

Recordings do not determine one classification problem. A recipe supplies the
target, eligible intervals, cases, processing and generalisation question.
The record helpers retain those choices around `ClassificationDataset` and
ordinary aeon/sklearn estimators, following [#247](https://github.com/aeon-toolkit/aeon-neuro/issues/247).

## Responsibilities

| Component | Responsibility |
| --- | --- |
| Loader and preparation recipe | Acquire recordings, resolve target columns and class mappings, select cases, perform declared processing, retain exclusions and original-source coordinates |
| `ClassificationDataset` | Carry aligned signals, labels, groups, cases, exclusions and output metadata |
| Problem record helpers | Resolve known output metadata, retain explicit choices/origins, fingerprint cases/signals, resolve saved assignments and check reconstruction |
| Evaluation code or existing framework | Check capabilities, clone and fit pipelines inside saved folds, select candidates using training data, compute outputs and supply effective settings |
| Experiment record helper | Link caller-supplied settings and outputs to the exact problem and assignments; reject inconsistent output references |
| Caller/companion tools | Persist records and arrays, compute source-file hashes, install environments and orchestrate replay |

These helpers neither infer a scientific target nor execute a saved Python object.
`validate_group_split` keeps its conservative behaviour; policy-aware validation
continues to use `validate_classification_split`.

## Define and reconstruct a problem

The complete, executable example is
[`examples/reproducible_classification.py`](https://github.com/aeon-toolkit/aeon-neuro/blob/main/examples/reproducible_classification.py).
From a source checkout with dependencies installed:

```sh
python examples/reproducible_classification.py /tmp/classification-records
```

The output directory must be new. The example generates recordings, constructs a
cohort/window problem and a within-person event/session problem, then independently
reconstructs them with reversed storage order. It selects a logistic-regression
pipeline using saved inner folds, evaluates the selected settings on held-out
cases, and records an aeon majority baseline on those same held-out cases. The
scores demonstrate interfaces only; they are not EEG research results.

The public boundary is:

```python
problem = make_classification_problem(
    dataset, definition, splits=splits, sources=current_source_digests
)
# Save problem with ordinary JSON; retain the source/array artifacts separately.

fresh_dataset = existing_preparation_recipe()
resolved = validate_classification_problem(
    fresh_dataset, problem,
    definition=current_definition,
    sources=recomputed_source_digests,
)
fit_indices = resolved["outer"]["partitions"]["development"]["indices"]
```

`definition` requires:

- `dataset`: identity, release/version and explicit recording selection.
- `target`: source description and a list of `{value, label}` mappings. A list
  avoids losing the distinction between integer and string values in JSON keys.
- `case_definition`: timing/selection description and explicit exclusions policy.
- `processing`: ordered named preparation steps; `[]` means none added.

Ordered `channels` and `sampling_frequency` are derived from known output metadata
when omitted. Explicit values must agree with known metadata. Units come only
from output metadata or an explicit declaration; missing units remain `null`.
Units may be shared or a channel-aligned list with unknown entries. Acquisition
facts can remain in the original metadata rather than being guessed or copied.
Origins distinguish `explicit`, `metadata` (with a source) and `unknown` values.
Additional JSON fields can retain study-specific choices and limitations.

`sources` maps stable source names to SHA-256 digests and descriptions. Include
recordings, participant/event tables and the preparation recipe as appropriate.
The caller must independently hash current source bytes: reusing an old digest
does not check a source. The helpers do not open paths or download data.

## Saved assignments and scope

Split names and partition names have no implicit meaning. Each saved split has:

```python
{
    "partitions": {"fit": ["case-a", "case-b"], "check": ["case-c", "case-d"]},
    "policy": {
        "disjoint_by": [["participant_id"]],
        "source_fields": ["recording_id", "start_sample", "stop_sample"],
        "require_all_classes": ["fit", "check"],
    },
    "scope": {"split": "outer", "partition": "development"},
}
```

These short IDs illustrate the format; real cases must satisfy the declared
class coverage and identity policy. Omitted scope means all included cases.
Inner scope explicitly references a parent partition, preventing held-out cases
from entering the inner split. Parent references may appear in any dictionary
order; cyclic references are rejected. Assignments contain IDs, not seeds.

`resolve_classification_split` also works independently when only saved IDs and
a split policy need validation. It does not assert unchanged samples/settings.

For new-person evaluation, declare participant independence. For within-person
sessions, deliberately declare a compound participant/session identity and role
restrictions. For paired players, preserve pair independence. Use original source
coordinates for overlap checks; processed array coordinates may be different.
Trial-local coordinates require a source identity that includes the trial.

## Experiment records and model selection

After fitting ordinary estimators, call `make_classification_experiment` with
the problem, estimator description, effective parameters, software versions and
evaluations. Each evaluation names its split and fitting/scoring partitions and
supplies aligned predictions and computed metrics. Optional probabilities require
the complete fitting class order. Exact fitting/scoring IDs are copied into the
record. Seeds belong in the supplied parameters, including an explicit `None`
when unset; an unset seed does not guarantee repeatability.

The example evaluates two candidate settings only on saved inner folds. It records
each candidate run, chooses the largest mean inner balanced accuracy (with a
declared tie rule), and links the selected run and all candidate record IDs in
the final experiment. The record helper validates these references structurally;
it does not fetch candidate records, verify the selection rule or tune a model.

Capability checks and fitting stay with evaluation code. Learned scaling, CSP,
feature selection and other learned transformations must be inside each fold's
pipeline. No implicit flattening, padding, resampling or imputation occurs in the
record helpers. They cannot inspect hidden estimator CV or certify preprocessing
that was already performed on the entire dataset.

## Identity, storage and compatibility

Records use `schema_version=1`, a kind and a SHA-256 `record_id` over canonical
JSON content. Unknown schema versions and modified records are rejected.
Changes to interpretation require an explicit schema-version decision and migration
guidance; the helpers do not silently upgrade old records.

Problem records keep cases/exclusions sorted by identity and omit only the
mechanical `row` field. Signal fingerprints include dtype, shape and exact bytes.
Reordering cases is allowed; changing samples, labels, groups, source bounds,
metadata, settings, exclusions or source evidence fails reconstruction. Identical
values in another dtype or with different NaN bit patterns are different signals.
Fingerprinting reads every included sample, one case at a time; it is not a cheap
metadata-only operation and does not retain another complete signal collection.

JSON records contain no fitted pickle or executable object. Environment versions
and supplied source identities are evidence, not an environment installer or
guarantee of bitwise equality across machines. Use explicit artifact storage and
replay tooling when needed.

## Dataset preparation integration

Existing COVID19/RPS eligibility and processing recipes stay unchanged. Supply
their final `ClassificationDataset`, original source bounds, saved case assignments
and freshly computed source/recipe digests. Prepared archive importers may construct
that container directly, provided arrays, labels, groups and metadata agree.
Incomplete exclusion evidence must remain explicitly described in the definition.

BIDS labeling/preprocessing work such as
[#243](https://github.com/aeon-toolkit/aeon-neuro/pull/243) should supply the same
constructed cases and label provenance. It need not duplicate split or record
logic. Feature composition under
[#245](https://github.com/aeon-toolkit/aeon-neuro/issues/245) stays an ordinary
estimator/pipeline, whose fitting scope and effective settings the caller records.

See the [proposal and evidence](classification_records_proposal.md) for the
upstream scope and follow-ups under discussion.
