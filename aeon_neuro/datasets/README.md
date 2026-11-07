# Shared EEG information and classification cases

`inspect_bids_eeg` reads local EDF/BDF headers and sidecars without loading signal
samples. `load_bids_dataset` reuses the source opener to return recordings with
`EEGDatasetInfo`, preserving dataset/participant facts, channels, events, source
geometry and file snapshots. The existing `load_bids_data` interface remains
available. See the [source inspection guide](../../docs/bids-problem-input.md)
and [loaded metadata guide](../../docs/bids-dataset-info.md).

The existing `make_recording_cases`, `make_fixed_length_cases`, `make_event_cases`
and `build_classification_dataset` accept shared information as their metadata
argument. Case labels/groups remain explicit. The builder retains detached
source/history/construction provenance and exposes case-aligned `dataset.info`;
no authored problem definition or JSON schema is required.

`info.select(X, selector)` copies signals with their matching information.
`info.process` executes supported channel selection, scaling, crop, resample,
filter and reference operations, updating current layouts and history together.
`with_history` records an external account without claiming an operation ran.
Original acquisition facts remain separate from current channels, rate, units
and sample support. Logical case-channel aliases preserve original channel picks.

The [EEG information guide](../../docs/eeg-information.md) covers case construction,
optional estimator calls, subject validation, detached table views, serialization
and `.ts` history comments. Direct `info.current_metadata` and
`info.events_for_cases()` remain available for plain-descriptor integrations;
pass the object itself to retain its complete source/history context.

Source events are retained before scientific selection. Negative/outside onsets
produce warnings; off-grid events keep an unknown exact sample coordinate.
Case extraction still requires explicit, exact bounds. EEG JSON inheritance
uses the same parsed-byte snapshots as the other inspected metadata.

Known original source support is enforced by split validation and classifier
inner validation. Pointwise recording operations preserve exact case intervals;
filtering/resampling retain conservative input support. Single-case subject
lookup indexes the current row directly rather than scanning the collection.

Resampling executes the requested rational rate ratio through SciPy, keeping
the source grid independent of output-count rounding. Reduced factors above
100,000 are rejected before filter allocation. Inner validation checks class
coverage on both sides of stratified and subject-based splits; a rare class
lost through rounding blocks fitting instead of changing the declared split.

## Authored definitions and planning

`load_classification_problem` constructs cases from loaded arrays, descriptors
and explicit problem/binding documents. `make_classification_problem` accepts
`EEGDatasetInfo` directly, retaining source/history and effective configuration
under `dataset.metadata["provenance"]`. Dictionary and JSON-file inputs use the
same experimental rules for explicit targets, groups, channels and timing.
See the [problem guide](../../docs/problem-format.md) for every supported field.

`resolve_classification_problem` validates documents before source discovery.
`plan_classification_problem` uses descriptors/events or shared information to
produce a metadata-only case inventory. `materialize_classification_problem`
opens only selected sources through an explicit reader, retaining the same
source/history/problem provenance and the realized selection separately.
The [planning guide](../../docs/problem-planning.md) explains memory, source
verification and stale-plan checks. Planning does not fetch signal arrays.

Run `python examples/problem_loading.py` and
`python examples/metadata_first_problem.py` for generated examples covering event
and cohort targets, explicit splits and deferred source opening. These APIs add
no runtime dependencies and reuse the source and case contracts above.
