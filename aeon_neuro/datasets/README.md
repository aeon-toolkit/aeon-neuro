# Dataset utilities

## NEMAR acquisition

`fetch_nemar_dataset` downloads an exact release and explicit EEG selection,
including compatible BIDS metadata and declared signal companions. Completed
selections are verified locally on reuse. Entity labels must match complete
alphanumeric tokens; malformed labels are excluded from subject/task selection.
BrainVision companion references come from the header's Common Infos section.
See [the dataset guide](../../docs/datasets.md#nemar-bids-recordings) for installation,
offline caches and reading EEGLAB files. Development installs include nemar-py
so the offline provider integration tests run alongside the ordinary suite.

## Aligned EEG information

`EEGDatasetInfo.from_arrays` records explicit rate, ordered channels, optional
units, subjects and groups for dense or variable-length collections. Direct
construction accepts optional source/dataset metadata without reading files.
Paired selection copies signals and information in the same order; `to_dict`
provides restorable detached state. See [the information guide](../../docs/eeg-information.md).
Both information classes list their public properties, types and optional values
in their class docstrings. The guide's
[dictionary reference](../../docs/eeg-information.md#dictionary-structure)
documents source state and current views together, including their different
selection scopes.
Fixed internal records use named dataclass fields; public properties and the
versioned dictionary format retain their detached values. Flexible source JSON,
table rows and custom metadata fields remain available without schema coercion.

Source participant IDs may be unknown (`None`), including in a collection with
known IDs. Aligned rows retain these unknowns; the serialized selected-participant
scope lists only known string/integer IDs, preserving their types through selection.

Source-only selection preserves omitted optional diagnostics and filters supplied
diagnostics to the selected recordings and dataset scope. Restored sample grids
must lie inside their half-open source support `[start, stop)`, excluding `stop`.
Declared case participants may differ from original recording subjects, such as
event-level actors. Unknown identities and independent groups remain supported.
NumPy integer identities and layout numbers normalize to Python values for
portable state. Grid checks reject overflow, and malformed diagnostics fail during construction. Empty collections
still validate dense channel/sample axes, numeric types and supplied units.


## Local BIDS sources

`inspect_bids_eeg` inspects supported EDF/BDF source headers and sidecars.
`load_bids_dataset` loads recordings with shared information through the verified
source opener. Existing case helpers accept that information and retain source
provenance in `dataset.info`. Explicitly supplied channel rows remain available
when an unrelated sidecar path is supplied; only an explicit absent channel
sidecar marks those rows unavailable. See [BIDS loading](../../docs/bids-dataset-info.md).

## Executed EEG processing

`info.process` keeps arrays, current layouts, original source support and executed
history aligned for channel selection, scaling, crop, resampling, filtering and
reference operations. It uses exact requested resampling ratios with bounded
filter factors. Changing the rate requires at least two input samples per row;
shorter rows raise `ValueError` without modifying arrays or information. See [the information guide](../../docs/eeg-information.md).

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
