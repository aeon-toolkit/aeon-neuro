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
