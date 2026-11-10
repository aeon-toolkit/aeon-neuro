# Dataset utilities

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
