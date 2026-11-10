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
