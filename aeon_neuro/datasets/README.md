# Dataset utilities

## Aligned EEG information

`EEGDatasetInfo.from_arrays` records explicit rate, ordered channels, optional
units, subjects and groups for dense or variable-length collections. Direct
construction accepts optional source/dataset metadata without reading files.
Paired selection copies signals and information in the same order; `to_dict`
provides restorable detached state. See [the information guide](../../docs/eeg-information.md).
