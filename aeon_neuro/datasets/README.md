# Classification problems on loaded recordings

`inspect_bids_eeg` reads local EDF/BDF headers and BIDS sidecars without loading
signal arrays. `load_bids_dataset` loads those sources with `EEGDatasetInfo`,
preserving dataset/participant facts, channel/event tables and source snapshots.
`info.select(X, selector)` copies signals and their metadata together.
`info.with_history(...)` appends caller-reported processing to a detached view;
it does not perform processing or update derived layouts.
See the [source inspection guide](../../docs/bids-problem-input.md) and
[loaded metadata guide](../../docs/bids-dataset-info.md). Both routes reuse the
existing source reader and preserve the legacy `load_bids_data` interface.

`load_classification_problem(recordings, metadata, problem, binding, events=...)`
constructs the existing `ClassificationDataset` from explicit problem choices.
Definitions and bindings can be dictionaries or JSON filenames. Fixed windows
use recording targets; event epochs can use event or recording targets.

The first format is experimental. It reuses the existing case builders and adds
no runtime dependencies. The [problem guide](../../docs/problem-format.md)
starts with runnable examples and lists every supported field and constraint,
including optional identities and explicit pandas target-code conversion.

Run `python examples/problem_loading.py` from a source checkout for a generated
example with two event targets, a cohort-window problem and a separately
validated participant-held-out split.
