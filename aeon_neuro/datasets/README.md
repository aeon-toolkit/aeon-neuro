# Classification problems on loaded recordings

For large datasets, use `resolve_classification_problem` without signals,
`plan_classification_problem` over a small metadata catalogue, then explicitly
call `materialize_classification_problem` to open selected sources for training.
The source opener can fetch and cache selected recordings at that boundary.
See [metadata-only planning](../../docs/problem-planning.md).

`load_classification_problem(recordings, metadata, problem, binding, events=...)`
constructs the existing `ClassificationDataset` from explicit problem choices.
Definitions and bindings can be dictionaries or JSON filenames. Fixed windows
use recording targets; event epochs can use event or recording targets.

The first format is experimental. It reuses the existing case builders and adds
no runtime dependencies. The [format reference](../../docs/problem-format.md)
lists every supported field and its constraints. See the
[loaded-problem guide](../../docs/problem-loading.md) for usage examples.

Run `python examples/problem_loading.py` from a source checkout for a generated
example with two event targets, a cohort-window problem and a separately
validated participant-held-out split.
