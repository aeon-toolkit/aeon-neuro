# Classification problems on loaded recordings

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
