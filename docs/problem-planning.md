# Define large-dataset problems before fetching signals

A problem definition should not require downloading a complete signal archive.
Use three explicit stages with the same [definition and binding vocabulary](problem-format.md):

```python
from aeon_neuro.datasets import (
    resolve_classification_problem,
    plan_classification_problem,
    materialize_classification_problem,
)

# No recording metadata or signal files are needed for definition validation.
definition = resolve_classification_problem("problem.json", "binding.json")

# Only small descriptors and event tables are needed for an exact case plan.
plan = plan_classification_problem(
    catalogue, "problem.json", "binding.json", events=event_rows
)

# Later, when preparing training, explicitly acquire/read selected recordings.
dataset = materialize_classification_problem(
    plan, open_recording, case_ids=selected_case_ids
)
```

The first two stages never invoke a signal opener or download a recording.
Planning hashes only its small implementation files, not dataset signal files.
Definitions can be saved before all source facts are available; exact planning
requires the descriptor fields below rather than guessed sample counts.

## Plan from shared BIDS information

The same planner accepts `EEGDatasetInfo`. Inspection supplies the source facts
without loading signal arrays; the object supplies events and reported history:

```python
from aeon_neuro.datasets import (
    EEGDatasetInfo,
    inspect_bids_eeg,
    open_bids_eeg_recording,
)

root = "/data/study"
info = EEGDatasetInfo(inspect_bids_eeg(root, subjects=["01", "02"]))
plan = plan_classification_problem(info, "problem.json", "binding.json")
dataset = materialize_classification_problem(
    plan,
    lambda descriptor: open_bids_eeg_recording(root, descriptor),
    case_ids=[plan["cases"][0]["case_id"]],
)
```

Do not pass a separate `events` argument with the object. Event epochs use its
events; fixed windows use recording facts without passing events to the case
rules. Participant targets still use explicit columns such as `participant.group`
and explicit class mappings. This adds no target inference or second reader.
An existing saved inspection report can be planned even while signals are absent;
reading its headers initially still belongs to source inspection.

Shared objects produce **version-2 plans** containing `input_info`, a detached
source report with any reported history. Plain descriptor/event inputs continue
to produce **version-1 plans**. Both retain the same case inventory fields and
use the same enumeration and validation rules as eager construction. Object
plans additionally fingerprint the metadata implementation and package version
file. Replaying a plan regenerates it from its authoritative inputs and compares
the whole result before calling a reader. Changing an event, source descriptor,
history record, case, configuration or implementation invalidates the saved plan.

Plan versions are distinct from source-report, provenance and authored-document
versions. Version 1 here keeps the earlier plain-input layout, not an automatic
migration of old artifacts. Saved plans must be regenerated after an implementation
fingerprint changes, retaining their authored definitions and source evidence.

## Eager and deferred provenance

When the opener supplies the same recording signals as the eager route, full
materialization produces the same samples, case identities, labels, groups and exclusions as
`make_classification_problem(X, info, problem, binding)`. The combined
`metadata["provenance"]` uses identical `source`, `history` and `problem` sections.
Its `construction` identifies the actual eager or deferred entrypoint and package
version. Execution through different routes is recorded explicitly.

`metadata["materialization"]` retains the plan ID, selected case IDs, unselected
case count and source-verification boundary. Source/history provenance keeps the
full planned input context even when selected cases require only one recording.
Earlier records may describe processing over a broader population; selecting
fewer cases does not narrow those earlier claims. `metadata["recordings"]` lists
only the recordings actually opened. Returned arrays and metadata are detached
from the plan and input object.

Plain-input materialization retains its separate `metadata["problem"]` contract.
Reported history is metadata, not an executable preprocessing pipeline. The
native BIDS opener returns source signals. If eager arrays were filtered or
scaled externally, an explicit opener must reproduce that signal state for sample
parity; the materializer neither applies nor verifies those reported operations.
Layout-changing transformations still need a separately supported mapping.
Neither route certifies arbitrary earlier processing or source content from a
configuration hash. The explicit source adapter remains responsible for content
verification. A saved plan is an inspection snapshot, not a signed certificate.

## A small source catalogue

Each recording descriptor supplies a stable `recording_id` (or `filepath`),
ordered `channel_names`, `sampling_frequency` and exact positive `n_samples`.
The existing binding reads its recording-level target/group/identity fields from
these descriptors. Event tables supply bound onset, event ID and target fields.
Descriptors and event tables must be JSON-compatible.

Retain pinned dataset/release identifiers, source URLs or asset references,
expected checksums, source sizes, units and prior processing in these ordinary
metadata records. The planner preserves them without treating a declared checksum
as a verification. Catalogue acquisition belongs to a source adapter: it can use
small release manifests, BIDS tables/sidecars or bounded header reads. Not every
source publishes enough information for an exact preview. Missing required facts
block planning while definition validation remains usable.

This API does not discover an OpenNeuro dataset from its ID or interpret arbitrary
remote directory layouts. Existing download and metadata adapters can supply
the catalogue; remote signal protocols do not belong in the reusable problem.

## Inspect and select cases

The plan records included cases, exclusions, output layouts, effective documents
and a `plan_id` bound to metadata, source references and implementation. It can
be round-tripped through JSON and retained beside the authored definition.

`dense_shape` is available when all cases have one channel order, sampling rate
and length; otherwise it is `null`. Heterogeneous catalogues can be inspected
before selecting a compatible subset for materialization. Group/class coverage
and actual evaluation assignments remain separate checks.

`estimated_case_bytes_float64` estimates the extracted arrays at eight bytes
per value. It is not the download size, decoder memory usage or an upper bound
on process RAM. The returned case inventory itself consumes metadata memory;
very large inventories may need narrower source/event selection.

Pass a nonempty unique list of eligible `case_ids` to select membership. Rows
remain in canonical plan order. Unknown or excluded IDs fail before opening
anything. A selected subset must have compatible channels and rate, with no
implicit resampling. Unselected recordings are not opened by the materializer.
The original plan retains the full inventory and its exclusions.

## An explicit acquisition and reader adapter

`open_recording(descriptor)` is called only during materialization, once for each
selected recording. It returns a newly owned public `mne.io.BaseRaw` object.
A source adapter should:

1. Resolve the pinned source release and the selected recording's required assets.
2. Fetch missing assets into a local cache and verify declared expected checksums.
3. Return an MNE reader, preferably with `preload=False` where the format supports it.

The materializer verifies decoded channel order, rate, sample count and an
explicit `first_samp` if supplied, then requests only selected intervals and
channel names with `get_data`. It checks returned shapes and finite real-numeric arrays,
retains native values and closes every handle on success or error. A changed
plan or implementation is rejected before invoking the adapter.

Fetching/caching is deliberately delegated rather than adding another download
client. Full content verification remains the adapter's responsibility; decoded
header agreement alone cannot prove it. No default opener or implicit network
access is provided. A format can require downloading/preloading a complete
selected recording even when only one short interval is selected. Companion
files must be handled by the adapter. Remote chunk streaming is outside this
first boundary.

The returned `ClassificationDataset` records selected IDs, plan identity and
the verification boundary. Case arrays are materialized in memory; this is not
out-of-core classifier fitting. Earlier source transformations, class meaning,
eligibility decisions and independence assertions still require appropriate
evidence. Validate the actual training/test assignments before fitting.

## Runnable demonstration

```bash
python examples/metadata_first_problem.py
```

The first part builds an eight-case plan from a hypothetical catalogue declaring
over 128 GiB of absent source files; its extracted float64 case estimate is 32 KiB.
It performs no download, source read or source-file hashing. This is a metadata
size demonstration, not a claim that those remote recordings were validated.

The second part creates tiny local FIF fixtures separately, shows that planning
leaves the training cache empty, then copies only one selected recording into
that cache at materialization. It checks exact case samples and repeat cache
reuse. Local copies model acquisition; no network transfer or large-data speed
benchmark is claimed.
