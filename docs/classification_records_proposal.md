# Classification record proposal and evidence for #247

This is a proposal for maintainer review, not a claim that the schema or ownership
boundary has already been accepted. It follows
[#247](https://github.com/aeon-toolkit/aeon-neuro/issues/247), building on merged
case construction (#241) and explicit split-policy validation (#248).

## Proposed upstream scope

Accept four helpers in `aeon_neuro.datasets`: resolve saved case IDs, construct a
resolved problem record, compare fresh construction against it, and retain a
caller-executed experiment. Include executable examples, focused tests and the
[record contract](classification_records.md).

Keep acquisition, target discovery, dataset-specific preprocessing, array storage,
archive indexing, model selection execution, metrics computation and replay
orchestration with recipes, existing frameworks or companion tools. No companion
package dependency, CLI, classifier factory, benchmark runner or storage layout
is introduced into aeon-neuro by this proposal.

Questions to resolve in review:

1. Are these helper names and their location in `datasets` appropriate?
2. Are the required definition fields, explicit unknowns and origin vocabulary
   sufficient without imposing a study-specific schema?
3. Should signal fingerprinting and caller-declared experiment records be part
   of this first upstream boundary, or split into a follow-up contribution?
4. Does the scope-reference representation fit BIDS/preparation and external
   evaluation callers without requiring either to adopt a new runner?

## Reproducible upstream acceptance examples

The source-tree example and tests require generated signals only:

1. Cohort-labelled windows and event/session-labelled cases use the same handoff,
   with different explicit independence policies.
2. Fresh preparation with reversed case order resolves saved assignments correctly;
   changed sources, settings, cases and signals are rejected in tests.
3. A sklearn feature pipeline and an aeon classifier consume the same cases and
   saved assignments and produce linked settings/output records.
4. Candidate selection uses only saved inner folds. Tests observe the exact
   feature arrays supplied to learned scaling and exclude outer held-out cases.

The existing PR example workflow is extended to execute the source-tree example.
Unit tests also exercise it where the source-tree example is available. Passing
local checks must not be described as passing hosted CI; the PR's check results
are the evidence for the latter.

## Companion prototype evidence

Before extracting these helpers, the companion prototype was tested locally on
2026-11-01 against existing prepared releases. These are reported local results,
not independent CI checks or a newly executed original-recording reconstruction.
The prototype's directory layout/schema is not the proposed public contract.

| Prepared release | Included cases | Outer + inner assignment checks | Selected model-check cases |
| --- | ---: | ---: | ---: |
| COVID19 | 36,535 | 6 | 344 |
| RockPaperScissors | 29,511 | 6 | 186 |
| EEGMAT | 2,160 | 6 | 144 |
| AlexMI | 320 | 4 | 32 |
| Weibo2014 | 1,580 | 5 | 40 |

Ten complete TRAIN/TEST archives (19,345,959,726 bytes) matched their supplied
publisher checksums. Every embedded label matched the case table. All 70,106
case assignments passed 27 policy validations. Numeric model checks used the
declared 746 selected cases, not all samples in the full releases.

Two classifiers ran and replayed across every saved fold and held-out partition
for all five derivatives (108 fits), producing byte-identical prediction/metric
files. CSP + LDA and aeon 1-NN also ran and replayed on the AlexMI derivative
(16 fits). Separate generated cohort/session examples reconstructed and replayed
four classifiers (40 fits). The companion suite had 38 passing tests.

COVID19 retained 4,120 enumerated excluded candidates and RPS retained 249.
The other three releases did not supply complete excluded-case tables: zero
enumerated entries must not be interpreted as no exclusions.

The companion's aggregate verification records and implementation fingerprints
are included in [the evidence snapshot](classification_records_evidence.json).
They identify the original local checks without transferring recordings or model
artifacts into the library repository. Publishing the complete standalone companion
and its reproduction instructions is tracked below; the aggregate snapshot alone
does not make those five release checks independently reproducible.

The proposed upstream helpers were additionally exercised on all five materialized
derivatives (746 cases, 27 assignment checks). Each record was rebuilt with reversed
case order and passed reconstruction validation. These results appear separately
as `upstream_helper_checks` in the evidence snapshot.

## Explicit follow-ups

These identifiers are discussion checkpoints, not already-open GitHub issues.
Maintainers can assign owners and convert accepted items to issues without
delaying the small shared contract unnecessarily.

| ID | Work and completion evidence | Relationship to closing #247 |
| --- | --- | --- |
| R1 | Agree public fields, helper names, schema-version policy and the responsibility boundary in PR review | Required |
| R2 | Merge the agreed implementation/examples after applicable CI passes | Required |
| R3 | Link the merged PR, acceptance examples, evidence and accepted follow-ups from #247 | Required |
| F1 | Publish the standalone companion and portable instructions for independently repeating the five prepared-release checks | Evidence distribution follow-up; scope/owner to agree |
| F2 | Connect the eventual #243 output to the shared case/record boundary, preserving label provenance and original source coordinates | Adapter follow-up; coordinate before claiming compatibility |
| F3 | Rebuild each real dataset from original sources and compare cases, exclusions, settings and signals; record incomplete evidence explicitly | Dataset-specific validation, not required by the initial generated acceptance examples |
| F4 | Run full-dataset evaluation with declared model selection, resources and reporting units | Separate benchmark experiments; subset scores do not satisfy this |
| F5 | Integrate record emission with an existing evaluation framework if repeated examples establish a reusable need | Optional follow-up; avoid a parallel benchmark framework |

The proposed closing condition is maintainer acceptance plus merged, CI-validated
helpers/examples and an issue-linked account of evidence and remaining work.
This document does not close the issue or substitute for those decisions.
