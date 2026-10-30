# EEG input and existing BIDS recordings

`convert_eeg_to_bids` accepts native EEG filenames, MNE Raw objects and existing
BIDS recordings identified by `mne_bids.BIDSPath`. It returns the written
`BIDSPath`. Conversion retains all channels and performs no filtering,
resampling, re-referencing, epoching or classification-label selection.

## Existing BIDS input

Download the desired dataset release, including the signal and applicable
metadata files. An OpenNeuro webpage URL is a source reference, not a converter
argument. Call the converter once per recording:

```python
from mne_bids import BIDSPath
from aeon_neuro.utils.bids import convert_eeg_to_bids

source = BIDSPath(
    root="data/ds008082",
    subject="CUCOV001",
    task="COVID",
    datatype="eeg",
    suffix="eeg",
    extension=".edf",
)
destination = source.copy().update(root="data/covid-subset")
written = convert_eeg_to_bids(source, destination)
```

For OpenNeuro ds006761, use the downloaded RPS dataset root, `task="RPS"`,
`extension=".bdf"` and the desired pair ID, for example `subject="31"`. An
RPS subject identifies a pair. Preserve all native channel identifiers and
published participant fields, then establish which acquisition groups belong to
which player before downstream channel selection. Native layouts can differ:
the pinned pair 31 file has 129 channels, while pair 11 has 285 channels with
four acquisition prefixes and auxiliary inputs. Those prefixes alone do not
establish the number or identities of recorded players.

This route requires the source identifiers and encoding to remain unchanged,
`format="auto"`, `allow_preload=False` and `overwrite=False`. Original event
tables are retained verbatim, so `events` and `event_id` must remain `None`;
explicit overrides raise before output is created. Source and destination
datasets must use separate folders, with neither nested inside the
other. Native files that necessarily preload, such as embedded EEGLAB data,
need the separate Raw/export route described below. Different source datasets
belong in different destination roots. Additional recordings from the same
dataset may be appended sequentially when existing shared metadata agrees.

MNE-BIDS reads the identified recording and selects its matching sidecars.
The converter retains:

- Original event TSV/JSON, including numeric codes, behavioral columns and
  their descriptions. Exported annotation descriptions alone cannot represent
  all this information.
- The matched channel table and electrode/coordinate files, including original
  units and coordinate frame. Dimensionless native Status channels retain their
  units when a channel table must be generated.
- The selected participant row, all its columns and the participant dictionary.
  Existing destination participant rows and scan rows survive later calls.
- Descriptive acquisition fields: manufacturer/model, reference, ground,
  placement scheme, task description/name, institution, filters and mains
  frequency. Dataset authors, licence and citations are retained. The source
  dataset DOI moves to `SourceDatasets`; the subset receives no copy of that DOI
  as its own identifier. Existing `SourceDatasets` entries are retained when the
  immediate source is added, without repeating an identical entry. Sources
  without a DOI use a local source-folder URI.

The converter checks event bounds and supplied sample positions, sampling rate
and channel correspondence before writing. These checks require events within
the sampled interval; negative onsets and events extending outside that interval
need separate review. Unknown event duration `n/a` is retained. Contradictory
sampling rates or event sample positions fail rather than changing the signal.
Published duration and EEG-count discrepancies issue warnings, and generated
metadata describes the decoded signal. EEG counts reflect imported channel
types; MNE reader defaults may label auxiliary inputs as EEG when source channel
roles are unavailable. Review those roles before analysis. Duration uses
`n_samples / sfreq`, which
includes the final sample interval. Events named `discontinuity` are retained
with a warning; they do not establish gap lengths or a valid analysis interval.

EEG JSON fields are resolved from applicable ancestor files, from dataset root
to recording directory. Lower-level keys override higher-level values; absent
keys remain inherited. Multiple applicable EEG JSON files at the same directory
level are rejected rather than resolved by filename specificity. Effective
known mains frequency is validated before reading and retained on read-back.
Original metadata files remain unchanged. Event/channel/coordinate sidecars
still use MNE-BIDS matching; their JSON dictionaries are not merged across levels.
Unlisted acquisition fields and
other files, such as stimuli and arbitrary ancillary documents, are outside this
route. Acquisition facts are attributed to the source release; they are not
independently verified or automatically reconciled with conflicting evidence.

Writing and metadata assembly finish in a temporary dataset before publication.
Existing recordings and conflicting shared metadata are protected. Temporary
space must accommodate one native recording. Publishing files is not a
filesystem transaction: inspect an interrupted move before retrying, and avoid
concurrent appends. Full BIDS validation remains a separate check.

EDF/BDF source layout is checked before reading sidecars: incomplete records,
wrong declared counts and discontinuous EDF+D/BDF+D are rejected. These guards
do not authenticate calibration or every vendor-specific layout.

Use `mne_bids.read_raw_bids(written)` downstream to read the recording with its
sidecars. Read participant columns and behavioral event tables directly when
they are needed: MNE Raw cannot expose every original field. Classification
intervals, exclusions, player selection and train/test splits remain downstream
decisions.

## Native filenames and Raw

A plain filename reads BrainVision `.vhdr`, EDF `.edf`, BioSemi `.bdf` or
continuous EEGLAB `.set` through the corresponding MNE reader. Required companion
files must be available. This route imports native metadata and annotations;
it does not inspect surrounding BIDS sidecars. Use `BIDSPath` explicitly for
existing BIDS input.

An MNE Raw source writes the metadata already represented in that object.
Preloaded recordings require `allow_preload=True` and an explicit export format,
such as `format="BrainVision"`, with its optional exporter installed. MNE-BIDS
annotation conventions still apply, including code reassignment and sample
rounding. Arbitrary native-format layouts, calibration and discontinuities are
not independently validated by this converter.

The BIDS-aware route adds no runtime dependencies. Generated-format unit tests
use the existing optional `pybv` exporter and skip when it is unavailable.

Dimensionless native STIM channels also retain their units in the plain filename
and Raw routes, using `arbitrary` in the generated channel table.

# EEG conversion and trigger units

`convert_eeg_to_bids` retains dimensionless native STIM channels, such as BioSemi
Status, using `arbitrary` in the generated BIDS channel table. This spelling
preserves the native MNE unit code when reading the output with MNE-BIDS.
Samples, trigger codes and voltage-channel units remain unchanged. No new
dependency or trigger extraction policy is introduced.
