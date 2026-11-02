# EEG input and existing BIDS recordings

`convert_eeg_to_bids` accepts native EEG filenames, MNE Raw objects and existing
BIDS recordings identified by `mne_bids.BIDSPath`. It returns the written
`BIDSPath`. Conversion retains all channels and performs no filtering,
resampling, re-referencing, epoching or classification-label selection.

## Continuous XDF input

Install the optional readers with `pip install 'aeon-neuro[eeg_io]'`. XDF and
compressed XDFZ input use PyXDF, selecting exactly one EEG stream and any desired
marker streams. Select streams explicitly when a file contains multiple EEG
recordings or unrelated markers:

```python
from aeon_neuro.utils.data_io import load_eeg_to_mne

raw = load_eeg_to_mne(
    "recording.xdf",
    select_streams=[{"type": "EEG"}, {"type": "Markers"}],
    timing_tolerance=0.25,
)
```

The supported EEG stream has at least two finite samples, unique channel labels,
consistent declared channel counts and voltage channel types. EEG, EOG, ECG,
EMG, SEEG, ECoG, DBS and BIO types are accepted; dimensionless STIM channels and
other measurements need a separate explicit adapter. Known V/mV/uV/µV/μV/nV
units are converted to MNE's volts. No channel identities or unknown units are
inferred, and descriptor-count mismatches fail before calibration can broadcast
one signal into several channels.

Original timestamps are inspected before PyXDF dejittering, then the corrected
timestamps are checked again. Both must fit a regular grid anchored at their
first and last samples. `timing_tolerance` is the maximum deviation in sample
periods, default 0.25, with `0 <= timing_tolerance < 0.5`; float representation
error is accounted for separately. The effective rate must agree with the checked
timestamps and be within 1% of the declared nominal rate, allowing bounded clock
drift in either direction. This rate check is separate from `timing_tolerance`;
a declared 100 Hz stream with regular 200 Hz timestamps is rejected even though
its timestamps form a regular grid. Original adjacent timestamps more than 1.5
nominal periods apart, nonmonotonic clocks, detected multiple segments and
insufficient timestamp precision fail. Clock resets and discontinuous recordings
need another explicit representation. Relaxing PyXDF's break thresholds or
disabling dejittering does not bypass these checks. No missing samples are filled
or signals interpolated.

PyXDF's clock correction, dejittering and stream-selection options still apply.
An explicit `on_chunk` callback runs once per chunk; checks apply to the callback's
output before dejittering. A failed callback is surfaced rather than accepted as
partial data. Original EEG timestamp vectors are retained and concatenated for
these checks, adding temporary timestamp storage without copying the signal
array for timestamp validation. XDF
remains fully preloaded even with `preload=False`; memory-map filenames are not
supported. Other signal streams and original per-sample timestamps are not
retained in the returned Raw. Validation covers the decoded streams, not every
possible file corruption or the correctness of acquisition/clock metadata.

Markers use corrected timestamps relative to the selected EEG's first timestamp.
Their sample counts must align and their times must be finite and fall between
the first and last EEG samples. Out-of-range markers fail before MNE can discard
them. Explicit selection can omit an unwanted marker stream; markers are not
silently filtered by their meaning. Recording wall-clock dates, references and
electrode positions are not invented.

### Exporting preloaded XDF to BIDS

Choose an output encoding and permit export of the preloaded recording:

```python
from mne_bids import BIDSPath
from aeon_neuro.utils.bids import convert_eeg_to_bids

destination = BIDSPath(root="output-bids", subject="01", task="experiment")
written = convert_eeg_to_bids(
    raw, destination, format="EDF", allow_preload=True
)
```

EDF export needs the optional `edfio` exporter (`pip install edfio`). Review
available acquisition metadata and provide event-code mappings when the
experiment defines them. EDF introduces amplitude quantisation; BrainVision is
another explicit export choice when its exporter is installed. MNE-BIDS maps
annotation times to samples, so fractional marker latencies may round. This is a
derived encoding, not a byte-preserving copy of XDF or a structural BIDS validator.
The source file remains unchanged. See the
[PyXDF importer](https://github.com/xdf-modules/pyxdf) for backend options.

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

BrainVision companion inputs (`.eeg`, `.vmrk`, `.amrk`) are resolved only through
a sibling `.vhdr`/`.ahdr` that references them. Discovery honours the header's
declared `Codepage`, including ANSI, and the legacy Latin-1 decoding fallback.
It accepts MNE's NeurOne `[Common infos]` section spelling as well as
`[Common Infos]`. This keeps companion loading consistent with direct header
loading for supported encodings and non-ASCII filenames. Ambiguous references
still require an explicit header; the native decoder remains MNE and its format
limitations still apply. A malformed unrelated sibling does not determine the
encoding of a valid header.

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
