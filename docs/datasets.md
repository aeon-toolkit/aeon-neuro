# EEG Datasets

## NEMAR BIDS recordings

Install `pip install 'aeon-neuro[nemar]'` for acquisition through
[nemar-py](https://github.com/nemarOrg/nemar-py). Select an exact NEMAR dataset
and release, then either explicit subjects with an optional task, or exact
primary recording filenames:

```python
from aeon_neuro.datasets import fetch_nemar_dataset

root, report = fetch_nemar_dataset(
    "nm000132", version="v1.1.1", subjects="021", task="P3",
    cache_dir="./nemar-cache", return_report=True,
)
print(root, report["selected_recordings"])
```

Subject and task labels are case-sensitive strings; `"021"` differs from `"21"`.
The optional `sub-` and `task-` prefixes are removed. Subject selections retain
every matching supported raw EEG recording across sessions and runs. Each
requested subject must have a matching recording. For a narrower selection:

```python
root = fetch_nemar_dataset(
    "nm000132", version="v1.1.1",
    recordings="sub-021/eeg/sub-021_task-P3_eeg.set",
    cache_dir="./nemar-cache",
)
```

The release is mandatory; `latest` is rejected. Releases with or without the
leading `v` identify the same cache. Supported primary formats are BIDS EEGLAB
`.set`, BrainVision `.vhdr` and EDF/BDF. Manifest-advertised same-stem EEGLAB
`.fdt` files and header-declared relative BrainVision companions are retained.
Applicable ancestor JSON/TSV metadata, participant tables, geometry and root
documentation are kept without modification. Geometry may declare coordinate
spaces absent from signal filenames. Participant tables retain unselected rows;
this is a selected local copy. Other modalities, derivatives, stimuli and
phenotype folders are outside this EEG selection.

The returned folder can be passed to `mne_bids.read_raw_bids`. For EEGLAB
`.set` recordings, include `.set` in the `formatts` argument when using
`aeon_neuro.datasets.load_bids.load_bids_data`; its default formats are
`(".vhdr", ".bdf", ".edf")`. Acquisition does not decode recordings,
validate complete BIDS compliance, correct metadata,
infer labels, construct epochs or choose preparation/evaluation. Catalogue
search and download previews belong in applications. EEGLAB files with unusual
external companion naming may need a separately acquired complete source copy.

Every downloaded file requires a declared size and SHA-256, MD5 or Git blob
SHA-1 checksum. nemar-py verifies transfers against the source manifest. The
report retains the manifest URL, requested and realized selection, downloader
version, source checksums and local size/SHA-256 for every file. MD5 and Git
hashes reflect the source's published evidence, not stronger authentication.
No independent authenticity claim follows from these checksums.

Completed entries live under `$AEON_DATA/nemar` or `~/aeon_data/nemar`, unless
`cache_dir` is supplied. Each release and canonical request has a separate
entry; overlapping requests can duplicate files. Failed transfers remain
unpublished and their temporary files are removed. Each reuse checks the whole
inventory, sizes and SHA-256 hashes without contacting NEMAR or importing the
optional client:

```python
root = fetch_nemar_dataset(
    "nm000132", version="v1.1.1", subjects="021", task="P3",
    cache_dir="./nemar-cache", download=False,
)
```

A missing offline selection raises `FileNotFoundError`. Changed or incomplete
entries raise `ValueError` and are not silently redownloaded or repaired. Use a
fresh cache directory or remove that entry explicitly before retrying. Treat the
returned folder as read-only and prepare/convert in another destination.

## OpenNeuro BIDS recordings

Install the optional download client with `pip install 'aeon-neuro[openneuro]'`.
`fetch_openneuro_dataset` accepts an OpenNeuro dataset ID plus an explicit release,
or a versioned OpenNeuro HTTPS URL. It returns a local BIDS folder, retaining
published files without preparation by default. Select subjects explicitly and
opt in to metadata preparation when needed for reading with MNE-BIDS:

```python
from aeon_neuro.datasets import fetch_openneuro_dataset
from mne_bids import BIDSPath, read_raw_bids

root = fetch_openneuro_dataset(
    "ds008082", version="1.0.0", subjects="CUCOV001",
    cache_dir="./openneuro-cache", prepare=True,
)
source = BIDSPath(
    root=root, subject="CUCOV001", task="COVID", datatype="eeg", suffix="eeg",
    extension=".edf",
)
raw = read_raw_bids(source)
```

To fetch a complete pinned snapshot without constructing a subject list:

```python
root = fetch_openneuro_dataset("ds008082", version="1.0.0", subjects="all")
```

`subjects="all"` includes all subjects and other nonhidden snapshot files,
including `phenotype`, `stimuli` and `derivatives` when present. These downloads
can be much larger than a subject selection. Whole snapshots and explicit
subject selections have separate cache entries.

The same helper accepts
`"https://openneuro.org/datasets/ds006761/versions/1.0.0"` with `subjects="31"`
for an RPS recording. Subject labels may include `sub-`; pass labels as strings
to retain leading zeros. A subject selection downloads **all files** in those
subject folders, including other modalities or sessions when present. RPS signal
files can be several gigabytes. Dataset-root files, including inherited JSON
sidecars and the original participant table, are retained. This is a selected
local copy; it does not prune participant rows.
Dataset-level folders such as `phenotype`, `stimuli` and `derivatives` are outside
an explicit subject selection; they are included with `subjects="all"`.

The helper returns a local BIDS folder. Use MNE-BIDS to read a recording;
`convert_eeg_to_bids` is not required for these existing BIDS recordings.
Signals are downloaded once per cache entry. With `prepare=True`, metadata is
prepared in place before the entry is published. No second conversion copy is
created. There is no filtering, resampling, temporal cropping, re-referencing or
channel removal.
Classification examples and player identities remain downstream decisions.

### Preparation and findings

Set `prepare=True` to check and correct metadata, and request the per-recording
report alongside the BIDS folder:

```python
root, report = fetch_openneuro_dataset(
    "ds008082", version="1.0.0", subjects="all", prepare=True, return_report=True
)
print(report["summary"])
```

Without preparation, `return_report=True` reports `prepared=False`, an empty
recordings list and the dataset identity, without a preparation summary.

Preparation reads supported primary EEG headers through MNE-BIDS without
preloading signals. EDF/BDF layout checks reject incomplete records, incorrect
declared counts and discontinuous EDF+D/BDF+D before decoding. It checks sampling
frequency, supplied channel order and event timing/sample correspondence.
Every supplied `sample` and `onset_sample` column is checked: values must be
integers, agree with each other and match onset within half a sample.
Recording duration is corrected from the
decoded sample grid. Channel counts are corrected only when a channel table
supplies roles; missing roles are reported and reader defaults are not treated
as verified acquisition facts. Event tables, signal files, participant fields
and other downloaded material are retained. Corrections to inherited metadata
are written at recording scope so other recordings do not change.

EEG JSON fields are inherited from applicable ancestor files, with lower-level
keys overriding higher-level values and absent keys remaining inherited. Multiple
applicable EEG JSON files at the same directory level are rejected. Effective
inherited fields are materialized in recording-specific JSON before MNE-BIDS
reading, retaining the original ancestor files. Reports include their paths,
per-field origins for original metadata and materialization actions. Event,
channel and coordinate sidecars continue to use MNE-BIDS matching; their JSON
dictionaries are not merged across hierarchy levels.

Discontinuity markers and events outside the sampled interval are reported and
retained without clipping or inventing gaps. Unknown mains frequency remains
unknown. Reader warnings and missing metadata appear in the report, and a summary
warning signals findings that need review. This is an EEG preparation check,
not full BIDS validation or a scientific assessment of usable analysis intervals.
Non-EEG recordings and derivatives are preserved without preparation.

Blocking frequency/channel inconsistencies, invalid event timing and unreadable
recordings raise `BIDSPreparationError`. Its `.report` contains per-recording
findings, and no completed entry is published. The report retains original EEG
metadata, each before/after correction and its evidence, plus size/SHA-256
provenance for downloaded files. Corrected dataset descriptions retain source
DOI attribution in `SourceDatasets` rather than claiming the original DOI.
When a release includes `datacite.yml`, that citation file is retained and
listed in `.bidsignore`, since it is outside the BIDS file schema.

Explicit corrections require an evidence reference and an exact relative
recording filename:

```python
corrections = {
    "sub-01/eeg/sub-01_task-rest_eeg.edf": {
        "source": "Verified acquisition record, revision 2",
        "metadata": {"PowerLineFrequency": 50, "EEGReference": "common"},
        "channel_types": {"ECG": "ECG"},
    }
}
root, report = fetch_openneuro_dataset(
    "ds000001", version="1.0.0", subjects="01",
    prepare=True, metadata_corrections=corrections, return_report=True,
)
```

The example assumes that its recording and evidence exist. Supported metadata
fields are `PowerLineFrequency`, `EEGReference`, `EEGGround`,
`EEGPlacementScheme`, `Manufacturer`, `ManufacturersModelName` and
`TaskDescription`. Channel corrections require an existing channel table with
verified units; supported types are `EEG`, `EOG`, `ECG`, `EMG`, `MISC` and `TRIG`.
Source strings record evidence but do not authenticate it. Sampling frequency
cannot be changed through descriptive corrections. The default `prepare=False`
fetches published originals without preparation; corrections require
`prepare=True`.

The default cache is `$AEON_DATA/openneuro`, or `~/aeon_data/openneuro` when
`AEON_DATA` is unset. Each dataset, release and canonical subject selection has
its own entry. Preparation policy and sourced corrections also identify the
entry, so changed decisions cannot silently reuse earlier output. Overlapping
selections or different preparation policies can duplicate recordings. Policy
upgrades also create a new entry so earlier prepared data cannot bypass new
validation or inheritance checks. A completed entry
is reused without importing the optional client or contacting the network:

```python
root = fetch_openneuro_dataset(
    "ds008082", version="1.0.0", subjects="CUCOV001",
    cache_dir="./openneuro-cache", prepare=True, download=False,
)
```

Each reuse reads all cached files and checks their sizes and SHA-256 hashes,
including large signals, and verifies the stored preparation report. This detects local changes after download, rather than
providing independently authenticated remote checksums. Transfers use
[openneuro-py's verification and retry behavior](https://github.com/openneuro-py/openneuro-py).
The source is explicitly OpenNeuro, irrespective of the client's mirror setting.
Failed downloads never become completed cache entries; temporary files are
removed, so a subsequent call starts a new transfer. A missing cache with
`download=False` raises `FileNotFoundError`. Incomplete or altered completed
entries raise `ValueError`; use a fresh cache directory or remove the identified
entry before retrying. Treat returned folders as read-only, and write conversions
or edits elsewhere.
For constructed recording/window/event cases, see
[validating classification assignments](classification-problems.md). It describes
explicit independence policies, session roles, original-source overlap and inner
validation scope using the shared case container.

## Selecting recordings from a local BIDS folder

`aeon_neuro.datasets.load_bids.load_bids_data` matches subject, session, task,
acquisition and run identifiers exactly, without prefixes or automatic zero
padding. Every requested value must match at least one recording after all
filters are applied. For example, `subjects=["1,", "2"]` raises for `"1,"` even
when subject `"2"` exists. The error names all unmatched values and lists
available dataset identifiers before reading any signals. Every combination
of requested values need not exist; `None` means no filter for that entity.


## EEG Data Characteristics
| Problem               | train cases | test cases | dimensions | length  | num classes  |
|---                    |---    |---    |---    |---    |---|
| Blink                 | 500   | 450   | 4   | 510   |  2 |
| EyesOpenShut          | 56    | 42    | 14  |  128  | 2  |
| FaceDetection         | 5890  | 3524  | 144 |  62   |  2 |
| FingerMovements       | 316   | 100   | 28  |  50   |  2 |
| HandMovementDirection | 160   | 74    | 10  |  400  | 4  |
| MindReading           | 727   | 653   | 204 | 200   |  2 |
| MotorImagery          | 278   | 100   | 64  | 3000  | 2  |
| SelfRegulationSCP1    | 268   | 293   | 6   | 869   |  2 |
| SelfRegulationSCP2    | 200   | 180   | 7   | 1152  |  2 |

## Classification
Benchmark classification experiments, for example with aeon directly:

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_from_ts_file

# Load a dataset in .ts format (train and test splits live side by side).
X_train, y_train = load_from_ts_file("./EyesOpenShut/EyesOpenShut_TRAIN.ts")
X_test, y_test = load_from_ts_file("./EyesOpenShut/EyesOpenShut_TEST.ts")

classifier = RocketClassifier()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
```

For full benchmarking experiments (resampling, results files, critical
difference diagrams), see
[tsml-eval](https://github.com/time-series-machine-learning/tsml-eval).
Some exploratory results:

### Best classifier accuracies (default train split,  default classifier settings)

| Problem             |	Majority Class	   | Best Acc  | Best Classifier  |
|---                  |---              |---        |---                 |
| Blink		          |   |  1.000000	| Arsenal/RocketClassifier       |
| EyesOpenShut		  |    |   0.523810	| DrCIF/MUSE/RocketClassifier    |
| FaceDetection		  |   |   0.678490	| CNNClassifier              |
| FingerMovements		| | 0.560000	    | ShapeletTransformClassifier   |
| HandMovementDirection |   | 0.581081	| CNNClassifier |
| MindReading		    | | 0.595712	    | FreshPRINCE |
| MotorImagery		    |  |   0.590000	| HIVECOTEV2 |
| SelfRegulationSCP1	|	| 0.897611	| FreshPRINCE |
| SelfRegulationSCP2	|	| 0.538889	| KNeighborsTimeSeriesClassifier/TDE |

### Best aeon classifier accuracies (30 resamples, default classifier settings)

| Problem             |	Majority Class	   | Best Acc  | Best Classifier  |
|---                  |---         |---        |---                 |
| Blink		          |            | 0.9999	   | TemporalDictionaryEnsemble |
| EyesOpenShut		  |  | 0.6643	 | KNeighborsTimeSeriesClassifier |
| FaceDetection		  |  | 0.7187 | 	CNNClassifier |
| FingerMovements	  |  | 0.5813 | 	Mini-ROCKET |
| HandMovementDirection|  | 0.5775 | 	CNNClassifier |
| MindReading		  |  | 0.7369 | 	Mini-ROCKET |
| MotorImagery		  |  | 0.5423 | 	TemporalDictionaryEnsemble |
| SelfRegulationSCP1  |  | 0.9110 | 	Multi-ROCKET |
| SelfRegulationSCP2  |  |	0.5531 | 	MUSE |

### Results discussion
[Blink](https://github.com/Kelvin9811/EEG-Blink-dataset)

[EyesOpenShut](https://archive.ics.uci.edu/ml/machine-learning-databases/00264/)

[FaceDetection (MEG)](https://www.kaggle.com/c/decoding-the-human-brain/data)
The [leaderboard](https://www.kaggle.
com/competitions/decoding-the-human-brain/leaderboard) shows a best accuracy of 0.
75501, although it is not clear if the results are directly comparable to those
above. This needs clarification.


[FingerMovements](http://www.bbci.de/competition/ii/berlin_desc.html)

[HandMovementDirection](http://bbci.de/competition/iv/)

[MindReading (MEG)](https://www.researchgate.net/publication/239918465_ICANNPASCAL2_Challenge_MEG_Mind_Reading_--_Overview_and_Results)

[MotorImagery](http://bbci.de/competition/iii/desc_I.html)

[SelfRegulationSCP1](http://bbci.de/competition/ii/tuebingen_desc_i.html)

[SelfRegulationSCP2](http://bbci.de/competition/ii/tuebingen_desc_i.html)
