# EEG Datasets

## OpenNeuro BIDS recordings

Install the optional download client with `pip install 'aeon-neuro[openneuro]'`.
`fetch_openneuro_dataset` accepts an OpenNeuro dataset ID plus an explicit release,
or a versioned OpenNeuro HTTPS URL. It prepares primary EEG metadata and returns
a local BIDS folder that can be read directly. Select subjects explicitly:

```python
from aeon_neuro.datasets import fetch_openneuro_dataset
from mne_bids import BIDSPath, read_raw_bids

root = fetch_openneuro_dataset(
    "ds008082", version="1.0.0", subjects="CUCOV001", cache_dir="./openneuro-cache"
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
Signals are downloaded once per cache entry and metadata is prepared in place
before the entry is published. No second conversion copy is created. There is
no filtering, resampling, temporal cropping, re-referencing or channel removal.
Classification examples and player identities remain downstream decisions.

### Preparation and findings

Request the per-recording report alongside the BIDS folder:

```python
root, report = fetch_openneuro_dataset(
    "ds008082", version="1.0.0", subjects="all", return_report=True
)
print(report["summary"])
```

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
    metadata_corrections=corrections, return_report=True,
)
```

The example assumes that its recording and evidence exist. Supported metadata
fields are `PowerLineFrequency`, `EEGReference`, `EEGGround`,
`EEGPlacementScheme`, `Manufacturer`, `ManufacturersModelName` and
`TaskDescription`. Channel corrections require an existing channel table with
verified units; supported types are `EEG`, `EOG`, `ECG`, `EMG`, `MISC` and `TRIG`.
Source strings record evidence but do not authenticate it. Sampling frequency
cannot be changed through descriptive corrections. Use `prepare=False` to fetch
the published originals without preparation; corrections then are disallowed.

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
    cache_dir="./openneuro-cache", download=False,
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
Benchmark classification experiments, results generated with code like this

```python
        dataset = "EyesOpenShut"
        results_dir = "C:/temp/" # where to results
        data_dir = "C:/temp/" # Location of data in ts format data_dir/<dataset>/
        resample = 0        # 0 indicates default train test
        cls_name = "RocketClassifier" # name used for results name
        from aeon.classification.kernel_based import RocketClassifier
        from aeon.benchmarking.experiments import \
            load_and_run_classification_experiment
        classifier = RocketClassifier()
        load_and_run_classification_experiment(
            overwrite=False,
            problem_path=data_dir,
            results_path=results_dir,
            cls_name=cls_name,
            classifier=classifier,
            dataset=dataset,
            resample_id=resample,
            build_train=False,
            predefined_resample=False,
        )
```

This generates a result file called testResample0.csv. Some exploratory results:

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
