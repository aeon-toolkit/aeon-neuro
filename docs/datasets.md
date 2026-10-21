# EEG Datasets


## EEG Data Characteristics
| Problem               | train cases | test cases | dimensions | length  | num classes  |
|---                    |---    |---    |---    |---    |---|
| Blink                 | 500   | 450   | 4   | 510   |  2 |
| COVID19               | 25319 | 11216 | 19  | 400   |  2 |
| EyesOpenShut          | 56    | 42    | 14  |  128  | 2  |
| FaceDetection         | 5890  | 3524  | 144 |  62   |  2 |
| FingerMovements       | 316   | 100   | 28  |  50   |  2 |
| HandMovementDirection | 160   | 74    | 10  |  400  | 4  |
| MindReading           | 727   | 653   | 204 | 200   |  2 |
| MotorImagery          | 278   | 100   | 64  | 3000  | 2  |
| SelfRegulationSCP1    | 268   | 293   | 6   | 869   |  2 |
| SelfRegulationSCP2    | 200   | 180   | 7   | 1152  |  2 |

## COVID19

COVID19 distinguishes COVID-survivor (`covid`) from close-contact
(`close_contact`) cohort membership using eyes-closed resting EEG from
[OpenNeuro ds008082 v1.0.0](https://openneuro.org/datasets/ds008082/versions/1.0.0)
(source commit `c6f603a6e9c7117715dce56ba48836c4515e7dd3`, CC0).
Cite the [source paper](https://doi.org/10.1016/j.cnp.2026.02.007) and source dataset.
The metadata's `Control` group denotes close contacts, not an independently
recruited healthy control cohort. This is an exploratory classification task.

### Availability and loading

**Publication is pending.** COVID19 can currently be loaded from a local archive.
The catalogue uses the existing unavailable-record sentinel `49`; it is not a
published COVID19 Zenodo record ID. Replace it with the actual published record ID
before enabling remote downloads or marking this contribution ready for release.

Place `COVID19_TRAIN.ts` and `COVID19_TEST.ts` together in
`<archive_root>/COVID19/`, then load a partition:

```python
from aeon_neuro.datasets import load_eeg_classification

X_train, y_train = load_eeg_classification(
    "COVID19", split="TRAIN", extract_path="/path/to/archive_root"
)
X_test, y_test = load_eeg_classification(
    "COVID19", split="TEST", extract_path="/path/to/archive_root"
)
```

Each row is one window in `(example, channel, time)` order. Values are in volts;
the prepared arrays are float32, while the text loader returns floating-point
arrays at its default precision. Keep the companion group arrays and
`examples_TRAIN.tsv` / `examples_TEST.tsv`: `.ts` files do not encode participants.
Text loading uses more memory than the corresponding NumPy memory maps.

### Preparation and provenance

All 173 source recordings were screened (87 `Covid`, 86 `Control`).
`sub-CUCOV173` has no eye-state events and contributes no eligible windows, leaving
172 participants. Labels come from `participants.tsv: group`, with `Covid → covid`
and `Control → close_contact`. No label is inferred from EEG features.

The recipe uses nonoverlapping two-second windows on the [10, 480) second grid,
stopping earlier when the initial baseline enters eyes-open alternation,
stimulation or recovery. The latest non-discontinuity event at each window's
start must explicitly indicate `65 / eyes_closed`. A code-2 discontinuity strictly
inside a window excludes that window; markers exactly on its boundaries are not
crossed. Code 2 preserves the remembered eye state. Calling or other non-rest
events suspend eligibility until explicit rest resumes. State persistence across
discontinuities is a documented recipe assumption.

The 19 channels, in order, are `Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6
FZ CZ PZ`. EDF calibration yields volts at the original 200 Hz. Each window is
common-average referenced and temporally centered per channel. No additional
bandpass, notch, resampling, interpolation or amplitude rejection is applied;
no preprocessing filter crosses an event boundary.

The 70/30 split uses sorted eligible participant IDs, cohort stratification and
`random_state=42`. All windows from a participant stay together:

| Partition | Participants | Windows | close_contact | covid | Shape |
|---|---:|---:|---:|---:|---|
| TRAIN | 120 | 25,319 | 12,892 | 12,427 | 25,319 × 19 × 400 |
| TEST | 52 | 11,216 | 5,801 | 5,415 | 11,216 × 19 × 400 |

There are 40,655 candidate windows, of which 36,535 are retained. Exclusion reasons
are: 3,512 not eyes closed; 235 without documented eyes-closed state; 221 outside
the initial baseline; 99 outside the recording; 32 crossing non-rest events; and
21 crossing discontinuities. Every candidate decision is recorded in `candidates.tsv`.

The release recipe contains `02_prepare_covid.ipynb`, `preparation_io.py`,
`preparation_checks.py`, pinned dependencies, participant and example manifests,
source file hashes, preparation metadata, export and validation scripts, and
checksums. The `.ts` validation report checks every exported value against the
float32 arrays, every label and row mapping, and group separation using aeon's
parser in bounded chunks. Source reconstruction and event-boundary checks are
reported separately. The source EEG is obtained from OpenNeuro rather than
redistributed in the recipe.

### Evaluation

Use participant-grouped folds within TRAIN (the recipe uses seed 42), and fit all
learned preprocessing inside those folds. Report both window-level and
participant-level metrics; average window probabilities per participant with
class order preserved. Unequal window counts must not turn a person into multiple
independent participants. The cohort has been explored previously; changing the
seed does not establish independent external validation. No expert artifact
selection or clinical diagnostic validation is claimed.

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
