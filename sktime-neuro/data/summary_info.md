# EEG Problems


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
        from sktime.classification.kernel_based import RocketClassifier
        from sktime.benchmarking.experiments import \
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

### Best sktime classifier accuracies (30 resamples, default classifier settings)

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


### All results, (default train split,  default classifier settings)

### All results, (averaged over 30 resamples, default classifier settings)