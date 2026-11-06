"""Exclusions for the aeon estimator checks run on aeon-neuro estimators.

These are estimators which currently fail the checks. Entries should be removed as
the estimators are fixed.
"""

__all__ = ["EXCLUDE_ESTIMATORS", "EXCLUDED_TESTS"]

# Estimators which cannot be created for testing at all
EXCLUDE_ESTIMATORS = ["EpochSeriesTransformer"]

# Checks to skip for an estimator, keyed by estimator name
EXCLUDED_TESTS = {
    "BandPassSeriesTransformer": ["check_no_multithreading_param"],
    "BandPowerSeriesTransformer": ["check_no_multithreading_param"],
    "CBraModClassifier": [
        "check_non_state_changing_method",
        "check_fit_updates_state_and_cloning",
        "check_persistence_via_pickle",
        "check_classifier_output",
    ],
    "DeepConvNetClassifier": ["check_non_state_changing_method"],
    "EEGNetClassifier": [
        "check_non_state_changing_method",
        "check_fit_updates_state_and_cloning",
        "check_persistence_via_pickle",
        "check_classifier_output",
    ],
    "Riemannian": ["check_no_multithreading_param"],
    "RiemannianKNNClassifier": [
        "check_no_multithreading_param",
        "check_classifier_output",
    ],
    "RiemannianMDMClassifier": ["check_no_multithreading_param"],
    "UMAPChannelCreator": ["check_persistence_via_pickle", "check_fit_deterministic"],
}
