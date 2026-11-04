"""Utilities for loading datasets."""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "load_kdd_example",
    "load_kdd_full_example",
    "load_eeg_classification",
    "fetch_openneuro_dataset",
    "BIDSPreparationError",
    "ClassificationDataset",
    "build_classification_dataset",
    "make_recording_cases",
    "make_fixed_length_cases",
    "make_event_cases",
    "validate_group_split",
    "validate_classification_split",
    "load_classification_problem",
    "inspect_bids_eeg",
    "open_bids_eeg_recording",
    "EEGDatasetInfo",
    "RecordingInfo",
    "load_bids_dataset",
]

from aeon_neuro.datasets._bids_input import inspect_bids_eeg, open_bids_eeg_recording
from aeon_neuro.datasets._bids_prepare import BIDSPreparationError
from aeon_neuro.datasets._classification import (
    ClassificationDataset,
    build_classification_dataset,
    make_event_cases,
    make_fixed_length_cases,
    make_recording_cases,
    validate_group_split,
)
from aeon_neuro.datasets._classification_split import validate_classification_split
from aeon_neuro.datasets._data_loaders import load_eeg_classification
from aeon_neuro.datasets._dataset_info import (
    EEGDatasetInfo,
    RecordingInfo,
    load_bids_dataset,
)
from aeon_neuro.datasets._openneuro import fetch_openneuro_dataset
from aeon_neuro.datasets._problem import load_classification_problem
from aeon_neuro.datasets._single_problem_loaders import (
    load_kdd_example,
    load_kdd_full_example,
)
