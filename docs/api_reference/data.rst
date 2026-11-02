.. _data_ref:

Data
====

Usage
-----

.. toctree::
   :maxdepth: 1

   ../classification-problems

Classification case construction
--------------------------------

.. autoclass:: aeon_neuro.datasets.ClassificationDataset

.. autofunction:: aeon_neuro.datasets.make_recording_cases

.. autofunction:: aeon_neuro.datasets.make_fixed_length_cases

.. autofunction:: aeon_neuro.datasets.make_event_cases

.. autofunction:: aeon_neuro.datasets.build_classification_dataset

.. autofunction:: aeon_neuro.datasets.validate_group_split

.. autofunction:: aeon_neuro.datasets.validate_classification_split

EEG data loading, preprocessing and labeling
--------------------------------------------

.. autofunction:: aeon_neuro.datasets.load_bids_data

.. autofunction:: aeon_neuro.datasets.label_bids_recordings

.. autofunction:: aeon_neuro.datasets.preprocess_eeg
