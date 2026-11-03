.. _data_ref:

Data
====

OpenNeuro fetching and preparation
---------------------------------

.. autofunction:: aeon_neuro.datasets.fetch_openneuro_dataset

.. autoexception:: aeon_neuro.datasets.BIDSPreparationError

Usage
-----

.. toctree::
   :maxdepth: 1

   ../classification-problems
   ../problem-format
   ../problem-loading

Classification case construction
--------------------------------

.. autofunction:: aeon_neuro.datasets.load_classification_problem

.. autoclass:: aeon_neuro.datasets.ClassificationDataset

.. autofunction:: aeon_neuro.datasets.make_recording_cases

.. autofunction:: aeon_neuro.datasets.make_fixed_length_cases

.. autofunction:: aeon_neuro.datasets.make_event_cases

.. autofunction:: aeon_neuro.datasets.build_classification_dataset

.. autofunction:: aeon_neuro.datasets.validate_group_split

.. autofunction:: aeon_neuro.datasets.validate_classification_split
