.. _classification_ref:

Classification
==============

``aeon-neuro`` classifiers for EEG data in the collection format
(n_cases, n_channels, n_timepoints).


Shared EEG information
----------------------

.. autoclass:: aeon_neuro.classification.BaseEEGClassifier
   :members: fit, predict, predict_proba
   :no-inherited-members:

See :doc:`../eeg-information` for optional side information, current layout
validation.

Deep learning
-------------

.. currentmodule:: aeon_neuro.classification.deep_learning


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BraindecodeClassifier
    CBraModClassifier
    DeepConvNetClassifier
    EEGConformerClassifier
    EEGNetClassifier


Feature based
-------------

.. currentmodule:: aeon_neuro.classification.feature_based


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CSPClassifier
    XDAWNClassifier
