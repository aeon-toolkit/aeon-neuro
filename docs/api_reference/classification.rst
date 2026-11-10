.. _classification_ref:

Classification
==============

``aeon-neuro`` classifiers for EEG data in the collection format
(n_cases, n_channels, n_timepoints).


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


Distance based
--------------

.. currentmodule:: aeon_neuro.classification.distance_based


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RiemannianKNNClassifier
    RiemannianMDMClassifier
    TemplateCCAClassifier
    TRCAClassifier


Feature based
-------------

.. currentmodule:: aeon_neuro.classification.feature_based


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CSPClassifier
    DifferentialEntropyClassifier
    XDAWNClassifier
