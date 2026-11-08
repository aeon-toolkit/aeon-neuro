.. _transformations_ref:

Transformations
===============

Series transformers
-------------------

Transformers applied to a single series of shape ``(n_channels, n_timepoints)``.

.. currentmodule:: aeon_neuro.transformations.series


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BandPassSeriesTransformer
    BandPowerSeriesTransformer
    EpochSeriesTransformer
    NuttallStrand

Collection transformers
-----------------------

Transformers applied to a collection of series of shape ``(n_cases, n_channels, n_timepoints)``.

.. currentmodule:: aeon_neuro.transformations.collection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DownsampleCollectionTransformer
    LogVariance

Channel selection
-----------------

.. currentmodule:: aeon_neuro.transformations.collection.channel_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BPSO
    CLeVerCluster
    ChannelFilter
    CLeVerHybrid
    CLeVerRank
    DetachRocketChannelSelector
    Riemannian

Collection
----------

.. currentmodule:: aeon_neuro.transformations.collection


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DownsampleCollectionTransformer
    ERPCovariances
    LogVariance
    TimeCrop

Channel creation
----------------

.. currentmodule:: aeon_neuro.transformations.collection.channel_creation


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CommonSpatialPatterns
    UMAPChannelCreator
    XDAWN
