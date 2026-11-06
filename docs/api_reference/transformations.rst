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

Collection
----------

.. currentmodule:: aeon_neuro.transformations.collection


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DownsampleCollectionTransformer
    ERPCovariances
    LogVariance

Channel creation
----------------

.. currentmodule:: aeon_neuro.transformations.collection.channel_creation


.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CommonSpatialPatterns
    UMAPChannelCreator
    XDAWN
