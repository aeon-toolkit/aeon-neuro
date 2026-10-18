.. _utils_ref:

Utility functions
=================

BIDS conversion
---------------

``convert_eeg_to_bids`` accepts a native BrainVision filename or an existing MNE
Raw recording and returns the written ``BIDSPath``. Supplied metadata is assumed
correct; MNE-BIDS handles writing and its existing validation. No analysis
processing or metadata repair is added. Native-path reading does not import
existing BIDS sidecars.

.. autofunction:: aeon_neuro.utils.bids.convert_eeg_to_bids

BrainVision ingestion
---------------------

Use ``load_brainvision_to_mne`` to retain channel metadata, sampling information
and annotations. ``load_brainvision_to_numpy`` retains the existing array-only
interface and optional EEG channel selection.

.. autofunction:: aeon_neuro.utils.data_io.load_brainvision_to_mne

.. autofunction:: aeon_neuro.utils.data_io.load_brainvision_to_numpy

Auxiliary information
---------------------

``load_auxiliary_info`` returns ``None`` with a warning when its JSON file cannot
be read or decoded. The directory prefix must include its trailing separator.

.. autofunction:: aeon_neuro.utils.data_io.load_auxiliary_info

Metadata preparation
---------------------

``prepare_eeg_metadata`` reports unknown/invalid mains frequency and applies
explicit channel-role/frequency corrections on a Raw copy. Corrections require
a supplied evidence reference and produce a JSON-compatible before/after report.
Samples and annotations remain unchanged; declared unit changes are recorded,
without sample rescaling or calibration repair. The report assesses mains
frequency only and does not establish complete BIDS metadata validity.

.. autofunction:: aeon_neuro.utils.metadata.prepare_eeg_metadata
