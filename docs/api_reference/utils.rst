.. _utils_ref:

Utility functions
=================

BIDS conversion
---------------

``convert_eeg_to_bids`` accepts BrainVision, EDF/EDF+, BioSemi BDF/BDF+ and
continuous EEGLAB filenames, or an existing MNE Raw recording, and returns the
written ``BIDSPath``. Supplied metadata is assumed
correct; MNE-BIDS handles writing and its existing validation. No analysis
processing or metadata repair is added. Native-path reading does not import
existing BIDS sidecars.

Dimensionless native STIM channels, including BioSemi Status, use ``arbitrary``
units in the generated channel table so MNE-BIDS read-back retains their native
unit code. Samples and trigger codes are not rescaled.

.. autofunction:: aeon_neuro.utils.bids.convert_eeg_to_bids

EEG ingestion
-------------

``load_eeg_to_mne`` selects the native MNE reader for ``.vhdr``, ``.edf``,
``.bdf`` or continuous ``.set`` files. It preserves the reader's complete Raw
representation and accepts format-specific reader options. MNE may upsample
mixed-rate EDF/BDF channels; discontinuities are not independently validated.
Some embedded EEGLAB recordings require explicit preloaded export when writing
BIDS. For existing BIDS datasets, use ``mne_bids.read_raw_bids`` to include their
sidecars.

.. autofunction:: aeon_neuro.utils.data_io.load_eeg_to_mne

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

EEG conversion guide
--------------------

.. toctree::
   :maxdepth: 1

   ../ingestion/README
