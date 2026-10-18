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

Acquisition events
------------------

``convert_eeg_to_bids`` forwards explicit ``events`` and ``event_id`` to MNE-BIDS
alongside native annotations. This allows acquisition triggers, such as events
extracted from a BDF Status channel in an MNE Raw, to reach the BIDS event table.
Supply acquisition-appropriate extraction settings and descriptions for every
code and annotation; no scientific class mapping is inferred. MNE-BIDS aligns
event-table onsets to samples, so fractional annotation latencies may round by
up to half a sample. Status-channel samples remain unchanged.

Explicit event arrays/files are checked for integer sample bounds and mapping
coverage before calling the writer, preventing omitted events and partial
metadata for these invalid inputs. Other write failures retain MNE-BIDS behaviour.
Explicit events with nonzero ``first_samp`` are rejected because the underlying
writer can shift their timing; use a complete zero-based recording.
