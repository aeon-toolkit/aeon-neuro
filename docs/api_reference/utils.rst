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

Automatic EEG loading
---------------------

``load_eeg_to_mne`` loads a recording into an MNE Raw object and selects the
reader from the file extension and, for ambiguous formats, the file header.
It does not add filtering, rereferencing, channel selection or epoching.

.. code-block:: python

    from aeon_neuro.utils.data_io import load_eeg_to_mne

    raw = load_eeg_to_mne("recording.edf")
    raw = load_eeg_to_mne("recording.vhdr", preload=True)
    raw = load_eeg_to_mne("recording.xdf")

Supported formats
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Format
     - Input and reader selection
   * - Classic BrainVision
     - ``.vhdr``/``.ahdr``; ``.eeg``/``.vmrk``/``.amrk`` when a nearby
       header explicitly references the input.
   * - EDF/EDF+, BDF, GDF
     - ``.edf``, ``.bdf``, ``.gdf`` via MNE.
   * - Continuous EEGLAB
     - ``.set`` with its linked ``.fdt`` file when present.
   * - Neuroscan / ANT Neuro
     - ``.cnt``; header signatures distinguish the vendor.
   * - EGI
     - Continuous ``.egi``/``.raw`` binary, or a complete ``.mff`` directory.
   * - Nihon Kohden
     - ``.eeg`` with a recognized manufacturer header.
   * - Nicolet, Persyst, eXimia
     - ``.data`` (plus ``.head``), ``.lay`` (plus ``.dat``), ``.nxe``.
   * - FIF
     - ``.fif`` or ``.fif.gz`` via MNE.
   * - Recorder 2 / BVRF
     - ``.bvrh``/``.bvrd``/``.bvrm``/``.bvri`` via optional PyBVRF.
   * - XDF
     - ``.xdf``/``.xdfz`` via optional PyXDF; one EEG stream plus markers.

Other formats accepted by the installed ``mne.io.read_raw`` dispatcher are
delegated to MNE. Keep companion files together.


.. autofunction:: aeon_neuro.utils.data_io.load_eeg_to_mne
