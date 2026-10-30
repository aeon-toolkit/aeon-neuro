.. _utils_ref:

Utility functions
=================

BIDS conversion
---------------

``convert_eeg_to_bids`` accepts BrainVision, EDF/EDF+, BioSemi BDF/BDF+ and
continuous EEGLAB filenames, an existing MNE Raw recording, or an explicit
source ``BIDSPath``, and returns the written ``BIDSPath``. For filenames and Raw,
supplied metadata is assumed
correct; MNE-BIDS handles writing and its existing validation. No analysis
processing or metadata repair is added. Native-path reading does not import
existing BIDS sidecars.

Dimensionless native STIM channels, including BioSemi Status, use ``arbitrary``
units in the generated channel table so MNE-BIDS read-back retains their native
unit code. Samples and trigger codes are not rescaled.

A source ``BIDSPath`` reads the recording with MNE-BIDS and preserves matched
event/channel tables, coordinate files and selected participant fields outside
Raw. Identifiers and native encoding stay unchanged. Output uses a separate
dataset root, protects existing recordings and warns on published duration or
EEG-count discrepancies. This route does not merge all JSON inheritance levels
or infer acquisition facts. See ``docs/ingestion/README.md`` for the preservation
contract, limitations and a COVID/RPS example.

.. autofunction:: aeon_neuro.utils.bids.convert_eeg_to_bids

EEG ingestion
-------------

``load_eeg_to_mne`` selects the native MNE reader for ``.vhdr``, ``.edf``,
``.bdf`` or continuous ``.set`` files. It preserves the reader's complete Raw
representation and accepts format-specific reader options. MNE may upsample
mixed-rate EDF/BDF channels. The reader rejects incomplete native records and
discontinuous EDF+D/BDF+D before decoding, while accepting unknown record counts
when all records are complete. Calibration and arbitrary corruption remain
outside these layout checks.
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

Keep companion files together.


.. autofunction:: aeon_neuro.utils.data_io.load_eeg_to_mne
Metadata preparation
---------------------

``prepare_eeg_metadata`` reports unknown/invalid mains frequency and applies
explicit channel-role/frequency corrections on a Raw copy. Corrections require
a supplied evidence reference and produce a JSON-compatible before/after report.
Samples and annotations remain unchanged; declared unit changes are recorded,
without sample rescaling or calibration repair. The report assesses mains
frequency only and does not establish complete BIDS metadata validity.

.. autofunction:: aeon_neuro.utils.metadata.prepare_eeg_metadata

EEG conversion guide
--------------------

.. toctree::
   :maxdepth: 1

   ../ingestion/README

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
Explicit events with nonzero ``first_samp`` and unknown annotation time origin
are rejected because the underlying writer can shift their timing. Dated offsets
retain MNE's sample convention; use a complete zero-based recording when timing
is unknown rather than assigning an arbitrary measurement date.
