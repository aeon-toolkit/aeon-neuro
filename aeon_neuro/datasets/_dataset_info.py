"""Keep inspected BIDS source facts aligned with in-memory recordings."""

from copy import deepcopy
from math import isfinite

import numpy as np

from aeon_neuro.datasets._bids_input import inspect_bids_eeg, open_bids_eeg_recording
from aeon_neuro.datasets._classification import (
    _integer,
    _recording_info,
    _recordings,
    _samples,
)


class RecordingInfo:
    """Read-only views of one recording's inspected source metadata.

    Parameters
    ----------
    metadata : dict
        Source descriptor returned by ``inspect_bids_eeg``.
    events : list of dict
        Inspected event rows for this recording, in their source order.

    Notes
    -----
    Usually obtained from ``EEGDatasetInfo.recordings``. Returned dictionaries
    and lists are detached copies. These source facts do not track arbitrary
    changes to external arrays or certify the correctness of source annotations.
    """

    def __init__(self, metadata, events):
        self._metadata = deepcopy(metadata)
        self._events = deepcopy(events)

    @property
    def recording_id(self):
        """Original root-relative recording identity."""
        return self._metadata["recording_id"]

    @property
    def metadata(self):
        """Detached complete descriptor for the existing case APIs."""
        return deepcopy(self._metadata)

    @property
    def entities(self):
        """Published BIDS identifiers, without their entity prefixes."""
        return {
            name: self._metadata.get(key)
            for name, key in (
                ("subject", "subject"),
                ("session", "session"),
                ("task", "task"),
                ("acquisition", "acq"),
                ("run", "run"),
            )
        }

    @property
    def sfreq(self):
        """Decoded source sampling frequency in Hz."""
        return self._metadata["sampling_frequency"]

    @property
    def channel_names(self):
        """Tuple of str: Decoded source channel names in signal-array order."""
        return tuple(self._metadata["channel_names"])

    @property
    def channels(self):
        """List of dict or None: Original channel rows, absent when unavailable."""
        if self._metadata["sidecar_paths"]["channels"] is None:
            return None
        return deepcopy(self._metadata["channel_metadata"])

    @property
    def eeg_json(self):
        """Resolved EEG JSON source fields, without changing their meaning."""
        return deepcopy(self._metadata["eeg_metadata"])

    @property
    def events(self):
        """List of dict: Original event rows and normalized sample coordinates."""
        return deepcopy(self._events)

    @property
    def electrodes(self):
        """Dict or None: Original electrode table as columns and rows."""
        return deepcopy(self._metadata["electrodes"])

    @property
    def coordsystem(self):
        """Dict or None: Source coordinate system and units, without conversion."""
        return deepcopy(self._metadata["coordsystem"])

    @property
    def scan(self):
        """Dict or None: Matched scans row and its original zero-based row index."""
        return deepcopy(self._metadata["scan"])

    def bad_channels(self):
        """Return source-declared bad channels; raise if status is unavailable.

        Returns
        -------
        tuple of str
            Channel names with published status ``bad``, in source order.
        """
        rows = self.channels
        if rows is None or any(
            row.get("status") not in ("good", "bad") for row in rows
        ):
            raise ValueError("Channel status is unavailable or unrecognized")
        return tuple(row["name"] for row in rows if row["status"] == "bad")

    def channels_of_type(self, channel_type):
        """Return channels with the requested source-declared type.

        Parameters
        ----------
        channel_type : str
            BIDS channel type such as ``EEG``; matching ignores case.

        Returns
        -------
        tuple of str
            Matching names in source order. Missing types raise an error.
        """
        if not isinstance(channel_type, str) or not channel_type.strip():
            raise ValueError("Supply a nonempty channel type")
        rows = self.channels
        if rows is None or any(
            not row.get("type") or row["type"] == "n/a" for row in rows
        ):
            raise ValueError("Channel type metadata is unavailable")
        return tuple(
            row["name"] for row in rows if row["type"].upper() == channel_type.upper()
        )


class EEGDatasetInfo:
    """Source metadata aligned with a collection of loaded BIDS recordings.

    Parameters
    ----------
    report : dict
        Ready version-2 report from ``inspect_bids_eeg`` or ``to_dict``. Retains
        the complete source participant table separately from the selected
        recording scope. An optional ``history`` list contains version-1
        caller-reported processing records, validated on construction.

    Notes
    -----
    Construction copies metadata, validates source/event alignment and performs
    no signal reads. Use ``load_bids_dataset`` for verified source loading.
    Table views are portable dictionaries/rows rather than pandas objects.
    Processing records document caller-supplied claims, not observed array
    operations. They do not update source facts or validate derived layouts.
    Automatic target/group inference is not provided.
    """

    def __init__(self, report):
        if (
            not isinstance(report, dict)
            or type(report.get("version")) is not int
            or report["version"] != 2
        ):
            raise ValueError("Supply a version-2 inspection report")
        if report.get("ready") is not True:
            messages = "; ".join(
                issue["message"]
                for issue in report.get("issues", [])
                if issue.get("severity") == "error"
            )
            raise ValueError(
                f"The inspection report must be ready before loading: {messages}"
            )
        self._report = deepcopy(report)
        _validate_history(self._report.get("history", []))
        try:
            entries = self._report["recordings"]
            if not isinstance(entries, list):
                raise ValueError("Recording descriptors must be a list")
            entries = _recording_info(entries) if entries else []
            ids = [entry["recording_id"] for entry in entries]
            if self._report["source_ids"] != ids:
                raise ValueError("Source IDs and recording descriptors are not aligned")
            participants = sorted({entry["participant_id"] for entry in entries})
            if self._report["dataset"]["selected_participants"] != participants:
                raise ValueError("Selected participants do not match recording scope")
            grouped = {entry["recording_id"]: [] for entry in entries}
            by_id = {entry["recording_id"]: entry for entry in entries}
            seen = set()
            for row in self._report["events"]:
                recording_id = row["recording_id"]
                if recording_id not in grouped:
                    raise ValueError("Event refers to an unselected recording")
                sample = _integer(row["bids_onset_sample"], "bids_onset_sample")
                if not 0 <= sample < by_id[recording_id]["n_samples"]:
                    raise ValueError("Event sample is outside its recording")
                expected = _samples(
                    float(row["onset"]),
                    by_id[recording_id]["sampling_frequency"],
                    "event onset",
                )
                if sample != expected:
                    raise ValueError("Event sample disagrees with its source onset")
                identity = (recording_id, row["bids_event_id"])
                if identity in seen:
                    raise ValueError("Duplicate event identity in inspection report")
                seen.add(identity)
                grouped[recording_id].append(row)
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete inspection report metadata") from exc
        self._report["recordings"] = entries
        self._recordings = tuple(
            RecordingInfo(entry, grouped[entry["recording_id"]]) for entry in entries
        )

    def __len__(self):
        return len(self._recordings)

    @property
    def recordings(self):
        """Tuple of RecordingInfo: Recording views in exactly the signal order."""
        return self._recordings

    @property
    def source_ids(self):
        """Tuple of str: Selected source identities in signal order."""
        return tuple(self._report["source_ids"])

    @property
    def participant_ids(self):
        """Tuple of str: Source participant identities, one per recording."""
        return tuple(entry["participant_id"] for entry in self._report["recordings"])

    @property
    def metadata(self):
        """List of dict: Detached descriptors for existing problem/case APIs."""
        return deepcopy(self._report["recordings"])

    @property
    def events(self):
        """List of dict: Detached events for the selected recordings."""
        return deepcopy(self._report["events"])

    @property
    def dataset(self):
        """Dataset source facts and the current selected participant scope."""
        return deepcopy(self._report["dataset"])

    @property
    def description(self):
        """Dict or None: Original dataset description."""
        return deepcopy(self._report["dataset"]["description"])

    @property
    def readme(self):
        """Str or None: Original dataset README text."""
        return self._report["dataset"]["readme"]

    @property
    def participants(self):
        """Dict or None: Complete source participant table as columns and rows."""
        return deepcopy(self._report["dataset"]["participants"])

    @property
    def participants_json(self):
        """Dict or None: Original participant-column dictionary."""
        return deepcopy(self._report["dataset"]["participants_json"])

    @property
    def history(self):
        """Detached processing records in append order, empty when unreported."""
        return deepcopy(self._report.get("history", []))

    def with_history(
        self, step, parameters, *, implementation, version, fit_scope="unspecified"
    ):
        """Return a detached metadata view with one reported processing step.

        Parameters
        ----------
        step : str
            Nonempty operation name, such as ``scale`` or ``filter``.
        parameters : dict
            Effective parameters using only portable JSON values: string-keyed
            dictionaries, lists, strings, finite floats, integers, bools or None.
        implementation : str
            Nonempty qualified function, package or implementation identifier.
        version : str
            Nonempty version of that implementation, saved independently of the
            record schema version. Supply an actual version, not an estimate.
        fit_scope : str, default="unspecified"
            Description of fitting inputs, or ``none`` for an unfitted operation.
            The default explicitly leaves this unreported. No scope is inferred.

        Returns
        -------
        EEGDatasetInfo
            New view with the record appended for the currently selected source
            IDs. The parent, its history and all source facts remain unchanged.

        Notes
        -----
        This method records a caller's account; it neither performs processing
        nor checks that arrays were processed as described. A crop/resampling
        note does not make source layouts or events describe a derived array.
        Selection retains earlier records and their original application scopes,
        which may include sources no longer selected. Empty views cannot append
        a step because there are no current sources to identify.
        """
        if not len(self):
            raise ValueError("Cannot append processing history to an empty view")
        record = {
            "version": 1,
            "step": step,
            "parameters": parameters,
            "implementation": implementation,
            "implementation_version": version,
            "fit_scope": fit_scope,
            "source_ids": list(self.source_ids),
        }
        # Validate before copying so recursive/nonportable caller objects fail
        # with the same error as malformed saved records.
        _validate_history([record])
        report = self.to_dict()
        report["history"] = self.history + [record]
        return EEGDatasetInfo(report)

    def to_dict(self):
        """Return detached source metadata and optional reported history.

        Returns
        -------
        dict
            Version-2 source report including original snapshots. A view with
            reported steps also includes the optional ``history`` extension.
            Pass this dictionary to ``EEGDatasetInfo`` to restore both.
        """
        return deepcopy(self._report)

    def validate_layout(self, X):
        """Check the collection count, shapes and numeric types against metadata.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Recording collection in this metadata's order.

        Notes
        -----
        Layout checks cannot detect an external permutation or value mutation
        when array shapes remain the same. Keep the returned X/info pair together.
        """
        if not isinstance(X, (list, tuple, np.ndarray)):
            raise ValueError("Supply a collection of recordings")
        if isinstance(X, np.ndarray) and (
            X.ndim != 3 or X.dtype.kind not in "fiu" or 0 in X.shape[1:]
        ):
            raise ValueError(
                "Dense recordings must be real numeric arrays with three axes"
            )
        if len(X) != len(self):
            raise ValueError(
                "X and metadata must contain the same number of recordings"
            )
        if len(self):
            _recordings(X, self._report["recordings"])

    def select(self, X, selector):
        """Copy recordings and matching metadata together in the requested order.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Loaded recordings in this metadata's order.
        selector : slice or one-dimensional sequence of int or bool
            Unique nonnegative recording indices, or a mask matching the count.
            Empty selections are allowed. Negative/duplicate indices are rejected.

        Returns
        -------
        X_selected : np.ndarray or list of np.ndarray
            Detached signals, retaining the recording axis. Dense arrays remain
            arrays; list/tuple inputs produce a list of separate arrays.
        info_selected : EEGDatasetInfo
            Matching source descriptors, events, scope and relevant diagnostics.
        """
        self.validate_layout(X)
        indices = _selection(selector, len(self))
        # Detach both sides: later array or metadata edits must not mutate a
        # parent selection that may already be used by another experiment.
        selected = (
            np.take(X, indices, axis=0)
            if isinstance(X, np.ndarray)
            else [X[i].copy() for i in indices]
        )
        report = self.to_dict()
        # Retain complete earlier processing scopes: narrowing their source IDs
        # could hide that an operation was fitted on a larger population.
        report["recordings"] = [report["recordings"][i] for i in indices]
        report["source_ids"] = [entry["recording_id"] for entry in report["recordings"]]
        ids = set(report["source_ids"])
        report["events"] = [row for i in indices for row in self._recordings[i].events]
        report["issues"] = [
            issue
            for issue in report["issues"]
            if issue["recording_id"] is None or issue["recording_id"] in ids
        ]
        report["dataset"]["selected_participants"] = sorted(
            {entry["participant_id"] for entry in report["recordings"]}
        )
        return selected, EEGDatasetInfo(report)


def _validate_history(history):
    if not isinstance(history, list):
        raise ValueError("Processing history must be a list of version-1 records")
    fields = {
        "version",
        "step",
        "parameters",
        "implementation",
        "implementation_version",
        "fit_scope",
        "source_ids",
    }
    for record in history:
        if (
            not isinstance(record, dict)
            or set(record) != fields
            or type(record["version"]) is not int
            or record["version"] != 1
        ):
            raise ValueError("Invalid version-1 processing history record")
        for field in ("step", "implementation", "implementation_version", "fit_scope"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Processing history {field} must be nonempty text")
        ids = record["source_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "Processing history source_ids must be unique source names"
            )
        try:
            if type(record["parameters"]) is not dict:
                raise ValueError
            _json_history_value(record["parameters"])
        except (ValueError, RecursionError) as exc:
            raise ValueError(
                "Processing history parameters must be portable JSON"
            ) from exc


def _json_history_value(value):
    # Reject tuple conversion, non-string keys and NumPy objects rather than
    # silently changing a parameter's type during JSON serialization.
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_history_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_history_value(item)
        return
    raise ValueError("Nonportable processing parameter")


def _selection(selector, count):
    if isinstance(selector, slice):
        return np.asarray(list(range(count))[selector], dtype=int)
    if isinstance(selector, (list, tuple)):
        booleans = [isinstance(value, (bool, np.bool_)) for value in selector]
        if any(booleans) and not all(booleans):
            raise ValueError("Do not mix boolean masks and integer indices")
    indices = np.asarray(selector)
    if indices.ndim != 1:
        raise ValueError("Selection must be a one-dimensional index sequence or mask")
    if indices.dtype.kind == "b":
        if len(indices) != count:
            raise ValueError("Selection mask must match the recording count")
        return np.flatnonzero(indices)
    if indices.size == 0:
        return np.asarray([], dtype=int)
    if (
        indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("Recording indices must be unique integers within bounds")
    return indices.astype(int)


def load_bids_dataset(
    root, *, subjects=None, sessions=None, tasks=None, acqs=None, runs=None
):
    """Load supported BIDS recordings with their inspected source metadata.

    Parameters
    ----------
    root : path-like
        Local BIDS root containing EDF/BDF sources. No acquisition occurs.
    subjects, sessions, tasks, acqs, runs : str, list of str or None, default=None
        Exact BIDS selections passed to ``inspect_bids_eeg``.

    Returns
    -------
    X : np.ndarray or list of np.ndarray
        Complete recordings in source order. Equal shapes, channel names/order
        and sampling rates produce a 3D array; otherwise a list of 2D arrays.
    info : EEGDatasetInfo
        Aligned metadata, source facts and events, without inferring a target.

    Notes
    -----
    This explicit operation reads complete signals into memory and closes each
    owned source handle on success or failure. It reuses inspected snapshots and
    the source opener. It does not modify the legacy ``load_bids_data`` loader,
    expand companion-file support or apply preprocessing/montage corrections.
    """
    info = EEGDatasetInfo(
        inspect_bids_eeg(
            root,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            acqs=acqs,
            runs=runs,
        )
    )
    recordings = []
    for descriptor in info.metadata:
        with open_bids_eeg_recording(root, descriptor) as raw:
            recordings.append(raw.get_data())
    info.validate_layout(recordings)
    reference = info.recordings[0]
    compatible = all(
        values.shape == recordings[0].shape
        and recording.channel_names == reference.channel_names
        and recording.sfreq == reference.sfreq
        for values, recording in zip(recordings, info.recordings, strict=True)
    )
    return (np.stack(recordings) if compatible else recordings), info
