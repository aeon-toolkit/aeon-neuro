"""Keep inspected BIDS source facts aligned with in-memory recordings."""

from copy import deepcopy
from math import isfinite

import numpy as np

from aeon_neuro.datasets._bids_input import (
    _event_sample,
    inspect_bids_eeg,
    open_bids_eeg_recording,
)
from aeon_neuro.datasets._classification import (
    _integer,
    _recording_info,
    _recordings,
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

    @property
    def events_frame(self):
        """Detached pandas view of source events and exact sample coordinates."""
        import pandas as pd

        return pd.DataFrame(self.events)

    @property
    def channels_frame(self):
        """Detached pandas view of source channels, or None when absent."""
        import pandas as pd

        return None if self.channels is None else pd.DataFrame(self.channels)


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
    Source tables have portable row and optional pandas views. A case-level view
    additionally records current layouts and original contributing support.
    ``process`` executes supported operations and updates that view with history;
    ``with_history`` only records a caller's account. Neither changes original
    acquisition facts or detects arbitrary external array edits.
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
                sample = row["bids_onset_sample"]
                if sample is not None:
                    sample = _integer(sample, "bids_onset_sample")
                expected = _event_sample(
                    float(row["onset"]),
                    by_id[recording_id]["sampling_frequency"],
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
        from aeon_neuro.datasets._info_view import _validate_view

        self._view = (
            _validate_view(self._report["view"], by_id)
            if "view" in self._report
            else None
        )

    def __len__(self):
        return (
            len(self._view["rows"]) if self._view is not None else len(self._recordings)
        )

    def __getitem__(self, selector):
        """Select detached information in the same order as X[selector]."""
        indices = _selection(selector, len(self))
        if self._view is None:
            # Reuse paired selection without creating full dummy recordings.
            report = self.to_dict()
            report["recordings"] = [report["recordings"][i] for i in indices]
            report["source_ids"] = [row["recording_id"] for row in report["recordings"]]
            ids = set(report["source_ids"])
            report["events"] = [
                row for i in indices for row in self._recordings[i].events
            ]
            report["dataset"]["selected_participants"] = sorted(
                {row["participant_id"] for row in report["recordings"]}
            )
            report["issues"] = [
                issue
                for issue in report["issues"]
                if issue["recording_id"] is None or issue["recording_id"] in ids
            ]
        else:
            report = self.to_dict()
            # to_dict already detached the whole view. Copying it again for
            # every selected case would make ordinary CV selection quadratic.
            view = report["view"]
            view["rows"] = [view["rows"][i] for i in indices]
            view["layouts"] = [view["layouts"][i] for i in indices]
        return EEGDatasetInfo(report)

    @classmethod
    def from_arrays(
        cls,
        X,
        *,
        sfreq,
        channel_names,
        units=None,
        subjects=None,
        case_ids=None,
        groups=None,
    ):
        """Describe case arrays using explicit layout and optional known identities.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Three-dimensional collection or list of channel/sample arrays.
        sfreq : float
            Known positive sampling frequency in Hz.
        channel_names : sequence of str
            Unique current channel names in array order.
        units : list of str or None, default=None
            Current per-channel units; None retains unknown units.
        subjects : sequence or None, default=None
            Known subject IDs per row; None retains unknown identities.
        case_ids : sequence of str or None, default=None
            Unique row identities, generated within this collection if omitted.
        groups : sequence or None, default=None
            Explicit per-row groups. If omitted, uses the supplied subjects.

        Returns
        -------
        EEGDatasetInfo
            Detached case information with no invented source acquisition facts.

        Notes
        -----
        This route supplies no BIDS acquisition facts. Unknown units/subjects stay
        None, and generated case IDs identify this collection only.
        """
        from aeon_neuro.datasets._info_view import _from_arrays

        report = {
            "version": 2,
            "ready": True,
            "recordings": [],
            "source_ids": [],
            "events": [],
            "issues": [],
            "dataset": {
                "selected_participants": [],
                "description": None,
                "readme": None,
                "participants": None,
                "participants_json": None,
            },
            "view": _from_arrays(
                X, sfreq, channel_names, units, subjects, case_ids, groups
            ),
        }
        return cls(report)

    @classmethod
    def from_classification(cls, dataset):
        """Describe constructed cases while retaining their original source report.

        Parameters
        ----------
        dataset : ClassificationDataset
            Constructed cases with shared source/history provenance, usually
            from passing this information object to the ordinary case builder.

        Returns
        -------
        EEGDatasetInfo
            Detached case identities, current layout, source context and history.
        """
        source = dataset.metadata.get("provenance", {}).get("source")
        if source is None:
            raise ValueError(
                "Case metadata requires shared source provenance; pass "
                "EEGDatasetInfo as metadata to build_classification_dataset"
            )
        info = cls(source)
        report = info.to_dict()
        from aeon_neuro.datasets._info_view import _case_view

        report["view"] = _case_view(info, dataset)
        report["history"] = deepcopy(dataset.metadata["provenance"]["history"])
        report["case_provenance"] = deepcopy(dataset.metadata["provenance"])
        return cls(report)

    @property
    def cases(self):
        """Detached case-level identities and original contributing sample support."""
        if self._view is None:
            raise ValueError("This object describes recordings, not constructed cases")
        return deepcopy(self._view["rows"])

    @property
    def case_ids(self):
        """Case identities in exactly the current signal order."""
        return tuple(row["case_id"] for row in self.cases)

    @property
    def groups(self):
        """Aligned declared case groups, or source participants for recordings."""
        return np.asarray(
            (
                [row["group_id"] for row in self.cases]
                if self._view is not None
                else self.participant_ids
            ),
            dtype=object,
        )

    def subject_of(self, index):
        """Return a known subject identity for one aligned row, or None.

        Parameters
        ----------
        index : int
            Nonnegative index within the current view.

        Returns
        -------
        str, int or None
            Original known subject identity, or None when not supplied.
        """
        index = _integer(index, "subject index")
        if not 0 <= index < len(self):
            raise ValueError("Subject index is outside the information view")
        return self.participant_ids[index]

    @property
    def layouts(self):
        """Detached current layouts, separate from original source descriptors."""
        if self._view is None:
            from aeon_neuro.datasets._info_view import _recording_view

            return _recording_view(self)["layouts"]
        return deepcopy(self._view["layouts"])

    @property
    def sfreq(self):
        """Current sampling frequency when all aligned rows agree."""
        from aeon_neuro.datasets._info_view import _uniform

        return _uniform(self.layouts, "sampling_frequency")

    @property
    def channel_names(self):
        """Current ordered channels when all aligned rows agree."""
        from aeon_neuro.datasets._info_view import _uniform

        return _uniform(self.layouts, "channel_names")

    @property
    def units(self):
        """Current per-channel units when all aligned rows agree."""
        from aeon_neuro.datasets._info_view import _uniform

        return _uniform(self.layouts, "units")

    @property
    def case_table(self):
        """Detached pandas view of case identities and source support."""
        import pandas as pd

        return pd.DataFrame(self.cases)

    @property
    def participants_frame(self):
        """Detached pandas view of the complete source participant population."""
        import pandas as pd

        table = self.participants
        return (
            None
            if table is None
            else pd.DataFrame(table["rows"], columns=table["columns"])
        )

    def events_for_cases(
        self, label_column="trial_type", *, group_column="participant_id"
    ):
        """Return event rows for make_event_cases using literal source labels/groups.

        Parameters
        ----------
        label_column : str, default="trial_type"
            Source event column holding literal labels.
        group_column : str, default="participant_id"
            Source recording field holding explicit groups.

        Returns
        -------
        list of dict
            Independent event rows with identities, onsets, labels and groups.

        Notes
        -----
        Callers can edit returned rows for explicit class mappings or grouping
        by event fields. No files are opened by this convenience.
        """
        from aeon_neuro.datasets._info_view import _events_for_cases, _recording_inputs

        metadata, events = _recording_inputs(self)
        return _events_for_cases(events, metadata, label_column, group_column)

    @property
    def recordings(self):
        """Original source recording views retained by this information context.

        Before case construction these align with loaded recordings. Case-level
        views retain the source context separately from current case row order.
        """
        return self._recordings

    @property
    def source_ids(self):
        """Tuple of str: Selected source identities in signal order."""
        if self._view is not None:
            return tuple(
                dict.fromkeys(
                    row["recording_id"] or "array:" + row["case_id"]
                    for row in self._view["rows"]
                )
            )
        return tuple(self._report["source_ids"])

    @property
    def participant_ids(self):
        """Source participant identities, one per aligned recording or case."""
        if self._view is not None:
            return tuple(row["participant_id"] for row in self._view["rows"])
        return tuple(entry["participant_id"] for entry in self._report["recordings"])

    @property
    def current_metadata(self):
        """Detached recording descriptors matching current arrays and event times.

        Use with ``events_for_cases`` and the plain recording/case APIs after
        supported processing or selection. Only channel names, sampling rate and
        sample count describe the current layout; other fields retain source
        context. Original descriptors remain in ``metadata``. Already segmented
        cases raise because they are not recording inputs for case construction.
        """
        from aeon_neuro.datasets._info_view import _current_metadata

        return _current_metadata(self)

    @property
    def metadata(self):
        """List of dict: Detached original acquisition descriptors."""
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

    def process(self, X, operation, **parameters):
        """Execute a supported operation and return detached signals and information.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Finite real arrays aligned with this information view.
        operation : str
            One of channels, scale, crop, resample, filter or reference.
        **parameters : dict
            Explicit operation parameters, documented in the EEG information guide.

        Returns
        -------
        X_processed : np.ndarray or list of np.ndarray
            Detached processed arrays in the original row order.
        info_processed : EEGDatasetInfo
            Updated current layouts, original support and executed history.

        Notes
        -----
        Operations are ``channels`` (names), ``scale`` (factor, units), ``crop``
        (tmin/tmax, optional start_time), ``resample`` (sfreq), ``filter``
        (l_freq/h_freq, optional method), and ``reference`` (channels).
        Original source metadata and coordinates are preserved. Current layouts,
        conservative contributing support and an executed history record change
        together. Operations run independently per aligned row, without fitting.
        """
        from aeon_neuro.datasets._info_processing import _process

        return _process(X, self, operation, parameters)

    def to_ts_header(self):
        """Format portable source, current-layout and history metadata as .ts comments.

        Returns
        -------
        str
            Portable information formatted as comment text for an aeon .ts writer.

        Notes
        -----
        These comment lines precede the ordinary .ts structural header. They do
        not replace its problem/class declarations and contain no signal values.
        """
        import json

        return (
            "# aeon-neuro-info "
            + json.dumps(
                self.to_dict(),
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )

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
        if self._view is not None:
            from aeon_neuro.datasets._info_view import _check_layout

            _check_layout(X, self._view)
            return
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
        if self._view is not None:
            return selected, self[indices]
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


def _source_provenance(info, implementation):
    """Keep source/history independent of higher-level problem definitions."""
    from aeon_neuro import __version__

    source = info.to_dict()
    history = source.pop("history", [])
    return {
        "version": 1,
        "source": source,
        "history": history,
        "construction": {"implementation": implementation, "version": __version__},
    }


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
