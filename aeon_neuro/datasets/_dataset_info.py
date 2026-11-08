"""Keep supplied source facts aligned with current EEG arrays."""

from copy import deepcopy
from numbers import Real

import numpy as np

from aeon_neuro.datasets._classification import (
    _channels,
    _identifier,
    _integer,
    _recordings,
    _required,
)
from aeon_neuro.datasets._info_view import _event_sample


class RecordingInfo:
    """Read-only views of one recording's inspected source metadata.

    Parameters
    ----------
    metadata : dict
        Supplied source descriptor, including its identity and layout.
    events : list of dict
        Supplied event rows for this recording, in their source order.

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
        # An inspector can retain an empty normalized list for an absent sidecar.
        # Only an explicit absent channel path records that absence; unrelated
        # sidecar keys say nothing about the availability of supplied rows.
        if (
            "channels" in self._metadata.get("sidecar_paths", {})
            and self._metadata["sidecar_paths"]["channels"] is None
        ) or self._metadata.get("channel_metadata") is None:
            return None
        return deepcopy(self._metadata["channel_metadata"])

    @property
    def eeg_json(self):
        """Resolved EEG JSON source fields, without changing their meaning."""
        return deepcopy(self._metadata.get("eeg_metadata", {}))

    @property
    def events(self):
        """List of dict: Original event rows and normalized sample coordinates."""
        return deepcopy(self._events)

    @property
    def electrodes(self):
        """Dict or None: Original electrode table as columns and rows."""
        return deepcopy(self._metadata.get("electrodes"))

    @property
    def coordsystem(self):
        """Dict or None: Source coordinate system and units, without conversion."""
        return deepcopy(self._metadata.get("coordsystem"))

    @property
    def scan(self):
        """Dict or None: Matched scans row and its original zero-based row index."""
        return deepcopy(self._metadata.get("scan"))

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
    """Detached source facts and current layouts aligned with EEG arrays.

    Parameters
    ----------
    report : dict or None, default=None
        Version-2 state from ``to_dict``. Its optional version-1 view describes
        current rows separately from original source records.
    dataset : dict or None, default=None
        Explicit description, README, participants and participant dictionary.
        Use only when report is omitted. Missing optional facts stay None.
    recordings : list of dict or None, default=None
        Explicit source descriptors with recording_id, participant_id,
        channel_names, sampling_frequency and n_samples. Other source facts
        may be supplied without opening any files.
    events : list of dict or None, default=None
        Source events with recording_id, bids_event_id, onset and exact
        bids_onset_sample, or None for a known off-grid onset.

    Notes
    -----
    Construction validates supplied metadata and reads no files. It does not
    authenticate source contents. ``from_arrays`` describes current case arrays
    without inventing acquisition facts. Selection returns detached values;
    equal-shaped external array permutations cannot be detected by layout checks.
    """

    def __init__(self, report=None, *, dataset=None, recordings=None, events=None):
        if report is not None and any(
            value is not None for value in (dataset, recordings, events)
        ):
            raise ValueError("Supply saved state or explicit metadata, not both")
        if report is None:
            if recordings is not None and (
                not isinstance(recordings, list)
                or any(not isinstance(row, dict) for row in recordings)
                or any(
                    not {"recording_id", "participant_id"} <= row.keys()
                    for row in recordings
                )
            ):
                raise ValueError(
                    "Supply source records with recording and participant IDs"
                )
            if dataset is not None and not isinstance(dataset, dict):
                raise ValueError("Dataset metadata must be a dictionary")
            recordings = [] if recordings is None else deepcopy(recordings)
            dataset = {} if dataset is None else deepcopy(dataset)
            defaults = dict(
                description=None,
                readme=None,
                participants=None,
                participants_json=None,
                selected_participants=_participant_scope(recordings),
            )
            defaults.update(dataset)
            # Share the portable encoding with source adapters. Readiness here
            # means usable supplied metadata, not inspected or verified files.
            report = dict(
                version=2,
                ready=True,
                issues=[],
                dataset=defaults,
                source_ids=[row["recording_id"] for row in recordings],
                recordings=recordings,
                events=[] if events is None else events,
            )
        if (
            not isinstance(report, dict)
            or type(report.get("version")) is not int
            or report["version"] != 2
        ):
            raise ValueError("Supply version-2 information state")
        if report.get("ready") is not True:
            messages = "; ".join(
                issue["message"]
                for issue in report.get("issues", [])
                if issue.get("severity") == "error"
            )
            raise ValueError(f"The supplied metadata must be ready: {messages}")
        self._report = deepcopy(report)
        try:
            entries = self._report["recordings"]
            if not isinstance(entries, list):
                raise ValueError("Recording descriptors must be a list")
            entries = _recording_info(entries) if entries else []
            ids = [entry["recording_id"] for entry in entries]
            if self._report["source_ids"] != ids:
                raise ValueError("Source IDs and recording descriptors are not aligned")
            participants = _participant_scope(entries)
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
                    raise ValueError("Duplicate event identity in supplied metadata")
                seen.add(identity)
                grouped[recording_id].append(row)
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete supplied metadata") from exc
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
            report["dataset"]["selected_participants"] = _participant_scope(
                report["recordings"]
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
        # A single lookup must not rebuild the identities for every case.
        rows = (
            self._view["rows"] if self._view is not None else self._report["recordings"]
        )
        return rows[index]["participant_id"]

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

    def to_dict(self):
        """Return detached version-2 metadata for restoration with EEGDatasetInfo.

        Returns
        -------
        dict
            Source context and the optional current view, without signal arrays.
            This is metadata serialization, not a certificate of signal contents.
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
        return selected, self[indices]

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

    @property
    def history(self):
        """Detached processing records in append order, empty when unreported."""
        return deepcopy(self._report.get("history", []))


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


def _participant_scope(recordings):
    # Unknown identities stay on aligned rows, not in the known source scope.
    # Separate identifier types when ordering so integer 1 and text "1" survive.
    participants = {
        _identifier(row["participant_id"], "participant_id")
        for row in recordings
        if row["participant_id"] is not None
    }
    return sorted(participants, key=lambda value: (isinstance(value, str), value))


def _recording_info(metadata):
    # Inspecting source metadata and checking loaded arrays must use the same
    # identity/layout rules, without requiring dummy signals for inspection.
    if not isinstance(metadata, (list, tuple)) or not metadata:
        raise ValueError("Supply nonempty recording descriptors.")
    info, ids = [], set()
    for entry in metadata:
        item = deepcopy(dict(entry))
        _required(item, ("channel_names", "sampling_frequency"))
        recording_id = item.get("recording_id", item.get("filepath"))
        if (
            not isinstance(recording_id, str)
            or not recording_id.strip()
            or recording_id in ids
        ):
            raise ValueError(
                "Recording metadata needs a unique nonempty recording_id or filepath."
            )
        ids.add(recording_id)
        item["recording_id"] = recording_id
        item["channel_names"] = _channels(item["channel_names"])
        rate = item["sampling_frequency"]
        if (
            not isinstance(rate, Real)
            or isinstance(rate, bool)
            or not np.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError("sampling_frequency must be finite and positive.")
        item["sampling_frequency"] = float(rate)
        _required(item, ("n_samples",))
        item["n_samples"] = _integer(item["n_samples"], "n_samples")
        if item["n_samples"] < 1:
            raise ValueError("n_samples must be positive.")
        info.append(item)
    return info


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
