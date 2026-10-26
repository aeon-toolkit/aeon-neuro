"""Assign labels to BIDS EEG recordings using the file, events or TSV metadata."""

import os
from collections import Counter

import numpy as np
import pandas as pd


def label_bids_recordings(
    X,
    metadata,
    class_names,
    bids_path="../",
    label_level="recording",
    label_source="auto",
    label_column=None,
    sfreq=None,
):
    """
    Assign labels to BIDS EEG recordings or events.

    Searches recording metadata and, when requested, BIDS TSV files.
    Event-level labeling returns one row per event rather than one
    label per recording.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        EEG recordings.

    metadata : list of dict
        Metadata corresponding to X.

    class_names : list of str
        Exact class names. Matching is case-sensitive.

    bids_path : str, default="../"
        Root directory of the BIDS dataset.

    label_level : {"recording", "event"}, default="recording"
        "recording": assign one label per recording.
        "event": read event labels from events.tsv.

    label_source : str, default="auto"
        For recording-level labeling:
            "auto": search metadata, then participants.tsv.
            "metadata": search recording metadata only.
            "participants": search participants.tsv only.
            "events": search the recording's events.tsv.
            A TSV path: use that TSV as the label source.

        For event-level labeling:
            "auto" or "events": discover the matching events.tsv
            for each recording.
            A TSV path: use that TSV for all recordings.

    label_column : str or None, default=None
        TSV column containing the class label.
        For event-level labeling, defaults to "trial_type".
        For recording-level labeling, None searches all values in
        the relevant TSV rows for exact class-name matches.

    Returns
    -------
    For label_level="recording":
        X : original recordings
        y : object ndarray, one label per recording
        class_counts : dict
        unmatched_count : int
        label_info : list of dict with label provenance

    For label_level="event":
        X : original recordings
        events : pandas.DataFrame with event labels and TSV provenance
        class_counts : dict
        unmatched_count : int
        label_info : list of dict with TSV provenance
    """
    # Validate inputs.
    if len(X) != len(metadata):
        raise ValueError("X and metadata must have the same length.")

    if not class_names:
        raise ValueError("Provide at least one class name.")

    if any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError("Class names must be non-empty strings.")

    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must not contain duplicates.")

    if label_level not in {"recording", "event"}:
        raise ValueError("label_level must be 'recording' or 'event'.")

    if not isinstance(label_source, str) or not label_source:
        raise ValueError("label_source must be a non-empty string.")

    if label_level == "event" and label_source not in {"auto", "events"}:
        explicit_tsv = label_source
    elif label_source not in {"auto", "metadata", "participants", "events"}:
        explicit_tsv = label_source
    else:
        explicit_tsv = None

    bids_path = os.path.abspath(os.fspath(bids_path))

    if explicit_tsv is not None:
        if not os.path.isabs(explicit_tsv):
            explicit_tsv = os.path.join(bids_path, explicit_tsv)
        explicit_tsv = os.path.abspath(explicit_tsv)

    # Event-level labeling and EEG segmentation.
    if label_level == "event":
        event_column = label_column or "trial_type"
        event_rows = []
        label_info = []
        segments = []

        for recording_index, record in enumerate(metadata):
            paths = _get_recording_paths(record, bids_path)

            tsv_path = explicit_tsv if explicit_tsv is not None else paths["events_tsv"]

            info = {
                "filepath": record.get("filepath"),
                "label_source_tsv": None,
                "label_column": event_column,
            }

            if not os.path.isfile(tsv_path):
                label_info.append(info)
                continue

            df = _read_tsv(tsv_path)

            if df is None or df.empty:
                label_info.append(info)
                continue

            if event_column not in df.columns:
                raise ValueError(
                    f"Column {event_column!r} not found "
                    f"in {tsv_path!r}. Available columns: "
                    f"{list(df.columns)}"
                )

            if "onset" not in df.columns:
                raise ValueError(f"'onset' column not found in {tsv_path!r}.")

            # Resolve the sampling frequency.
            current_sfreq = sfreq

            if current_sfreq is None:
                current_sfreq = record.get(
                    "sfreq",
                    record.get("sampling_frequency"),
                )

            if current_sfreq is None:
                raise ValueError(
                    f"Sampling frequency is missing for "
                    f"{record.get('filepath')!r}. Pass sfreq "
                    f"or add 'sfreq' to the recording metadata."
                )

            current_sfreq = float(current_sfreq)

            if not np.isfinite(current_sfreq) or current_sfreq <= 0:
                raise ValueError(f"Invalid sampling frequency: " f"{current_sfreq!r}")

            # EEG data must have shape (channels, samples).
            recording = np.asarray(X[recording_index])

            if recording.ndim != 2:
                raise ValueError(
                    f"Recording {recording_index} must have "
                    f"shape (channels, samples); got "
                    f"{recording.shape}."
                )

            n_samples = recording.shape[1]

            info["label_source_tsv"] = os.path.relpath(tsv_path, bids_path)
            label_info.append(info)

            for row_index, row in df.iterrows():
                value = str(row[event_column])

                # Event labels must exactly match class names.
                label = value if value in class_names else None

                try:
                    onset = float(row["onset"])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid onset in {tsv_path!r}, " f"row {row_index}."
                    )

                if not np.isfinite(onset) or onset < 0:
                    raise ValueError(
                        f"Invalid onset {onset!r} in " f"{tsv_path!r}, row {row_index}."
                    )

                duration = None

                if "duration" in df.columns and str(row["duration"]).strip() not in {
                    "",
                    "n/a",
                    "NA",
                    "nan",
                }:
                    try:
                        duration = float(row["duration"])
                    except (TypeError, ValueError):
                        duration = None

                # Convert event onset and duration to sample indices.
                start_sample = int(round(onset * current_sfreq))

                end_sample = (
                    start_sample + int(round(duration * current_sfreq))
                    if duration is not None and np.isfinite(duration) and duration > 0
                    else None
                )

                segment = None

                # Extract a segment only when duration is valid
                # and the event overlaps the recording.
                if (
                    end_sample is not None
                    and start_sample < n_samples
                    and end_sample > start_sample
                ):
                    segment_start = max(0, start_sample)
                    segment_end = min(n_samples, end_sample)

                    if segment_end > segment_start:
                        segment = recording[:, segment_start:segment_end].copy()

                event = row.to_dict()
                event.update(
                    {
                        "label": label,
                        "onset_seconds": onset,
                        "duration_seconds": duration,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "segment": segment,
                        "source_tsv": os.path.relpath(tsv_path, bids_path),
                        "source_row": int(row_index),
                        "recording_index": recording_index,
                        "recording_filepath": record.get("filepath"),
                        "subject": record.get("subject"),
                        "session": record.get("session"),
                        "task": record.get("task"),
                    }
                )

                event_rows.append(event)

                # One segment per event, aligned with events rows.
                segments.append(segment)

        events = pd.DataFrame(event_rows)

        if events.empty:
            events = pd.DataFrame(
                columns=[
                    "label",
                    "onset_seconds",
                    "duration_seconds",
                    "start_sample",
                    "end_sample",
                    "segment",
                    "source_tsv",
                    "source_row",
                    "recording_index",
                    "recording_filepath",
                    "subject",
                    "session",
                    "task",
                ]
            )

        class_counts = (
            {name: int((events["label"] == name).sum()) for name in class_names}
            if not events.empty
            else {name: 0 for name in class_names}
        )

        unmatched_count = int(events["label"].isna().sum()) if not events.empty else 0

        # Return segmented EEG arrays rather than the original recordings.
        return (
            segments,
            events,
            class_counts,
            unmatched_count,
            label_info,
        )

    # Recording-level labeling.
    y = []
    label_info = []

    for record in metadata:
        paths = _get_recording_paths(record, bids_path)
        matches = set()
        used_source = None
        used_column = None

        # Search filename, relative path and BIDS metadata.
        if label_source in {"auto", "metadata"}:
            search_fields = [
                record.get("filepath", ""),
                record.get("filename", ""),
                record.get("subject", ""),
                record.get("session") or "",
                record.get("task") or "",
                record.get("acq") or "",
                record.get("run") or "",
            ]

            for field in search_fields:
                matches.update(_find_class_matches(field, class_names))

            if matches:
                used_source = "metadata"

        # Determine which TSV files may provide labels.
        if explicit_tsv is not None:
            candidates = [explicit_tsv]
        elif label_source == "participants":
            candidates = [paths["participants_tsv"]]
        elif label_source == "events":
            candidates = [paths["events_tsv"]]
        elif label_source == "auto":
            recording_tsv = os.path.splitext(paths["filepath"])[0] + ".tsv"

            candidates = [
                recording_tsv,
                paths["participants_tsv"],
                paths["events_tsv"],
            ]
        else:
            candidates = []

        # Search TSVs only when the preceding source did not match.
        if not matches:
            for tsv_path in candidates:
                if not os.path.isfile(tsv_path):
                    continue

                tsv_matches, column = _get_tsv_class_matches(
                    record,
                    tsv_path,
                    paths,
                    class_names,
                    label_column,
                )

                if tsv_matches:
                    matches.update(tsv_matches)
                    used_source = os.path.relpath(tsv_path, bids_path)
                    used_column = column

                # Stop once this TSV provides evidence.
                if tsv_matches:
                    break

        label = next(iter(matches)) if len(matches) == 1 else None

        y.append(label)
        label_info.append(
            {
                "filepath": record.get("filepath"),
                "label": label,
                "label_source_tsv": (
                    used_source if used_source not in {None, "metadata"} else None
                ),
                "label_source": used_source,
                "label_column": used_column,
                "candidate_matches": sorted(matches),
            }
        )

    class_counts = {name: 0 for name in class_names}
    class_counts.update(Counter(label for label in y if label is not None))

    unmatched_count = sum(label is None for label in y)

    return (
        X,
        np.asarray(y, dtype=object),
        class_counts,
        unmatched_count,
        label_info,
    )


def _find_class_matches(text, class_names):
    """Find case-sensitive class names contained in the text."""
    text = str(text)
    return {name for name in class_names if name in text}


def _read_tsv(tsv_path):
    """Read a TSV while preserving empty strings."""
    try:
        return pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
    except (
        OSError,
        pd.errors.ParserError,
        UnicodeError,
    ):
        return None


def _get_recording_paths(record, bids_path):
    """Get paths associated with a BIDS recording."""
    filepath = os.path.abspath(
        os.path.join(
            bids_path,
            os.fspath(record["filepath"]),
        )
    )

    eeg_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    stem = os.path.splitext(basename)[0]

    subject_dir = os.path.join(bids_path, f"sub-{record['subject']}")

    session = record.get("session")
    session_dir = (
        os.path.join(subject_dir, f"ses-{session}") if session else subject_dir
    )

    event_stem = stem[:-4] if stem.endswith("_eeg") else stem

    return {
        "filepath": filepath,
        "directory": eeg_dir,
        "basename": basename,
        "stem": stem,
        "subject_dir": subject_dir,
        "session_dir": session_dir,
        "events_tsv": os.path.join(eeg_dir, event_stem + "_events.tsv"),
        "participants_tsv": os.path.join(bids_path, "participants.tsv"),
    }


def _get_tsv_rows(tsv_path, record, paths):
    """Select TSV rows identifying the recording or participant."""
    df = _read_tsv(tsv_path)

    if df is None or df.empty:
        return None, None

    actual_tsv = os.path.normcase(os.path.abspath(tsv_path))

    recording_tsv = os.path.splitext(paths["filepath"])[0] + ".tsv"

    recording_tsv = os.path.normcase(os.path.abspath(recording_tsv))

    events_tsv = os.path.normcase(os.path.abspath(paths["events_tsv"]))

    if actual_tsv in {recording_tsv, events_tsv}:
        return df, df

    # Match participant or recording identifiers in dataset-level TSVs.
    identifiers = {
        str(record.get("subject", "")),
        f"sub-{record.get('subject', '')}",
        paths["basename"],
        paths["stem"],
    }

    id_columns = {
        "participant_id",
        "subject",
        "subject_id",
        "filename",
        "file",
        "recording",
        "recording_id",
    }

    columns = [col for col in df.columns if col.lower() in id_columns]

    if not columns:
        return df, df.iloc[0:0]

    mask = pd.Series(False, index=df.index)

    for col in columns:
        mask |= df[col].astype(str).isin(identifiers)

    return df, df.loc[mask]


def _get_tsv_class_matches(record, tsv_path, paths, class_names, label_column):
    """Find class labels in relevant TSV rows."""
    df, relevant = _get_tsv_rows(tsv_path, record, paths)

    if df is None or relevant is None or relevant.empty:
        return set(), None

    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Column {label_column!r} not found in {tsv_path!r}.")
        values = relevant[label_column].tolist()
    else:
        values = relevant.to_numpy().ravel().tolist()

    matches = set()

    for value in values:
        matches.update(_find_class_matches(value, class_names))

    return matches, label_column
