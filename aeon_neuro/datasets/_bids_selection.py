"""Discover BIDS EEG recordings without silently dropping requested identifiers."""

from mne_bids import find_matching_paths


def _find_bids_paths(
    root,
    *,
    subjects=None,
    sessions=None,
    tasks=None,
    acqs=None,
    runs=None,
    extensions=(".vhdr", ".bdf", ".edf"),
):
    filters = {
        "subjects": (subjects, "subject"),
        "sessions": (sessions, "session"),
        "tasks": (tasks, "task"),
        "acqs": (acqs, "acquisition"),
        "runs": (runs, "run"),
    }
    paths = find_matching_paths(
        root,
        subjects=subjects,
        sessions=sessions,
        tasks=tasks,
        acquisitions=acqs,
        runs=runs,
        datatypes=["eeg"],
        extensions=list(extensions),
        check=False,
    )
    missing = []
    for name, (requested, entity) in filters.items():
        if requested is None:
            continue
        values = [requested] if isinstance(requested, str) else requested
        matched = {getattr(path, entity) for path in paths}
        absent = sorted(set(values) - matched)
        if absent:
            missing.append((name, entity, absent))
    if missing:
        # Checking the combined selection catches values excluded by another
        # filter too. It does not require every Cartesian-product combination.
        # An unfiltered inventory is needed only for a useful failure message.
        inventory = find_matching_paths(
            root,
            datatypes=["eeg"],
            extensions=list(extensions),
            check=False,
        )
        details = []
        for name, entity, absent in missing:
            available = sorted({getattr(path, entity) for path in inventory} - {None})
            display = repr(available[:20]) + (" ..." if len(available) > 20 else "")
            details.append(f"{name} {absent!r}; dataset values: {display}")
        raise ValueError(
            "No EEG recordings found for requested values within the combined "
            "selection: " + "; ".join(details)
        )
    if not paths:
        raise ValueError(
            f"No EEG recordings found in {str(root)!r} matching the selection."
        )
    return paths
