"""Validate supplied case assignments without imposing an evaluation protocol."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from aeon_neuro.datasets._classification import (
    ClassificationDataset,
    _identifier,
    _integer,
)


def validate_classification_split(
    dataset,
    partitions,
    *,
    disjoint_by,
    source_fields,
    eligible_indices=None,
    partition_values=None,
    require_all_classes=(),
):
    """Check supplied assignments against an explicit classification policy.

    Parameters
    ----------
    dataset : ClassificationDataset
        Constructed cases with aligned signals, labels and groups. Case identities
        must remain unique and their labels/groups must agree with the arrays.
    partitions : mapping of str to sequence of int
        At least two named, nonempty partitions of global case indices. Together
        they must cover exactly the eligible cases, without duplicate assignments.
        Names have no implicit meaning; this function does not generate splits.
    disjoint_by : sequence of sequences of str
        Nonempty list of independent identity keys. Each inner sequence defines
        one compound identity, for example ``[("participant_id", "session")]``.
        Every specified identity must be present and confined to one partition.
        Use ``[("participant_id",)]`` for an unseen-person experiment instead.
    source_fields : sequence of three str
        Case fields identifying an original source coordinate grid, its inclusive
        start sample and exclusive stop sample. For unmodified source recordings,
        use ``("recording_id", "start_sample", "stop_sample")``. Copies or crops
        need a shared original-source identity and bounds in that original grid.
        Overlapping source intervals across partitions are rejected independently
        of the grouping policy; touching intervals and within-partition overlap
        are allowed. Bounds must be nonnegative integers with stop greater than
        start. No file discovery, hashing or coordinate conversion is performed.
    eligible_indices : sequence of int or None, default=None
        Explicit scope of global indices. None means all cases. For an inner
        validation split, supply the parent training indices so that outer test
        cases cannot enter that split. The caller establishes the parent scope.
    partition_values : mapping or None, default=None
        Optional ``{partition: {field: allowed_values}}`` restrictions. For example,
        ``{"train": {"session": ["T"]}, "test": {"session": ["E"]}}`` rejects
        swapped session roles. Fields and nonempty allowed values are explicit.
    require_all_classes : sequence of str, default=()
        Partition names that must each contain every class in the eligible scope.
        For example, ``("train",)`` rejects held-out labels absent from training.
        No class-coverage requirement is inferred from a partition's name.

    Returns
    -------
    dict
        Independent snapshot containing ``eligible_indices``, ``policy`` and
        ``partitions``. Each partition records ordered ``indices`` and ``case_ids``.
        Values use ordinary Python lists, strings and integers. The snapshot does
        not remain validated if the dataset or the assignments are later changed.

    Raises
    ------
    ValueError
        If cases, policy, scope, assignments, identities, roles, class coverage or
        source intervals violate the declared requirements.

    Notes
    -----
    This is a separate policy-aware path; ``validate_group_split`` retains its
    conservative independent-group behavior. A session policy must not substitute
    for participant independence when the question concerns unseen people.

    Equal source identities assert a common original coordinate grid. Distinct
    recording IDs alone cannot reveal copies of the same physical source. Supply
    canonical source identities and original bounds when aliases or transforms
    are involved. Include any filter/sequence context in the supplied support.
    Validation cannot discover missing identity links, certify earlier fitting,
    check hidden estimator CV or infer chronology across separate sources. Shared
    information provenance additionally enforces its original source support,
    including dependencies broader than the displayed case intervals.
    """
    if not isinstance(dataset, ClassificationDataset):
        raise ValueError("dataset must be a ClassificationDataset.")
    size = len(dataset.cases)
    if (
        len(dataset.X) != size
        or np.shape(dataset.y) != (size,)
        or np.shape(dataset.groups) != (size,)
    ):
        raise ValueError("Dataset signals, labels, groups and cases are not aligned.")
    identities = []
    for index, row in enumerate(dataset.cases):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Case identities must be unique nonempty strings.")
        identities.append(case_id)
        if (
            _identifier(row.get("label"), "label") != dataset.y[index]
            or _identifier(row.get("group_id"), "group_id") != dataset.groups[index]
        ):
            raise ValueError("Case labels/groups no longer agree with the arrays.")
    if len(set(identities)) != size:
        raise ValueError("Case identities must be unique nonempty strings.")

    scope = (
        _case_indices(np.arange(size), size)
        if eligible_indices is None
        else _case_indices(eligible_indices, size)
    )
    if not isinstance(partitions, Mapping) or len(partitions) < 2:
        raise ValueError("Supply at least two named partitions.")
    names = _field_names(list(partitions), "partition names")
    indices = {name: _case_indices(partitions[name], size) for name in names}
    assigned = [index for values in indices.values() for index in values]
    if len(set(assigned)) != len(assigned) or set(assigned) != set(scope):
        raise ValueError("Partitions must cover eligible cases exactly once.")

    if not isinstance(disjoint_by, (list, tuple)) or not disjoint_by:
        raise ValueError("disjoint_by must declare compound identity fields.")
    keys = [_field_names(fields, "disjoint_by") for fields in disjoint_by]
    if len(set(keys)) != len(keys):
        raise ValueError("disjoint_by must not repeat an identity rule.")
    source_fields = _field_names(source_fields, "source_fields")
    if len(source_fields) != 3:
        raise ValueError("source_fields must name source, start and stop fields.")
    required = _field_names(require_all_classes, "require_all_classes", empty=True)
    if not set(required) <= set(names):
        raise ValueError("Class coverage names must identify supplied partitions.")
    roles = _role_rules(partition_values, names)

    membership = [{} for _ in keys]
    sources = {}
    classes = {dataset.y[index] for index in scope}
    for name, values in indices.items():
        if name in required and {dataset.y[index] for index in values} != classes:
            raise ValueError(
                f"Partition {name!r} does not contain all eligible classes."
            )
        for index in values:
            row = dataset.cases[index]
            for fields, seen in zip(keys, membership):
                identity = tuple(_identity(row.get(field), field) for field in fields)
                previous = seen.setdefault(identity, name)
                if previous != name:
                    raise ValueError(f"Identity {fields!r} crosses split partitions.")
            for field, allowed in roles.get(name, {}).items():
                if _identity(row.get(field), field) not in allowed:
                    raise ValueError(f"Partition {name!r} forbids value of {field!r}.")
            source = _identity(row.get(source_fields[0]), source_fields[0])
            start = _integer(row.get(source_fields[1]), source_fields[1])
            stop = _integer(row.get(source_fields[2]), source_fields[2])
            if start < 0 or stop <= start:
                raise ValueError("Source bounds must satisfy 0 <= start < stop.")
            sources.setdefault(source, []).append((start, stop, name))

    _check_source_overlap(sources)
    if dataset.metadata.get("provenance", {}).get("source") is not None:
        # A displayed window can depend on a broader, earlier processing input.
        _validate_info_source_support(dataset.info, indices)

    return {
        "eligible_indices": scope,
        "policy": {
            "disjoint_by": [list(fields) for fields in keys],
            "source_fields": list(source_fields),
            "partition_values": roles,
            "require_all_classes": list(required),
        },
        "partitions": {
            name: {"indices": values, "case_ids": [identities[i] for i in values]}
            for name, values in indices.items()
        },
    }


def _check_source_overlap(
    sources, message="Source sample intervals overlap across partitions."
):
    for intervals in sources.values():
        stops = {}
        # Track the furthest stop for each partition: an immediately preceding
        # interval can be shorter than another still-overlapping interval.
        for start, stop, name in sorted(intervals):
            if any(end > start for part, end in stops.items() if part != name):
                raise ValueError(message)
            stops[name] = max(stops.get(name, 0), stop)


def _validate_info_source_support(info, partitions):
    from aeon_neuro.datasets._info_view import _recording_view

    view = info._view if info._view is not None else _recording_view(info)
    sources = {}
    for name, indices in partitions.items():
        for index in indices:
            row = view.rows[index]
            # Array-only cases have no original-source identity. Equal shapes or
            # generated IDs cannot establish a shared recording coordinate grid.
            if row.recording_id is not None:
                start, stop = row.source_support
                sources.setdefault(row.recording_id, []).append((start, stop, name))
    _check_source_overlap(
        sources, "Original source support overlaps across partitions."
    )


def _case_indices(values, size):
    """Reject boolean indices before NumPy can coerce a mixed list to integers."""
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or not values.size:
        raise ValueError("Case indices must be nonempty one-dimensional integers.")
    if any(
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or value < 0
        or value >= size
        for value in values
    ) or len(set(values)) != len(values):
        raise ValueError("Case indices must be unique in-range integers, not booleans.")
    return [int(value) for value in values]


def _field_names(values, name, *, empty=False):
    """Require unambiguous sequences rather than treating strings as field lists."""
    if (
        not isinstance(values, (list, tuple))
        or (not values and not empty)
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must be a sequence of unique nonempty field names.")
    return tuple(values)


def _identity(value, field):
    """Keep unresolved BIDS identities from becoming an ordinary split group."""
    value = _identifier(value, field)
    if isinstance(value, str) and value.strip().lower() == "n/a":
        raise ValueError(f"Missing identity for {field!r}.")
    return value


def _role_rules(rules, names):
    """Copy and validate optional role constraints without guessing role names."""
    if rules is None:
        return {}
    if not isinstance(rules, Mapping) or not set(rules) <= set(names):
        raise ValueError("partition_values must reference supplied partitions.")
    result = {}
    for name, fields in rules.items():
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("Partition value rules must map fields to allowed values.")
        _field_names(list(fields), "partition_values")
        result[name] = {}
        for field, allowed in fields.items():
            if not isinstance(allowed, (list, tuple)) or not allowed:
                raise ValueError("Allowed partition values must be nonempty sequences.")
            result[name][field] = [_identity(value, field) for value in allowed]
    return result
