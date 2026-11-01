"""Describe classification problems and runs without owning their execution."""

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from aeon_neuro.datasets._classification import ClassificationDataset
from aeon_neuro.datasets._classification_split import validate_classification_split


def resolve_classification_split(
    dataset,
    partitions,
    *,
    disjoint_by,
    source_fields,
    eligible_case_ids=None,
    partition_values=None,
    require_all_classes=(),
):
    """Resolve saved case identities and validate an explicit split policy.

    Parameters
    ----------
    dataset : ClassificationDataset
        Current constructed cases. Storage order may differ from a previous run.
    partitions : mapping of str to sequence of str
        Named partitions of saved case IDs. Names have no implicit meaning.
    disjoint_by : sequence of sequences of str
        Independent compound identities, as in ``validate_classification_split``.
    source_fields : sequence of three str
        Original source identity, inclusive start and exclusive stop fields.
    eligible_case_ids : sequence of str or None, default=None
        Explicit scope. None means every current case. For inner evaluation,
        supply the saved IDs of the parent fitting partition.
    partition_values : mapping or None, default=None
        Optional declared restrictions on partition field values.
    require_all_classes : sequence of str, default=()
        Partitions required to contain all classes in the eligible scope.

    Returns
    -------
    dict
        The independent validation snapshot from
        ``validate_classification_split``, including current indices and IDs.

    Raises
    ------
    ValueError
        If IDs are missing, ambiguous or repeated, or assignments violate policy.

    Notes
    -----
    This checks identities and assignments, not unchanged signals or preparation.
    Use ``validate_classification_problem`` to compare fresh construction with a
    saved problem record. No random split generation or fitting takes place.
    """
    if not isinstance(dataset, ClassificationDataset):
        raise ValueError("dataset must be a ClassificationDataset.")
    ids = _ids([c.get("case_id") for c in dataset.cases], "dataset case IDs")
    positions = {case_id: i for i, case_id in enumerate(ids)}

    def resolve(values):
        values = _ids(values, "saved case IDs")
        if not set(values) <= positions.keys():
            raise ValueError("Saved assignment references missing case IDs.")
        return [positions[value] for value in values]

    if not isinstance(partitions, Mapping):
        raise ValueError("partitions must map names to case IDs.")
    return validate_classification_split(
        dataset,
        {name: resolve(values) for name, values in partitions.items()},
        disjoint_by=disjoint_by,
        source_fields=source_fields,
        eligible_indices=(
            None if eligible_case_ids is None else resolve(eligible_case_ids)
        ),
        partition_values=partition_values,
        require_all_classes=require_all_classes,
    )


def make_classification_problem(dataset, definition, *, splits, sources):
    """Snapshot resolved choices, cases, signal identities and saved assignments.

    Parameters
    ----------
    dataset : ClassificationDataset
        Prepared cases. Samples are fingerprinted one case at a time, without
        altering, padding, flattening or retaining another copy of the signals.
    definition : dict
        JSON-compatible choices. Required fields are ``dataset`` (``id``,
        ``version``, ``selection``), ``target`` (``source``, ``mapping``),
        ``case_definition`` (``description``, ``exclusions``), and ordered
        ``processing`` steps (each with a ``name``). Target mapping is a list
        of ``{value, label}`` records, preserving integer versus string labels.
        Output ``channels`` and ``sampling_frequency`` are resolved from dataset
        metadata when omitted; supplied values must agree with known metadata.
        ``units`` may be a string, a channel-aligned list of strings/nulls, or
        null. Missing units become an explicit unknown. Optional ``origins``
        supplies provenance for explicit choices; derived origins are recorded.
    splits : dict
        Named splits, each with ``partitions`` (case IDs), ``policy`` (arguments
        to ``validate_classification_split``), and optional ``scope``. Scope is
        ``{split, partition}`` identifying a parent partition; omission means all
        cases. Parent references are resolved independently of dictionary order.
        Cycles, incomplete scope and outer cases in inner folds are rejected.
    sources : dict
        Nonempty mapping of source identities to JSON records with ``sha256``
        and ``description``. The caller computes hashes from source bytes and
        should include recipe/configuration evidence. This helper neither opens
        files nor verifies that a supplied digest is true.

    Returns
    -------
    dict
        Independent version-1 JSON record, with ``record_id`` content digest.
        Included cases and exclusions are sorted by ID. Only the mechanical
        ``row`` field is omitted; source provenance and extra fields are kept.
        Array shape, dtype and byte digest are recorded for each included case.

    Raises
    ------
    ValueError
        If the definition, metadata, labels, arrays, sources or splits disagree.

    Notes
    -----
    This is a record boundary, not a dataset loader, experiment runner or storage
    format for signals. Callers own acquisition, preparation, model selection and
    persistence. Fingerprints detect differences; they do not certify scientific
    choices, source authenticity or leakage in earlier processing.
    """
    if not isinstance(dataset, ClassificationDataset):
        raise ValueError("dataset must be a ClassificationDataset.")
    definition = _definition(dataset, definition)
    sources = _json(sources)
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Declare source identities and their content digests.")
    for key, source in sources.items():
        _text(key, "source identity")
        if not isinstance(source, dict):
            raise ValueError("Source identities require digest/description records.")
        _digest(source.get("sha256"), "source sha256")
        _text(source.get("description"), "source description")
    splits, _ = _splits(dataset, splits)
    labels = {entry["label"] for entry in definition["target"]["mapping"]}
    cases = []
    for index, case in enumerate(dataset.cases):
        if case["label"] not in labels:
            raise ValueError("Included class is absent from the target mapping.")
        array = np.asarray(dataset.X[index])
        if (
            array.ndim != 2
            or array.shape[0] != len(definition["channels"])
            or not array.shape[1]
            or array.dtype.kind not in "fiu"
        ):
            raise ValueError("Signals must be numeric channels-by-time arrays.")
        row = {k: v for k, v in case.items() if k != "row"}
        cases.append(
            {
                "case": _json(row),
                "signal": {
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(array).tobytes()
                    ).hexdigest(),
                },
            }
        )
    exclusions = _json(
        [{k: v for k, v in c.items() if k != "row"} for c in dataset.exclusions]
    )
    all_ids = [c["case"]["case_id"] for c in cases]
    for case in exclusions:
        all_ids.append(case.get("case_id"))
        _text(case.get("exclusion_reason"), "exclusion reason")
    _ids(all_ids, "included/excluded case IDs")
    return _seal(
        {
            "schema_version": 1,
            "kind": "classification_problem",
            "definition": definition,
            "sources": sources,
            "cases": sorted(cases, key=lambda c: c["case"]["case_id"]),
            "exclusions": sorted(exclusions, key=lambda c: c["case_id"]),
            "splits": splits,
        }
    )


def validate_classification_problem(dataset, record, *, definition, sources):
    """Compare fresh construction with a saved problem and resolve its assignments.

    Parameters
    ----------
    dataset : ClassificationDataset
        Freshly prepared cases, possibly reordered.
    record : dict
        Saved output from ``make_classification_problem``, including its digest.
    definition : dict
        Current recipe choices, independently supplied rather than assumed from
        the old record. They must resolve to the same definition and origins.
    sources : dict
        Independently recomputed current source identities and digests.

    Returns
    -------
    dict
        Split names mapped to validation snapshots with current array indices.

    Raises
    ------
    ValueError
        If a record, source, setting, case, exclusion, sample or assignment differs.

    Notes
    -----
    Sample equality means identical shape, dtype and bytes (including NaN bits).
    Case storage order and the mechanical ``row`` field may change. All other
    retained provenance must agree. The function does not execute a recipe.
    """
    _check_record(record, "classification_problem")
    fresh = make_classification_problem(
        dataset, definition, splits=record["splits"], sources=sources
    )
    for field in ("definition", "sources", "cases", "exclusions", "splits"):
        if _canonical(fresh[field]) != _canonical(record[field]):
            raise ValueError(f"Reconstructed problem differs in {field}.")
    return _splits(dataset, record["splits"])[1]


def make_classification_experiment(
    problem, *, estimator, parameters, versions, evaluations, selection=None
):
    """Record a caller-executed experiment against a resolved problem.

    Parameters
    ----------
    problem : dict
        Output of ``make_classification_problem``. Its record ID links the run to
        the resolved definition, sources, cases, policies and exact assignments.
    estimator : str
        Qualified estimator/pipeline description. No Python objects are serialized.
    parameters : dict
        Effective estimator and processing parameters, including explicit seeds.
        Caller-supplied JSON; learned operations should be inside the estimator.
    versions : dict
        Nonempty mapping of software names to version strings, including Python.
    evaluations : list of dict
        Each entry specifies ``split``, ``fit_partition``, ``score_partition``,
        ``predictions`` and ``metrics``. Predictions are ordered ``{case_id,
        label, prediction}`` records aligned with the saved scoring partition.
        Optional ``class_order`` and per-row ``probabilities`` must describe the
        complete fitting classes. Metrics are caller-computed JSON records.
        Exact fitting/scoring case IDs are added from the problem's assignments.
    selection : dict or None, default=None
        None records a predeclared estimator. Otherwise supply ``rule``,
        ``candidate_record_ids`` and ``selected_record_id``. The latter must be
        a candidate. The caller retains candidate records and ensures selection
        used only eligible training/validation data; this is not a tuning engine.

    Returns
    -------
    dict
        Independent version-1 JSON experiment record with a content digest.

    Raises
    ------
    ValueError
        If references, prediction alignment, classes or record fields are invalid.

    Notes
    -----
    This function does not fit, score, pickle or introspect an estimator. Recorded
    fitting scope is the caller's assertion, not proof of what a model actually
    saw. Revalidate fresh construction before fitting and check estimator input
    capabilities; no implicit conversion or internal CV is provided here.
    """
    _check_record(problem, "classification_problem")
    _text(estimator, "estimator")
    parameters, versions = _json(parameters), _json(versions)
    if not isinstance(parameters, dict) or not isinstance(versions, dict):
        raise ValueError("parameters and versions must be mappings.")
    _text(versions.get("python"), "Python version")
    for key, value in versions.items():
        _text(key, "software name")
        _text(value, "software version")
    evaluations = _json(evaluations)
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("Supply at least one evaluation.")
    cases = {c["case"]["case_id"]: c["case"] for c in problem["cases"]}
    for evaluation in evaluations:
        try:
            partitions = problem["splits"][evaluation["split"]]["partitions"]
            fit = partitions[evaluation["fit_partition"]]
            score = partitions[evaluation["score_partition"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Evaluation references an unknown split/partition."
            ) from exc
        if evaluation["fit_partition"] == evaluation["score_partition"]:
            raise ValueError("Fitting and scoring partitions must differ.")
        predictions = evaluation.get("predictions")
        if (
            not isinstance(predictions, list)
            or any(not isinstance(p, dict) for p in predictions)
            or [p.get("case_id") for p in predictions] != score
        ):
            raise ValueError("Predictions must follow the saved scoring case IDs.")
        classes = {cases[c]["label"] for c in fit}
        for prediction, case_id in zip(predictions, score):
            if (
                type(prediction.get("label")) is not type(cases[case_id]["label"])
                or prediction["label"] != cases[case_id]["label"]
            ):
                raise ValueError("Prediction target differs from the saved case label.")
            if (
                type(prediction.get("prediction")) not in {type(c) for c in classes}
                or prediction["prediction"] not in classes
            ):
                raise ValueError("Predicted class is absent from fitting cases.")
        _probabilities(evaluation, classes)
        if not isinstance(evaluation.get("metrics"), dict):
            raise ValueError("Supply caller-computed metrics as a mapping.")
        for key, values in (("fit_case_ids", fit), ("score_case_ids", score)):
            if key in evaluation and evaluation[key] != values:
                raise ValueError("Recorded fitting/scoring scope disagrees with split.")
            evaluation[key] = list(values)
    selection = _json(selection)
    if selection is not None:
        if not isinstance(selection, dict):
            raise ValueError("selection must be a record or null.")
        _text(selection.get("rule"), "selection rule")
        candidates = _ids(selection.get("candidate_record_ids"), "candidate records")
        for candidate in candidates:
            _digest(candidate, "candidate record ID")
        if selection.get("selected_record_id") not in candidates:
            raise ValueError("Selected record must be a declared candidate.")
    return _seal(
        {
            "schema_version": 1,
            "kind": "classification_experiment",
            "problem_id": problem["record_id"],
            "estimator": estimator,
            "parameters": parameters,
            "versions": versions,
            "evaluations": evaluations,
            "selection": selection,
        }
    )


def _definition(dataset, definition):
    """Resolve mechanical metadata and validate explicit scientific choices."""
    result = _json(definition)
    if not isinstance(result, dict):
        raise ValueError("definition must be a mapping.")
    for section, fields in (
        ("dataset", ("id", "version", "selection")),
        ("target", ("source",)),
        ("case_definition", ("description", "exclusions")),
    ):
        if not isinstance(result.get(section), dict):
            raise ValueError(f"Supply an explicit {section} record.")
        for field in fields:
            _text(result[section].get(field), f"{section}.{field}")
    mapping = result["target"].get("mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("target.mapping must be a nonempty value/label list.")
    values, labels = [], []
    for row in mapping:
        if not isinstance(row, dict) or set(row) != {"value", "label"}:
            raise ValueError("Target mapping entries require value and label.")
        for key, out in (("value", values), ("label", labels)):
            value = row[key]
            if type(value) not in (str, int) or value == "":
                raise ValueError("Target values/labels must be strings or integers.")
            out.append(value)
    if len(set(values)) != len(values) or len({type(v) for v in labels}) != 1:
        raise ValueError("Target values must be unique and labels uniformly typed.")
    processing = result.get("processing")
    if not isinstance(processing, list):
        raise ValueError("processing must be an ordered list; [] means no added steps.")
    for step in processing:
        if not isinstance(step, dict):
            raise ValueError("Each processing step must be a named record.")
        _text(step.get("name"), "processing step name")
    origins = result.setdefault("origins", {})
    if not isinstance(origins, dict):
        raise ValueError("origins must map definition fields to provenance records.")
    for key in ("dataset", "target", "case_definition", "processing"):
        origins.setdefault(key, {"kind": "explicit"})
    for key, meta_key in (
        ("channels", "channel_names"),
        ("sampling_frequency", "sampling_frequency"),
        ("units", "units"),
    ):
        known = _json(dataset.metadata.get(meta_key))
        if key in result:
            if known is not None and result[key] != known:
                raise ValueError(f"Explicit {key} disagrees with dataset metadata.")
            origins.setdefault(key, {"kind": "explicit"})
        else:
            result[key] = known
            origins[key] = (
                {"kind": "metadata", "source": f"dataset.metadata.{meta_key}"}
                if known is not None
                else {"kind": "unknown"}
            )
    _ids(result["channels"], "output channel names")
    rate = result["sampling_frequency"]
    if type(rate) not in (int, float) or rate <= 0:
        raise ValueError("Output sampling_frequency must be finite and positive.")
    units = result["units"]
    if isinstance(units, list):
        if len(units) != len(result["channels"]):
            raise ValueError("Per-channel units must align with output channels.")
        for unit in units:
            if unit is not None:
                _text(unit, "channel unit")
    elif units is not None:
        _text(units, "units")
    for key, origin in origins.items():
        if key not in result or key == "origins" or not isinstance(origin, dict):
            raise ValueError("Origins must identify definition fields.")
        if origin.get("kind") not in {"explicit", "metadata", "unknown"}:
            raise ValueError("Origin kind must be explicit, metadata or unknown.")
        if origin["kind"] == "metadata":
            _text(origin.get("source"), "metadata origin")
        if origin["kind"] == "unknown" and result[key] is not None:
            raise ValueError("Unknown origins require an explicit null value.")
    return result


def _splits(dataset, splits):
    """Resolve parent scopes without assigning meaning to partition names."""
    splits = _json(splits)
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Supply named saved splits.")
    reports, active = {}, set()

    def resolve(name):
        if name in active:
            raise ValueError("Split scopes must not contain cycles.")
        if name in reports:
            return reports[name]
        _text(name, "split name")
        active.add(name)
        spec = splits[name]
        if not isinstance(spec, dict) or not {"partitions", "policy"} <= set(spec):
            raise ValueError("Splits require partitions and policy.")
        if set(spec) - {"partitions", "policy", "scope"}:
            raise ValueError("Unknown split fields.")
        scope = spec.get("scope")
        eligible = None
        if scope is not None:
            if not isinstance(scope, dict) or set(scope) != {"split", "partition"}:
                raise ValueError("Scope must identify a parent split and partition.")
            _text(scope["split"], "parent split")
            _text(scope["partition"], "parent partition")
            if scope["split"] not in splits:
                raise ValueError("Unknown parent split.")
            parent = resolve(scope["split"])
            if scope["partition"] not in parent["partitions"]:
                raise ValueError("Unknown parent partition.")
            eligible = parent["partitions"][scope["partition"]]["case_ids"]
        policy = spec["policy"]
        allowed = {
            "disjoint_by",
            "source_fields",
            "partition_values",
            "require_all_classes",
        }
        if (
            not isinstance(policy, dict)
            or not {"disjoint_by", "source_fields"} <= set(policy)
            or set(policy) - allowed
        ):
            raise ValueError("Supply a split policy; scope is declared separately.")
        report = resolve_classification_split(
            dataset, spec["partitions"], eligible_case_ids=eligible, **policy
        )
        spec["scope"], spec["policy"] = scope, report["policy"]
        reports[name] = report
        active.remove(name)
        return report

    for name in splits:
        resolve(name)
    return splits, reports


def _probabilities(evaluation, classes):
    """Check optional probabilities with an explicit complete class order."""
    rows = evaluation["predictions"]
    present = ["probabilities" in row for row in rows]
    if "class_order" not in evaluation and not any(present):
        return
    order = evaluation.get("class_order")
    if (
        not isinstance(order, list)
        or len(order) != len(classes)
        or any(type(value) not in {type(c) for c in classes} for value in order)
        or set(order) != classes
        or not all(present)
    ):
        raise ValueError(
            "Probabilities require complete explicit class order and rows."
        )
    if any(
        not isinstance(row["probabilities"], list)
        or any(type(value) not in (int, float) for value in row["probabilities"])
        for row in rows
    ):
        raise ValueError("Probabilities must be numeric lists, not strings/booleans.")
    values = np.asarray([row["probabilities"] for row in rows], dtype=float)
    if (
        values.shape != (len(rows), len(order))
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1)
    ):
        raise ValueError("Invalid class probabilities.")


def _ids(values, name):
    """Validate identifiers before allowing lookup or coercion."""
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a nonempty sequence of unique strings.")
    values = list(values)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must be a nonempty sequence of unique strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique.")
    return values


def _text(value, name):
    """Require meaningful text without inventing missing facts."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _digest(value, name):
    """Require a canonical SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")


def _json(value):
    """Copy JSON values while rejecting coercions and executable objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json(v) for k, v in value.items()}
    raise ValueError("Records require finite JSON values and string mapping keys.")


def _canonical(value):
    """Encode independent of dictionary insertion order."""
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"))


def _seal(record):
    """Identify a plain record by its complete content."""
    record["record_id"] = hashlib.sha256(_canonical(record).encode()).hexdigest()
    return record


def _check_record(record, kind):
    """Reject changed records and unsupported formats before using assignments."""
    record = _json(record)
    if (
        not isinstance(record, dict)
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("kind") != kind
    ):
        raise ValueError("Unsupported classification record kind/version.")
    identity = record.pop("record_id", None)
    if _seal(record)["record_id"] != identity:
        raise ValueError("Classification record content digest differs.")
