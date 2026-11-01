"""Exercise portable problem records against fresh construction and caller runs."""

import hashlib
import json
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from aeon_neuro.datasets import (
    build_classification_dataset,
    make_classification_experiment,
    make_classification_problem,
    make_fixed_length_cases,
    resolve_classification_split,
    validate_classification_problem,
)


@pytest.fixture
def cohort():
    """Construct three pairs of people, with two cases per person."""
    signals = np.arange(2400, dtype=float).reshape(6, 2, 200)
    metadata = [
        {
            "recording_id": f"p{i}",
            "channel_names": ["C3", "C4"],
            "sampling_frequency": 100,
        }
        for i in range(6)
    ]
    cases = make_fixed_length_cases(
        signals,
        metadata,
        labels=[0, 1, 0, 1, 0, 1],
        groups=[f"p{i}" for i in range(6)],
        window_size=1,
    )
    for case in cases:
        case["participant_id"] = case["group_id"]
    dataset = build_classification_dataset(signals, metadata, cases)
    ids = [c["case_id"] for c in dataset.cases]
    policy = {
        "disjoint_by": [["participant_id"]],
        "source_fields": ["recording_id", "start_sample", "stop_sample"],
        "require_all_classes": ["development", "held-out"],
    }
    splits = {
        "outer": {
            "partitions": {"development": ids[:8], "held-out": ids[8:]},
            "policy": policy,
        },
        "inner": {
            "partitions": {"fit": ids[:4], "check": ids[4:8]},
            "policy": dict(policy, require_all_classes=["fit", "check"]),
            "scope": {"split": "outer", "partition": "development"},
        },
    }
    definition = {
        "dataset": {"id": "generated", "version": "1", "selection": "all six people"},
        "target": {
            "source": "explicit participant labels",
            "mapping": [
                {"value": "control", "label": 0},
                {"value": "case", "label": 1},
            ],
        },
        "case_definition": {
            "description": "one-second complete windows",
            "exclusions": "discard incomplete trailing windows",
        },
        "processing": [],
    }
    sources = {
        "recordings": {
            "sha256": hashlib.sha256(signals.tobytes()).hexdigest(),
            "description": "generated recordings",
        }
    }
    return dataset, definition, splits, sources


def freeze(cohort):
    """Make a record from the shared recipe inputs."""
    dataset, definition, splits, sources = cohort
    return make_classification_problem(
        dataset, definition, splits=splits, sources=sources
    )


def test_reordered_construction_and_json_round_trip(cohort):
    """Saved assignments survive JSON and case reordering, including child scopes."""
    record = freeze(cohort)
    ds, definition, _, sources = cohort
    saved = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    order = list(reversed(range(len(ds.cases))))
    ds.X, ds.y, ds.groups = ds.X[order], ds.y[order], ds.groups[order]
    ds.cases = [dict(ds.cases[i], row=j) for j, i in enumerate(order)]
    report = validate_classification_problem(
        ds, saved, definition=definition, sources=sources
    )
    assert report["outer"]["partitions"]["development"]["indices"] == list(
        range(11, 3, -1)
    )
    assert report["inner"]["eligible_indices"] == list(range(11, 3, -1))
    assert record == freeze(cohort)


def test_metadata_resolution_and_independent_records(cohort):
    """Derive only known output facts, preserve unknown units and avoid mutation."""
    ds, definition, splits, sources = cohort
    before = deepcopy((definition, splits, sources, ds.cases))
    record = freeze(cohort)
    resolved = record["definition"]
    assert resolved["channels"] == ["C3", "C4"]
    assert resolved["sampling_frequency"] == 100
    assert resolved["units"] is None
    assert resolved["origins"]["channels"] == {
        "kind": "metadata",
        "source": "dataset.metadata.channel_names",
    }
    assert resolved["origins"]["units"] == {"kind": "unknown"}
    record["splits"]["outer"]["partitions"]["development"].clear()
    assert (definition, splits, sources, ds.cases) == before
    ds.metadata["units"] = ["V", None]
    assert freeze(cohort)["definition"]["units"] == ["V", None]


@pytest.mark.parametrize(
    "change",
    [
        "samples",
        "dtype",
        "label",
        "group",
        "bounds",
        "extra_case_field",
        "missing_case",
        "source",
        "settings",
        "channels",
        "rate",
        "units",
        "exclusions",
    ],
)
def test_reconstruction_rejects_changed_inputs(cohort, change):
    """Detect changes beyond row order, including signals and excluded candidates."""
    record = freeze(cohort)
    ds, definition, _, sources = cohort
    if change == "samples":
        ds.X[0, 0, 0] += 1
    elif change == "dtype":
        ds.X = ds.X.astype(np.float32)
    elif change == "label":
        ds.cases[0]["label"] = 1
        ds.y[0] = 1
    elif change == "group":
        ds.cases[0]["group_id"] = "different"
        ds.groups = ds.groups.astype(object)
        ds.groups[0] = "different"
    elif change == "bounds":
        ds.cases[0]["stop_sample"] -= 1
    elif change == "extra_case_field":
        ds.cases[0]["event_source"] = "changed"
    elif change == "missing_case":
        ds.cases.pop()
    elif change == "source":
        sources["recordings"]["sha256"] = "0" * 64
    elif change == "settings":
        definition["processing"] = [{"name": "demean"}]
    elif change == "channels":
        ds.metadata["channel_names"] = ["C4", "C3"]
    elif change == "rate":
        ds.metadata["sampling_frequency"] = 200
    elif change == "units":
        ds.metadata["units"] = "V"
    else:
        ds.exclusions.append({"case_id": "excluded", "exclusion_reason": "new rule"})
    with pytest.raises(ValueError):
        validate_classification_problem(
            ds, record, definition=definition, sources=sources
        )


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "ambiguous_mapping",
        "mixed_labels",
        "origin",
        "unknown_origin",
        "channels",
        "rate",
        "units",
        "processing",
        "source",
        "non_json",
        "nonfinite",
        "scope_override",
        "scope_cycle",
        "missing_parent",
        "test_in_inner",
        "duplicate_id",
        "excluded_duplicate",
        "empty_prediction_problem",
    ],
)
def test_invalid_problem_contracts_fail(cohort, change):
    """Reject contradictory choices and malformed assignments before recording."""
    ds, definition, splits, sources = cohort
    if change == "target":
        definition["target"]["mapping"].pop()
    elif change == "ambiguous_mapping":
        definition["target"]["mapping"].append({"value": "case", "label": 0})
    elif change == "mixed_labels":
        definition["target"]["mapping"][1]["label"] = "1"
    elif change == "origin":
        definition["origins"] = {"target": {"kind": "guessed"}}
    elif change == "unknown_origin":
        definition["origins"] = {"target": {"kind": "unknown"}}
    elif change == "channels":
        definition["channels"] = ["C4", "C3"]
    elif change == "rate":
        definition["sampling_frequency"] = 200
    elif change == "units":
        definition["units"] = ["V"]
    elif change == "processing":
        definition["processing"] = [{}]
    elif change == "source":
        sources["recordings"]["sha256"] = "not-a-digest"
    elif change == "non_json":
        definition["callable"] = lambda: None
    elif change == "nonfinite":
        definition["threshold"] = float("nan")
    elif change == "scope_override":
        splits["outer"]["policy"]["eligible_indices"] = [0, 1]
    elif change == "scope_cycle":
        splits["outer"]["scope"] = {"split": "inner", "partition": "fit"}
    elif change == "missing_parent":
        splits["inner"]["scope"]["split"] = "absent"
    elif change == "test_in_inner":
        splits["inner"]["partitions"]["fit"][0] = ds.cases[-1]["case_id"]
    elif change == "duplicate_id":
        ds.cases[1]["case_id"] = ds.cases[0]["case_id"]
    elif change == "excluded_duplicate":
        ds.exclusions = [dict(ds.cases[0], exclusion_reason="bad")]
    else:
        splits.clear()
    with pytest.raises(ValueError):
        freeze(cohort)


def test_unequal_cases_are_fingerprinted_without_conversion(cohort):
    """Preserve unequal shapes and NaNs; capabilities belong to the estimator caller."""
    ds = cohort[0]
    ds.X = [x[:, : 99 if i % 2 else 100].copy() for i, x in enumerate(ds.X)]
    ds.X[0][0, 0] = np.nan
    record = freeze(cohort)
    assert {c["signal"]["shape"][1] for c in record["cases"]} == {99, 100}
    validate_classification_problem(ds, record, definition=cohort[1], sources=cohort[3])


def test_resolve_ids_rejects_missing_and_duplicate_assignments(cohort):
    """ID resolution does not silently drop cases or accept stale positions."""
    ds, _, splits, _ = cohort
    outer = splits["outer"]
    for bad in [["missing"], [ds.cases[0]["case_id"]] * 2, [0], "not-a-list"]:
        with pytest.raises(ValueError):
            resolve_classification_split(
                ds, dict(outer["partitions"], development=bad), **outer["policy"]
            )


def experiment(problem):
    """Describe a caller's held-out predictions without fitting a model."""
    by_id = {c["case"]["case_id"]: c["case"] for c in problem["cases"]}
    ids = problem["splits"]["outer"]["partitions"]["held-out"]
    return {
        "estimator": "example.Pipeline",
        "parameters": {"random_state": 42},
        "versions": {"python": "3.11", "example": "1"},
        "evaluations": [
            {
                "split": "outer",
                "fit_partition": "development",
                "score_partition": "held-out",
                "class_order": [0, 1],
                "predictions": [
                    {
                        "case_id": c,
                        "label": by_id[c]["label"],
                        "prediction": 0,
                        "probabilities": [0.6, 0.4],
                    }
                    for c in ids
                ],
                "metrics": {"accuracy": 0.5},
            }
        ],
    }


def test_experiment_scope_outputs_and_selection_are_independent(cohort):
    """Link caller outputs to exact assignments and explicit candidate records."""
    problem = freeze(cohort)
    kwargs = experiment(problem)
    first = make_classification_experiment(problem, **kwargs)
    kwargs["selection"] = {
        "rule": "maximum mean inner balanced accuracy",
        "candidate_record_ids": [first["record_id"]],
        "selected_record_id": first["record_id"],
    }
    result = make_classification_experiment(problem, **kwargs)
    assert result["problem_id"] == problem["record_id"]
    assert (
        result["evaluations"][0]["fit_case_ids"]
        == problem["splits"]["outer"]["partitions"]["development"]
    )
    assert "fit_case_ids" not in kwargs["evaluations"][0]
    json.dumps(result, allow_nan=False)
    kwargs["parameters"]["random_state"] = 7
    assert result["parameters"]["random_state"] == 42


@pytest.mark.parametrize(
    "change",
    [
        "tamper",
        "version",
        "order",
        "label",
        "prediction",
        "bool_prediction",
        "probabilities",
        "class_order",
        "bool_class_order",
        "string_probability",
        "scope",
        "same_partition",
        "selection",
        "versions",
    ],
)
def test_invalid_experiments_fail(cohort, change):
    """Reject inconsistent record references, output alignment and probabilities."""
    problem = freeze(cohort)
    kwargs = experiment(problem)
    evaluation = kwargs["evaluations"][0]
    if change == "tamper":
        problem["definition"]["units"] = "V"
    elif change == "version":
        problem["schema_version"] = True
    elif change == "order":
        evaluation["predictions"].reverse()
    elif change == "label":
        evaluation["predictions"][0]["label"] = 1
    elif change == "prediction":
        evaluation["predictions"][0]["prediction"] = 2
    elif change == "bool_prediction":
        evaluation["predictions"][0]["prediction"] = False
    elif change == "probabilities":
        evaluation["predictions"][0]["probabilities"] = [1.2, -0.2]
    elif change == "class_order":
        evaluation["class_order"] = [0, 0]
    elif change == "bool_class_order":
        evaluation["class_order"] = [False, True]
    elif change == "string_probability":
        evaluation["predictions"][0]["probabilities"] = ["0.6", "0.4"]
    elif change == "scope":
        evaluation["fit_case_ids"] = ["wrong"]
    elif change == "same_partition":
        evaluation["fit_partition"] = "held-out"
    elif change == "selection":
        kwargs["selection"] = {
            "rule": "declared",
            "candidate_record_ids": ["0" * 64],
            "selected_record_id": "1" * 64,
        }
    else:
        kwargs["versions"].pop("python")
    with pytest.raises(ValueError):
        make_classification_experiment(problem, **kwargs)


@pytest.mark.parametrize("kind", ["cohort", "sessions"])
def test_example_selection_and_learned_fitting_scope(kind, monkeypatch):
    """Execute the shipped recipe and observe the exact inputs to learned scaling."""
    from sklearn.preprocessing import StandardScaler

    example = (
        Path(__file__).resolve().parents[3] / "examples/reproducible_classification.py"
    )
    if not example.exists():
        pytest.skip("Source-tree example is not included in the installed wheel.")
    functions = runpy.run_path(str(example))
    observed = []
    original_fit = StandardScaler.fit

    def fit(self, X, y=None, **kwargs):
        observed.append(np.array(X, copy=True))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    result = functions["run_example"](kind)
    ds, _, _, _ = functions["construct"](kind)
    ids = {c["case_id"]: i for i, c in enumerate(ds.cases)}
    evaluations = [e for r in result["candidates"] for e in r["evaluations"]]
    evaluations += result["selected"]["evaluations"]
    assert len(observed) == len(evaluations) == 5
    held_out = set(result["problem"]["splits"]["outer"]["partitions"]["held-out"])
    for actual, evaluation in zip(observed, evaluations):
        fitting = evaluation["fit_case_ids"]
        assert not held_out.intersection(fitting)
        expected = functions["log_variance"](ds.X[[ids[c] for c in fitting]])
        np.testing.assert_array_equal(actual, expected)
    candidates = result["candidates"]
    assert all(e["split"] != "outer" for r in candidates for e in r["evaluations"])
    best = int(
        np.argmax(
            [
                np.mean([e["metrics"]["balanced_accuracy"] for e in r["evaluations"]])
                for r in candidates
            ]
        )
    )
    assert (
        result["selected"]["selection"]["selected_record_id"]
        == candidates[best]["record_id"]
    )
    assert result["selected"]["problem_id"] == result["baseline"]["problem_id"]
    again = functions["run_example"](kind)
    assert again == result
