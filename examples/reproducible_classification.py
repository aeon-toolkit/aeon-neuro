"""Reconstruct cases, select on saved inner folds and retain ordinary model runs.

Run ``python examples/reproducible_classification.py OUTPUT_DIRECTORY``. Signals
are generated for interface checks, not research results. No data downloads or
companion package are needed. Preparation, fitting and persistence remain here;
aeon-neuro only constructs cases, validates assignments and retains records.
"""

import argparse
import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import aeon
import numpy as np
import sklearn
from aeon.classification import DummyClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import aeon_neuro
from aeon_neuro.datasets import (
    build_classification_dataset,
    make_classification_experiment,
    make_classification_problem,
    make_event_cases,
    make_fixed_length_cases,
    validate_classification_problem,
)


def construct(kind):
    """Use existing case helpers with two deliberately different policies."""
    rng = np.random.default_rng(42)
    shared = {"source_fields": ["recording_id", "start_sample", "stop_sample"]}
    if kind == "cohort":
        X = rng.normal(size=(6, 2, 200)) * 1e-6
        metadata = [
            {
                "recording_id": f"p{i}",
                "channel_names": ["C3", "C4"],
                "sampling_frequency": 100,
            }
            for i in range(6)
        ]
        cases = make_fixed_length_cases(
            X,
            metadata,
            labels=[0, 1, 0, 1, 0, 1],
            groups=[f"p{i}" for i in range(6)],
            window_size=1,
        )
        for case in cases:
            case["participant_id"] = case["group_id"]
        policy = dict(shared, disjoint_by=[["participant_id"]])
        inner_policy = policy
        timing = "complete nonoverlapping one-second cohort-labelled windows"
    elif kind == "sessions":
        X = rng.normal(size=(2, 2, 800)) * 1e-6
        metadata = [
            {
                "recording_id": s,
                "channel_names": ["C3", "C4"],
                "sampling_frequency": 100,
            }
            for s in ("T", "E")
        ]
        events = [
            {
                "recording_id": s,
                "onset": i,
                "label": i % 2,
                "group_id": s,
                "participant_id": "p0",
                "session": s,
                "event_id": f"{s}-{i}",
            }
            for s in ("T", "E")
            for i in range(8)
        ]
        cases = make_event_cases(X, metadata, events, tmin=0, tmax=1)
        policy = dict(
            shared,
            disjoint_by=[["participant_id", "session"]],
            partition_values={
                "development": {"session": ["T"]},
                "held-out": {"session": ["E"]},
            },
        )
        inner_policy = dict(shared, disjoint_by=[["event_id"]])
        timing = "event label column; [0, 1) seconds relative to each onset"
    else:
        raise ValueError("Choose cohort or sessions.")
    dataset = build_classification_dataset(X, metadata, cases)
    # This known unit is a fact of the generated signal recipe, not an inferred
    # property of floating-point values or a BIDS task name.
    dataset.metadata["units"] = "V"
    ids = [c["case_id"] for c in dataset.cases]
    splits = {
        "outer": {
            "partitions": {"development": ids[:8], "held-out": ids[8:]},
            "policy": dict(policy, require_all_classes=["development", "held-out"]),
        },
        "fold-1": {
            "partitions": {"fit": ids[:4], "check": ids[4:8]},
            "policy": dict(inner_policy, require_all_classes=["fit", "check"]),
            "scope": {"split": "outer", "partition": "development"},
        },
        "fold-2": {
            "partitions": {"fit": ids[4:8], "check": ids[:4]},
            "policy": dict(inner_policy, require_all_classes=["fit", "check"]),
            "scope": {"split": "outer", "partition": "development"},
        },
    }
    definition = {
        "dataset": {
            "id": f"generated-{kind}",
            "version": "1",
            "selection": "all generated recordings",
            "seed": 42,
        },
        "target": {
            "source": "explicit recipe labels",
            "mapping": [{"value": 0, "label": 0}, {"value": 1, "label": 1}],
        },
        "case_definition": {
            "description": timing,
            "exclusions": "none for these complete generated cases",
        },
        "processing": [],
    }
    sources = {
        "recordings": {
            "sha256": hashlib.sha256(X.tobytes()).hexdigest(),
            "description": "generated recording bytes before extraction",
        },
        "recipe": {
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "description": "this executable preparation and evaluation recipe",
        },
    }
    return dataset, definition, splits, sources


def log_variance(X):
    """Explicitly extract channel features without fitting or implicit flattening."""
    return np.log(np.maximum(np.var(X, axis=2), 1e-30))


def pipeline(C):
    """Keep learned scaling and classification inside each fitting partition."""
    estimator = make_pipeline(
        FunctionTransformer(log_variance, validate=False),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000, random_state=42),
    )
    parameters = {
        "features": {
            **estimator[0].get_params(),
            "func": "log_variance (source identified by recipe digest)",
            "axis": 2,
            "floor": 1e-30,
        },
        "scaler": estimator[1].get_params(),
        "classifier": estimator[2].get_params(),
        "input_capabilities": {
            "multivariate": True,
            "unequal_length": False,
            "missing_values": False,
        },
    }
    return estimator, parameters


def evaluate(dataset, problem, resolved, estimator, parameters, split_names):
    """Fit ordinary clones on resolved indices and explicitly retain their outputs."""
    if not isinstance(dataset.X, np.ndarray) or dataset.X.ndim != 3:
        raise ValueError("This example requires equal-length EEG cases.")
    if not np.isfinite(dataset.X).all():
        raise ValueError("This example does not impute missing/infinite samples.")
    if hasattr(estimator, "get_tag") and dataset.X.shape[1] > 1:
        if not estimator.get_tag("capability:multivariate", False):
            raise ValueError("Estimator does not support multivariate input.")
    evaluations = []
    for name in split_names:
        fit_role, score_role = (
            ("development", "held-out") if name == "outer" else ("fit", "check")
        )
        partitions = resolved[name]["partitions"]
        fit, score = partitions[fit_role]["indices"], partitions[score_role]["indices"]
        model = clone(estimator)
        with threadpool_limits(limits=1):
            model.fit(dataset.X[fit], dataset.y[fit])
            predicted = model.predict(dataset.X[score])
            probabilities = model.predict_proba(dataset.X[score])
        evaluations.append(
            {
                "split": name,
                "fit_partition": fit_role,
                "score_partition": score_role,
                "class_order": model.classes_.tolist(),
                "predictions": [
                    {
                        "case_id": dataset.cases[index]["case_id"],
                        "label": dataset.y[index].item(),
                        "prediction": predicted[j].item(),
                        "probabilities": probabilities[j].tolist(),
                    }
                    for j, index in enumerate(score)
                ],
                "metrics": {
                    "balanced_accuracy": float(
                        balanced_accuracy_score(dataset.y[score], predicted)
                    )
                },
            }
        )
    return make_classification_experiment(
        problem,
        estimator=f"{type(estimator).__module__}.{type(estimator).__name__}",
        parameters=parameters,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "aeon": aeon.__version__,
            "aeon-neuro": aeon_neuro.__version__,
            "scipy": version("scipy"),
            "threadpoolctl": version("threadpoolctl"),
        },
        evaluations=evaluations,
    )


def run_example(kind):
    """Select using inner folds and score the chosen settings once on outer TEST."""
    dataset, definition, splits, sources = construct(kind)
    problem = make_classification_problem(
        dataset, definition, splits=splits, sources=sources
    )
    # Execute preparation again. Reordering deliberately tests identity-based
    # reconstruction; source hashes are independently recomputed by construct().
    fresh, definition, _, sources = construct(kind)
    order = list(reversed(range(len(fresh.cases))))
    fresh.X, fresh.y, fresh.groups = fresh.X[order], fresh.y[order], fresh.groups[order]
    fresh.cases = [dict(fresh.cases[i], row=j) for j, i in enumerate(order)]
    resolved = validate_classification_problem(
        fresh, problem, definition=definition, sources=sources
    )
    candidates = []
    settings = [0.1, 1.0]
    for C in settings:
        estimator, parameters = pipeline(C)
        candidates.append(
            evaluate(
                fresh, problem, resolved, estimator, parameters, ["fold-1", "fold-2"]
            )
        )
    means = [
        np.mean([e["metrics"]["balanced_accuracy"] for e in run["evaluations"]])
        for run in candidates
    ]
    # The tie rule is declared before observing held-out outcomes.
    best = int(np.argmax(means))
    estimator, parameters = pipeline(settings[best])
    held_out = evaluate(fresh, problem, resolved, estimator, parameters, ["outer"])
    selected = make_classification_experiment(
        problem,
        estimator=held_out["estimator"],
        parameters=held_out["parameters"],
        versions=held_out["versions"],
        evaluations=held_out["evaluations"],
        selection={
            "rule": "maximum mean inner balanced accuracy; ties choose smaller C",
            "candidate_record_ids": [r["record_id"] for r in candidates],
            "selected_record_id": candidates[best]["record_id"],
        },
    )
    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline_run = evaluate(
        fresh, problem, resolved, baseline, baseline.get_params(), ["outer"]
    )
    return {
        "problem": problem,
        "candidates": candidates,
        "selected": selected,
        "baseline": baseline_run,
    }


def main():
    """Save the two examples to an explicitly chosen new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for kind in ("cohort", "sessions"):
        result = run_example(kind)
        (args.output / f"{kind}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    logging.basicConfig(level=logging.INFO)
    logging.info("Saved both constructed-problem examples to %s", args.output)


if __name__ == "__main__":
    main()
