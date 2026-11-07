"""Define different classification problems on the same loaded recordings.

Run from a source checkout with ``python examples/problem_loading.py``.
All signals and metadata are generated; no dataset download is required.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from aeon.classification.dummy import DummyClassifier

from aeon_neuro.datasets import (
    load_classification_problem,
    validate_classification_split,
)


def main():
    """Construct event/cohort cases, then fit on a declared participant split."""
    # These stand in for arrays and metadata returned by load_bids_data or a
    # caller's reader. Class meanings come from the binding, not the reader.
    recordings = np.random.default_rng(42).normal(0, 1e-6, (4, 3, 256))
    metadata = [
        {
            "recording_id": f"recording-{i}",
            "subject": f"person-{i}",
            "session": "first",
            "cohort": "control" if i % 2 == 0 else "condition",
            "channel_names": ["C4", "C3", "Cz"],
            "sampling_frequency": 32,
            "units": "V",
        }
        for i in range(4)
    ]
    events = [
        {
            "recording_id": item["recording_id"],
            "trial_id": f"trial-{trial}",
            "cue_seconds": onset,
            "condition": 11 if trial % 2 == 0 else 22,
            "response": "correct" if trial in (0, 3) else "incorrect",
        }
        for item in metadata
        for trial, onset in enumerate((1, 3, 5, 7))
    ]
    folder = Path(__file__).with_name("problem_definitions")
    problem_file = folder / "imagery.problem.json"
    binding_file = folder / "imagery.binding.json"
    imagery = load_classification_problem(
        recordings, metadata, problem_file, binding_file, events=events
    )
    expected = [
        recordings[i, [1, 0], onset * 32 : onset * 32 + 16]
        for i in range(4)
        for onset in (1, 3, 5, 7)
    ]
    np.testing.assert_array_equal(imagery.X, expected)

    # A different target reuses the source intervals and case identities while
    # changing the problem identity and explicit label mapping.
    response_problem = json.loads(problem_file.read_text())
    response_binding = json.loads(binding_file.read_text())
    response_problem.update(
        id="response-correctness",
        question="Classify the recorded response outcome",
        class_labels=["correct", "incorrect"],
    )
    response_binding["problem"]["id"] = response_problem["id"]
    response_binding["target"].update(
        column="response", mapping={"correct": "correct", "incorrect": "incorrect"}
    )
    response = load_classification_problem(
        recordings, metadata, response_problem, response_binding, events=events
    )
    np.testing.assert_array_equal(response.X, imagery.X)
    assert [c["case_id"] for c in response.cases] == [
        c["case_id"] for c in imagery.cases
    ]
    assert not np.array_equal(response.y, imagery.y)

    window_problem = {
        "format_version": 1,
        "id": "cohort-windows",
        "version": "1",
        "question": "Classify a recording's declared cohort using one-second windows",
        "class_labels": ["control", "condition"],
        "segmentation": {"mode": "fixed_windows", "window_size": 1},
    }
    window_binding = deepcopy(response_binding)
    window_binding["problem"]["id"] = window_problem["id"]
    window_binding.pop("events")
    window_binding["target"] = {
        "source": "recording",
        "column": "cohort",
        "mapping": {"control": "control", "condition": "condition"},
    }
    cohort = load_classification_problem(
        recordings, metadata, window_problem, window_binding
    )
    assert cohort.X.shape == (32, 3, 32)

    # Construction does not choose a generalisation question. Keep this actual
    # participant-held-out assignment explicit and validate before fitting.
    split = validate_classification_split(
        imagery,
        {"train": list(range(8)), "test": list(range(8, 16))},
        disjoint_by=[("participant_id",)],
        source_fields=("recording_id", "start_sample", "stop_sample"),
        require_all_classes=("train", "test"),
    )
    train = split["partitions"]["train"]["indices"]
    test = split["partitions"]["test"]["indices"]
    model = DummyClassifier(strategy="prior").fit(imagery.X[train], imagery.y[train])
    predictions = model.predict(imagery.X[test])
    assert predictions.shape == (len(test),)
    logging.getLogger(__name__).info(
        "%s",
        json.dumps(
            {
                "imagery_shape": list(imagery.X.shape),
                "response_shape": list(response.X.shape),
                "cohort_shape": list(cohort.X.shape),
                "train_cases": len(train),
                "test_cases": len(test),
                "definition_sha256": imagery.metadata["problem"]["definition_sha256"],
                "scope": "Generated interface check, not an EEG performance estimate",
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
