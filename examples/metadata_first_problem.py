"""Plan a large catalogue, then demonstrate deferred caching with tiny fixtures.

The 128 GiB catalogue is metadata only: its signal files do not exist. The
separate materialization example copies small local generated files to model
a remote source and a training cache. No network download is performed.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np

from aeon_neuro.datasets import (
    materialize_classification_problem,
    plan_classification_problem,
    resolve_classification_problem,
)


def main():
    """Keep authoring and selection independent of recording acquisition."""
    folder = Path(__file__).with_name("problem_definitions")
    problem = folder / "imagery.problem.json"
    binding = folder / "imagery.binding.json"
    resolved = resolve_classification_problem(problem, binding)
    channels = ["C3", "C4", *[f"EEG{i}" for i in range(62)]]
    catalogue = [
        {
            "recording_id": f"R{i}",
            "subject": f"P{i}",
            "session": "first",
            "channel_names": channels,
            "sampling_frequency": 512,
            "n_samples": 2**29,
            "units": "V",
            "release": "example-release-1",
            "url": f"https://example.invalid/large-{i}.edf",
            "source_size_bytes": 64 * 2**30 + 16640,
        }
        for i in range(2)
    ]
    events = [
        {
            "recording_id": row["recording_id"],
            "trial_id": str(trial),
            "cue_seconds": onset,
            "condition": 11 if trial % 2 == 0 else 22,
        }
        for row in catalogue
        for trial, onset in enumerate((1, 3, 5, 7))
    ]
    large_plan = plan_classification_problem(catalogue, problem, binding, events=events)
    assert large_plan["dense_shape"] == [8, 2, 256]
    assert large_plan["estimated_case_bytes_float64"] == 32768
    assert sum(row["source_size_bytes"] for row in catalogue) > 128 * 2**30

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, cache = root / "source-fixture", root / "training-cache"
        source.mkdir()
        cache.mkdir()
        small_catalogue = []
        values = np.random.default_rng(42).normal(0, 1e-6, (2, 2, 256))
        for i in range(2):
            filename = f"R{i}_raw.fif"
            raw = mne.io.RawArray(
                values[i], mne.create_info(["C3", "C4"], 32, "eeg"), verbose=False
            )
            raw.save(source / filename, fmt="double", verbose=False)
            raw.close()
            small_catalogue.append(
                {
                    "recording_id": f"R{i}",
                    "subject": f"P{i}",
                    "session": "first",
                    "channel_names": ["C3", "C4"],
                    "sampling_frequency": 32,
                    "n_samples": 256,
                    "units": "V",
                    "release": "generated-fixture-1",
                    "filename": filename,
                    "expected_sha256": hashlib.sha256(
                        (source / filename).read_bytes()
                    ).hexdigest(),
                }
            )
        plan = plan_classification_problem(
            small_catalogue, problem, binding, events=events
        )
        # This small JSON plan can be saved, inspected and used to choose cases
        # while the training cache still contains no signal files.
        plan = json.loads(json.dumps(plan))
        assert not list(cache.iterdir())
        copied = []

        def open_recording(descriptor):
            """Model acquisition at the explicit training preparation boundary."""
            path = cache / descriptor["filename"]
            if not path.exists():
                shutil.copyfile(source / descriptor["filename"], path)
                copied.append(descriptor["recording_id"])
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != descriptor["expected_sha256"]
            ):
                raise ValueError("Cached fixture checksum differs from the catalogue")
            return mne.io.read_raw_fif(path, preload=False, verbose=False)

        selected = [
            row["case_id"] for row in plan["cases"] if row["recording_id"] == "R1"
        ]
        dataset = materialize_classification_problem(
            plan, open_recording, case_ids=selected
        )
        assert copied == ["R1"]
        assert [path.name for path in cache.iterdir()] == ["R1_raw.fif"]
        np.testing.assert_array_equal(
            dataset.X,
            [values[1, :, start : start + 16] for start in (32, 96, 160, 224)],
        )
        again = materialize_classification_problem(
            plan, open_recording, case_ids=selected
        )
        assert copied == ["R1"]
        np.testing.assert_array_equal(again.X, dataset.X)
        logging.getLogger(__name__).info(
            "%s",
            json.dumps(
                {
                    "large_catalogue_source_bytes": sum(
                        row["source_size_bytes"] for row in catalogue
                    ),
                    "large_plan_case_bytes_float64": large_plan[
                        "estimated_case_bytes_float64"
                    ],
                    "definition_sha256": resolved["definition_sha256"],
                    "cached_recordings": copied,
                    "materialized_shape": list(dataset.X.shape),
                    "scope": (
                        "Metadata-only size demonstration plus tiny local cache "
                        "fixtures; no large-data/network benchmark"
                    ),
                },
                indent=2,
            ),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
