"""Authored problems preserve processed eager/deferred case semantics."""

import numpy as np

from aeon_neuro.datasets import load_bids_dataset, make_classification_problem
from aeon_neuro.datasets.tests import test_dataset_info as fixtures
from aeon_neuro.datasets.tests.test_info_problem import _info_problem_documents

source = fixtures.source


def test_processed_recordings_keep_eager_deferred_case_layout_and_provenance(source):
    """A changed input grid is used for extraction while source joins stay original."""
    import mne

    from aeon_neuro.datasets import (
        materialize_classification_problem,
        plan_classification_problem,
    )

    X, info = load_bids_dataset(source[0])
    Y, changed = info.process(X, "crop", tmin=0.5, tmax=4.0)
    Y, changed = changed.process(Y, "resample", sfreq=16)
    Y, changed = changed.process(Y, "scale", factor=1e6, units="uV")
    problem, binding = _info_problem_documents("event_epochs")
    eager = make_classification_problem(Y, changed, problem, binding)
    np.testing.assert_array_equal(eager.X[0], Y[0, [1, 0], 16:32])
    assert eager.info.units == ("uV", "uV")
    assert eager.info.layouts[0]["source_origin_sample"] == 48
    assert eager.info.layouts[0]["source_sample_step"] == 2
    assert eager.info.recordings[0].sfreq == 32
    assert eager.cases[0]["source_event"]["acquisition_event"]["onset"] == "1"
    plan = plan_classification_problem(changed, problem, binding)
    sources = {row["recording_id"]: i for i, row in enumerate(info.metadata)}

    def opener(descriptor):
        index = sources[descriptor["recording_id"]]
        return mne.io.RawArray(
            Y[index], mne.create_info(["C3", "C4"], 16, "eeg"), verbose=False
        )

    deferred = materialize_classification_problem(plan, opener)
    np.testing.assert_array_equal(deferred.X, eager.X)
    assert deferred.info.to_dict()["view"] == eager.info.to_dict()["view"]
    for section in ("source", "history", "problem"):
        assert (
            deferred.metadata["provenance"][section]
            == eager.metadata["provenance"][section]
        )
