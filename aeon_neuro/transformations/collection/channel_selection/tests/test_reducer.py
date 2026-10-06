"""Tests for case and time reduction."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from aeon_neuro.transformations.collection.channel_selection import CaseTimeReducer


class _ConstantClassifier(ClassifierMixin, BaseEstimator):
    """Small deterministic proxy used by reducer tests."""

    def fit(self, X, y):
        """Store the first observed class."""
        self.class_ = np.unique(y)[0]
        return self

    def predict(self, X):
        """Predict the stored class."""
        return np.repeat(self.class_, X.shape[0])


class _FailingClassifier(ClassifierMixin, BaseEstimator):
    """Proxy that always fails during fitting."""

    def fit(self, X, y):
        """Raise a deliberate fitting error."""
        raise TypeError("deliberate proxy failure")

    def predict(self, X):
        """Return placeholder predictions."""
        return np.zeros(X.shape[0], dtype=int)


@pytest.fixture
def reduction_data():
    """Create a balanced collection for reducer tests."""
    rng = np.random.RandomState(0)
    return rng.normal(size=(24, 3, 20)), np.repeat([0, 1], 12)


def test_case_subsets_are_nested(reduction_data):
    """Larger case fractions should contain all cases from smaller fractions."""
    _, y = reduction_data
    reducer = CaseTimeReducer(random_state=7)
    available = np.arange(len(y))

    small = reducer._sample_case_indices(y, 0.25, available)
    medium = reducer._sample_case_indices(y, 0.5, available)

    assert set(small).issubset(medium)


@pytest.mark.parametrize("time_reduction", ["resample", "subsample"])
def test_fit_resample_and_transform_shapes(reduction_data, time_reduction):
    """The learned reduction should apply cases only to TRAIN and time to both."""
    X, y = reduction_data
    reducer = CaseTimeReducer(
        proxy_estimator=_ConstantClassifier(),
        strategy="joint",
        case_fractions=(0.5, 1.0),
        time_fractions=(0.5, 1.0),
        min_timepoints=1,
        time_reduction=time_reduction,
        random_state=0,
    )

    X_train, y_train = reducer.fit_resample(X, y)
    X_test = reducer.transform(X[:5])

    assert X_train.shape == (
        reducer.n_cases_selected_,
        X.shape[1],
        reducer.n_timepoints_selected_,
    )
    assert y_train.shape == (reducer.n_cases_selected_,)
    assert X_test.shape == (5, X.shape[1], reducer.n_timepoints_selected_)


def test_reduction_cost_uses_quadratic_case_scaling():
    """The cost proxy should reflect TDE's case-comparison workload."""
    X = np.ones((20, 2, 10))
    y = np.repeat([0, 1], 10)
    reducer = CaseTimeReducer(
        proxy_estimator=_ConstantClassifier(),
        strategy="case",
        case_fractions=(0.5, 1.0),
        random_state=0,
    ).fit(X, y)

    results = reducer.candidate_results_.set_index("candidate")
    half_cases = results.loc["case_0.5"]
    expected = half_cases["n_cases_final_train"] ** 2 * half_cases["n_timepoints_final"]

    assert half_cases["reduction_cost"] == expected


def test_fail_fast_controls_proxy_errors(reduction_data):
    """Proxy failures should be recorded or re-raised according to fail_fast."""
    X, y = reduction_data

    reducer = CaseTimeReducer(
        proxy_estimator=_FailingClassifier(),
        strategy="time",
        fail_fast=False,
        random_state=0,
    )
    with pytest.raises(RuntimeError, match="All proxy candidate evaluations failed"):
        reducer.fit(X, y)
    assert reducer.candidate_results_["error"].notna().all()

    reducer.set_params(fail_fast=True)
    with pytest.raises(TypeError):
        reducer.fit(X, y)


def test_summary_marks_score_as_tuning_score(reduction_data):
    """The summary should not present candidate selection as an unbiased score."""
    X, y = reduction_data
    reducer = CaseTimeReducer(
        proxy_estimator=_ConstantClassifier(),
        strategy="time",
        random_state=0,
    ).fit(X, y)

    summary = reducer.get_reduction_summary()

    assert summary["score_is_tuning_score"] is True
    assert summary["score"] == reducer.selection_score_
