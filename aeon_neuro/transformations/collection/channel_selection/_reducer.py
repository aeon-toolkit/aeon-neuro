"""Case/time reduction for component-level time series experiments."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class CaseTimeReducer(BaseEstimator):
    """Select a reduced training set and/or reduced time axis using a proxy.

    This is intended for research experiments where channel selection has already
    been applied and the aim is to reduce the cost of fitting a downstream time
    series classifier component.

    The reducer evaluates candidate reductions using a training-only validation
    split. It can reduce cases, time points, or both. Case reduction is applied
    only to the training set, while time reduction is applied consistently to both
    training and test sets.

    Parameters
    ----------
    proxy_estimator : estimator or None, default=None
        Classifier used to evaluate candidate reductions. If None, the class
        attempts to construct a small aeon IndividualTDE proxy.
    strategy : {"auto", "case", "time", "both", "joint"}, default="auto"
        Candidate family to evaluate.

        - "auto": use case reduction if n_cases > n_timepoints, else time
          reduction.
        - "case": evaluate case reduction candidates only.
        - "time": evaluate time reduction candidates only.
        - "both": evaluate case-only and time-only candidates.
        - "joint": evaluate all combinations of case and time reductions.
    case_fractions : tuple of float, default=(0.125, 0.25, 0.5, 1.0)
        Fractions of training cases to evaluate. Sampling is stratified where
        possible.
    time_fractions : tuple of float, default=(0.125, 0.25, 0.5, 1.0)
        Fractions of time points to evaluate. Time indices are evenly spaced.
    time_lengths : tuple of int or None, default=None
        Explicit time lengths to evaluate. If provided, these are added to the
        candidates generated from time_fractions.
    proxy_time_length : int or None, default=512
        For case-only candidates, optionally cap the time axis used in proxy
        evaluation. This does not affect the final transformed data unless a
        time-reduction candidate is selected. If None, no cap is applied.
    validation_size : float, default=0.33
        Fraction of the training set used as the internal validation set.
    scoring : {"balanced_accuracy", "accuracy"}, default="balanced_accuracy"
        Metric used to score proxy validation predictions.
    tolerance : float, default=0.01
        Select the cheapest candidate whose score is within tolerance of the
        best candidate score.
    min_cases_per_class : int, default=1
        Minimum number of selected cases per class where possible.
    min_timepoints : int, default=10
        Minimum number of time points for time-reduction candidates.
    time_reduction : {"resample", "subsample"}, default="resample"
        Method used to reduce the time axis. ``"resample"`` uses Fourier
        resampling, which applies anti-alias filtering. ``"subsample"`` retains
        evenly spaced original observations and is mainly provided for comparison
        with earlier experiments.
    fail_fast : bool, default=False
        If True, re-raise proxy evaluation errors immediately. If False, record
        failed candidates in ``candidate_results_`` and continue.
    random_state : int or None, default=None
        Random seed.
    n_jobs : int, default=1
        Used only by the default IndividualTDE proxy when supported.
    """

    def __init__(
        self,
        proxy_estimator: Any | None = None,
        strategy: str = "auto",
        case_fractions: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0),
        time_fractions: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0),
        time_lengths: tuple[int, ...] | None = None,
        proxy_time_length: int | None = 512,
        validation_size: float = 0.33,
        scoring: str = "balanced_accuracy",
        tolerance: float = 0.01,
        min_cases_per_class: int = 1,
        min_timepoints: int = 10,
        time_reduction: str = "resample",
        fail_fast: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.proxy_estimator = proxy_estimator
        self.strategy = strategy
        self.case_fractions = case_fractions
        self.time_fractions = time_fractions
        self.time_lengths = time_lengths
        self.proxy_time_length = proxy_time_length
        self.validation_size = validation_size
        self.scoring = scoring
        self.tolerance = tolerance
        self.min_cases_per_class = min_cases_per_class
        self.min_timepoints = min_timepoints
        self.time_reduction = time_reduction
        self.fail_fast = fail_fast
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the reducer on the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training collection, usually after channel selection.
        y : array-like of shape (n_cases,)
            Training labels.

        Returns
        -------
        self
            Fitted reducer.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self._validate_inputs(X, y)

        n_cases, n_channels, n_timepoints = X.shape
        self.n_cases_in_ = n_cases
        self.n_channels_in_ = n_channels
        self.n_timepoints_in_ = n_timepoints

        train_idx, val_idx = self._make_train_val_indices(y)
        candidates = self._make_candidates(n_cases, n_timepoints)

        rows = []
        for candidate in candidates:
            row = self._evaluate_candidate(
                X=X,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                candidate=candidate,
            )
            rows.append(row)

        self.candidate_results_ = pd.DataFrame(rows)
        selected_row = self._select_candidate(self.candidate_results_)

        self.selected_candidate_ = selected_row
        self.selection_score_ = float(selected_row["score"])
        self.best_candidate_score_ = float(selected_row["best_score"])
        self.case_fraction_ = float(selected_row["case_fraction"])
        self.time_indices_ = np.asarray(selected_row["time_indices"], dtype=int)

        self.case_indices_ = self._sample_case_indices(
            y=y,
            fraction=self.case_fraction_,
            available_indices=np.arange(n_cases),
        )

        self.n_cases_selected_ = len(self.case_indices_)
        self.n_timepoints_selected_ = len(self.time_indices_)

        return self

    def fit_resample(self, X, y):
        """Fit the reducer and return the reduced training data and labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training collection.
        y : array-like of shape (n_cases,)
            Training labels.

        Returns
        -------
        X_resampled : np.ndarray
            Reduced training collection.
        y_resampled : np.ndarray
            Reduced training labels.
        """
        return self.fit(X, y).resample_train(X, y)

    def resample_train(self, X, y):
        """Apply the learned case and time reduction to training data.

        Case reduction is applied here. Use transform for test data.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training collection.
        y : array-like of shape (n_cases,)
            Training labels.

        Returns
        -------
        X_resampled : np.ndarray
            Training collection after case and time reduction.
        y_resampled : np.ndarray
            Training labels after case reduction.
        """
        check_is_fitted(self, "case_indices_")

        X = np.asarray(X)
        y = np.asarray(y)
        self._validate_transform_input(X)

        if len(y) != X.shape[0]:
            raise ValueError("X and y have inconsistent numbers of cases.")

        X_resampled = self._reduce_time_axis(X[self.case_indices_])
        y_resampled = y[self.case_indices_]

        return X_resampled, y_resampled

    def transform(self, X):
        """Apply the learned time reduction to new data.

        This method is intended for validation or test data. It does not apply
        case reduction, because test cases must not be removed.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection to transform.

        Returns
        -------
        Xt : np.ndarray
            Collection after time reduction.
        """
        check_is_fitted(self, "time_indices_")

        X = np.asarray(X)
        self._validate_transform_input(X)

        return self._reduce_time_axis(X)

    def transform_test(self, X):
        """Alias for transform, intended to make experiment scripts explicit.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Test collection.

        Returns
        -------
        Xt : np.ndarray
            Test collection after time reduction.
        """
        return self.transform(X)

    def get_reduction_summary(self):
        """Return a dictionary summarising the selected reduction.

        Returns
        -------
        summary : dict
            Summary of the selected case and time reduction.
        """
        check_is_fitted(self, "selected_candidate_")

        return {
            "candidate": self.selected_candidate_["candidate"],
            "family": self.selected_candidate_["family"],
            "case_fraction": self.case_fraction_,
            "n_cases_in": self.n_cases_in_,
            "n_cases_selected": self.n_cases_selected_,
            "n_timepoints_in": self.n_timepoints_in_,
            "n_timepoints_selected": self.n_timepoints_selected_,
            "score": self.selected_candidate_["score"],
            "best_score": self.selected_candidate_["best_score"],
            "score_is_tuning_score": True,
            "tolerance": self.tolerance,
            "time_reduction": self.time_reduction,
            "case_indices": self.case_indices_.tolist(),
            "time_indices": self.time_indices_.tolist(),
        }

    def _validate_inputs(self, X, y):
        """Validate fit inputs."""
        if X.ndim != 3:
            raise ValueError(
                "CaseTimeReducer expects X with shape "
                "(n_cases, n_channels, n_timepoints)."
            )
        if len(y) != X.shape[0]:
            raise ValueError("X and y have inconsistent numbers of cases.")
        if self.strategy not in {"auto", "case", "time", "both", "joint"}:
            raise ValueError(
                "strategy must be one of {'auto', 'case', 'time', 'both', " "'joint'}."
            )
        if self.scoring not in {"balanced_accuracy", "accuracy"}:
            raise ValueError("scoring must be 'balanced_accuracy' or 'accuracy'.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be in the interval (0, 1).")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if self.min_cases_per_class < 1:
            raise ValueError("min_cases_per_class must be at least 1.")
        if self.min_timepoints < 1:
            raise ValueError("min_timepoints must be at least 1.")
        if self.time_reduction not in {"resample", "subsample"}:
            raise ValueError("time_reduction must be either 'resample' or 'subsample'.")
        if not isinstance(self.fail_fast, bool):
            raise ValueError("fail_fast must be a boolean.")

    def _validate_transform_input(self, X):
        """Validate transform inputs."""
        if X.ndim != 3:
            raise ValueError(
                "CaseTimeReducer expects X with shape "
                "(n_cases, n_channels, n_timepoints)."
            )
        if X.shape[1] != self.n_channels_in_:
            raise ValueError(
                f"X has {X.shape[1]} channels, but reducer was fitted with "
                f"{self.n_channels_in_} channels."
            )
        if X.shape[2] != self.n_timepoints_in_:
            raise ValueError(
                f"X has {X.shape[2]} time points, but reducer was fitted with "
                f"{self.n_timepoints_in_} time points."
            )

    def _make_train_val_indices(self, y):
        """Create a training-only validation split."""
        indices = np.arange(len(y))
        _, counts = np.unique(y, return_counts=True)
        stratify = y if np.min(counts) >= 2 else None

        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=None,
            )

        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

    def _resolve_strategy(self, n_cases, n_timepoints):
        """Resolve the strategy into candidate families."""
        if self.strategy == "auto":
            return ("case",) if n_cases > n_timepoints else ("time",)
        if self.strategy == "case":
            return ("case",)
        if self.strategy == "time":
            return ("time",)
        if self.strategy == "both":
            return ("case", "time")
        if self.strategy == "joint":
            return ("joint",)
        raise RuntimeError("Unreachable strategy branch.")

    def _make_candidates(self, n_cases, n_timepoints):
        """Construct candidate reductions."""
        families = self._resolve_strategy(n_cases, n_timepoints)

        case_fractions = self._normalise_fractions(self.case_fractions, "case")
        time_specs = self._make_time_specs(n_timepoints)

        candidates = []
        seen = set()

        def add_candidate(family, case_fraction, time_indices, name):
            key = (round(float(case_fraction), 12), tuple(time_indices.tolist()))
            if key in seen:
                return
            seen.add(key)

            candidates.append(
                {
                    "family": family,
                    "candidate": name,
                    "case_fraction": float(case_fraction),
                    "time_indices": np.asarray(time_indices, dtype=int),
                    "n_timepoints": len(time_indices),
                }
            )

        full_time = np.arange(n_timepoints, dtype=int)

        if "case" in families:
            for frac in case_fractions:
                add_candidate(
                    family="case",
                    case_fraction=frac,
                    time_indices=full_time,
                    name=f"case_{frac:g}",
                )

        if "time" in families:
            for name, time_indices in time_specs:
                add_candidate(
                    family="time",
                    case_fraction=1.0,
                    time_indices=time_indices,
                    name=name,
                )

        if "joint" in families:
            for frac in case_fractions:
                for time_name, time_indices in time_specs:
                    add_candidate(
                        family="joint",
                        case_fraction=frac,
                        time_indices=time_indices,
                        name=f"case_{frac:g}_{time_name}",
                    )

        add_candidate(
            family="full",
            case_fraction=1.0,
            time_indices=full_time,
            name="full",
        )

        return candidates

    def _normalise_fractions(self, fractions, name):
        """Validate and sort fractions."""
        if fractions is None or len(fractions) == 0:
            raise ValueError(f"{name}_fractions must contain at least one value.")

        clean = []
        for frac in fractions:
            if not 0 < frac <= 1:
                raise ValueError(
                    f"All {name}_fractions must be in the interval (0, 1]."
                )
            clean.append(float(frac))

        if 1.0 not in clean:
            clean.append(1.0)

        return tuple(sorted(set(clean)))

    def _make_time_specs(self, n_timepoints):
        """Create candidate time-index specifications."""
        specs = []

        for frac in self._normalise_fractions(self.time_fractions, "time"):
            length = self._length_from_fraction(frac, n_timepoints)
            idx = self._even_time_indices(n_timepoints, length)
            specs.append((f"time_{frac:g}", idx))

        if self.time_lengths is not None:
            for length in self.time_lengths:
                if length < 1:
                    raise ValueError("All time_lengths must be positive.")
                length = min(n_timepoints, max(self.min_timepoints, int(length)))
                idx = self._even_time_indices(n_timepoints, length)
                specs.append((f"time_len_{length}", idx))

        return specs

    def _length_from_fraction(self, fraction, n_timepoints):
        """Convert a time fraction to a valid number of time points."""
        length = int(math.ceil(fraction * n_timepoints))
        return min(n_timepoints, max(self.min_timepoints, length))

    @staticmethod
    def _even_time_indices(n_timepoints, length):
        """Return evenly spaced unique time indices."""
        if length >= n_timepoints:
            return np.arange(n_timepoints, dtype=int)

        idx = np.linspace(0, n_timepoints - 1, length).round().astype(int)
        idx = np.unique(idx)

        # np.unique can shorten the array after rounding. Fill deterministically.
        if len(idx) < length:
            missing = np.setdiff1d(np.arange(n_timepoints), idx, assume_unique=True)
            needed = length - len(idx)
            idx = np.sort(np.concatenate([idx, missing[:needed]]))

        return idx.astype(int)

    def _evaluate_candidate(self, X, y, train_idx, val_idx, candidate):
        """Evaluate one candidate reduction with the proxy estimator."""
        case_fraction = candidate["case_fraction"]
        final_time_indices = candidate["time_indices"]

        proxy_train_idx = self._sample_case_indices(
            y=y,
            fraction=case_fraction,
            available_indices=train_idx,
        )

        eval_time_indices = self._make_eval_time_indices(
            family=candidate["family"],
            final_time_indices=final_time_indices,
            n_timepoints=X.shape[2],
        )

        estimator = self._make_proxy_estimator()

        start = time.perf_counter()
        error = None

        try:
            X_train = self._reduce_time_axis(
                X[proxy_train_idx],
                time_indices=eval_time_indices,
            )
            y_train = y[proxy_train_idx]
            X_val = self._reduce_time_axis(
                X[val_idx],
                time_indices=eval_time_indices,
            )
            y_val = y[val_idx]

            estimator.fit(X_train, y_train)
            fit_time = time.perf_counter() - start

            pred_start = time.perf_counter()
            y_pred = estimator.predict(X_val)
            predict_time = time.perf_counter() - pred_start

            score = self._score(y_val, y_pred)

        except Exception as exc:
            if self.fail_fast:
                raise
            fit_time = time.perf_counter() - start
            predict_time = np.nan
            score = -np.inf
            error = repr(exc)

        n_final_cases = len(
            self._sample_case_indices(
                y=y,
                fraction=case_fraction,
                available_indices=np.arange(len(y)),
            )
        )
        n_final_timepoints = len(final_time_indices)
        # IndividualTDE uses 1-NN and leave-one-out estimates, so case-related
        # work is approximately quadratic. This is a ranking proxy, not runtime.
        cost = n_final_cases**2 * n_final_timepoints

        return {
            "family": candidate["family"],
            "candidate": candidate["candidate"],
            "case_fraction": float(case_fraction),
            "n_cases_proxy_train": len(proxy_train_idx),
            "n_cases_final_train": n_final_cases,
            "n_timepoints_proxy": len(eval_time_indices),
            "n_timepoints_final": n_final_timepoints,
            "time_indices": final_time_indices.tolist(),
            "score": score,
            "fit_time": fit_time,
            "predict_time": predict_time,
            "total_time": fit_time + (0.0 if np.isnan(predict_time) else predict_time),
            "reduction_cost": cost,
            "selected": False,
            "error": error,
        }

    def _make_eval_time_indices(self, family, final_time_indices, n_timepoints):
        """Create time indices used during proxy evaluation."""
        if family == "case" and self.proxy_time_length is not None:
            if n_timepoints > self.proxy_time_length:
                length = max(self.min_timepoints, int(self.proxy_time_length))
                length = min(n_timepoints, length)
                return self._even_time_indices(n_timepoints, length)

        return np.asarray(final_time_indices, dtype=int)

    def _sample_case_indices(self, y, fraction, available_indices):
        """Sample nested case subsets using deterministic class-wise orderings."""
        available_indices = np.asarray(available_indices, dtype=int)

        if fraction >= 1:
            return np.sort(available_indices)

        rng = np.random.default_rng(self.random_state)
        selected = []

        for cls in np.unique(y[available_indices]):
            cls_idx = available_indices[y[available_indices] == cls]
            if len(cls_idx) == 0:
                continue

            n_select = int(math.ceil(fraction * len(cls_idx)))
            n_select = max(self.min_cases_per_class, n_select)
            n_select = min(len(cls_idx), n_select)

            chosen = rng.permutation(cls_idx)[:n_select]
            selected.append(chosen)

        if len(selected) == 0:
            raise RuntimeError("No cases were selected.")

        return np.sort(np.concatenate(selected).astype(int))

    def _reduce_time_axis(self, X, time_indices=None):
        """Reduce the time axis using the configured method."""
        if time_indices is None:
            time_indices = self.time_indices_
        time_indices = np.asarray(time_indices, dtype=int)

        if len(time_indices) == X.shape[2]:
            return X
        if self.time_reduction == "subsample":
            return X[:, :, time_indices]
        return resample(X, num=len(time_indices), axis=2)

    def _select_candidate(self, candidate_results):
        """Select the cheapest candidate within tolerance of the best score."""
        finite = candidate_results[np.isfinite(candidate_results["score"])].copy()

        if finite.empty:
            raise RuntimeError(
                "All proxy candidate evaluations failed. Inspect "
                "candidate_results_ for errors."
            )

        best_score = finite["score"].max()
        eligible = finite[finite["score"] >= best_score - self.tolerance].copy()

        eligible = eligible.sort_values(
            by=[
                "reduction_cost",
                "n_cases_final_train",
                "n_timepoints_final",
                "score",
                "total_time",
            ],
            ascending=[True, True, True, False, True],
        )

        selected_idx = eligible.index[0]
        self.candidate_results_.loc[selected_idx, "selected"] = True

        selected_row = self.candidate_results_.loc[selected_idx].to_dict()
        selected_row["best_score"] = best_score
        selected_row["tolerance"] = self.tolerance

        return selected_row

    def _make_proxy_estimator(self):
        """Create a proxy estimator."""
        if self.proxy_estimator is not None:
            return self._safe_clone(self.proxy_estimator)

        return self._make_default_individual_tde()

    def _make_default_individual_tde(self):
        """Create a small IndividualTDE proxy with aeon API fallbacks."""
        try:
            from aeon.classification.dictionary_based import IndividualTDE
        except ImportError:
            from aeon.classification.dictionary_based._tde import IndividualTDE

        candidate_kwargs = [
            {
                "window_size": 16,
                "word_length": 8,
                "norm": False,
                "levels": 1,
                "igb": False,
                "alphabet_size": 4,
                "bigrams": False,
                "dim_threshold": 0.85,
                "max_dims": 20,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            },
            {
                "window_size": 16,
                "word_length": 8,
                "norm": False,
                "levels": 1,
                "igb": False,
                "alphabet_size": 4,
                "bigrams": False,
                "dim_threshold": 0.85,
                "max_dims": 20,
            },
            {
                "window_size": 16,
                "word_length": 8,
                "norm": False,
                "levels": 1,
                "alphabet_size": 4,
                "bigrams": False,
            },
        ]

        last_error = None
        for kwargs in candidate_kwargs:
            try:
                return IndividualTDE(**kwargs)
            except TypeError as exc:
                last_error = exc

        raise TypeError(
            "Could not construct IndividualTDE with known parameter sets."
        ) from last_error

    @staticmethod
    def _safe_clone(estimator):
        """Clone an estimator, falling back to deepcopy."""
        try:
            return clone(estimator)
        except Exception:
            return copy.deepcopy(estimator)

    def _score(self, y_true, y_pred):
        """Score validation predictions."""
        if self.scoring == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        if self.scoring == "accuracy":
            return accuracy_score(y_true, y_pred)
        raise RuntimeError("Unreachable scoring branch.")
