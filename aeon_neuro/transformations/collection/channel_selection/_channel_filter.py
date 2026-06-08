"""Filter-based channel selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Dict as TypingDict
from typing import List as TypingList

import numpy as np
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector
from sklearn.feature_selection import mutual_info_classif

__all__ = ["ChannelFilter"]


class ChannelFilter(BaseChannelSelector):
    """Select channels using a cheap filter score.

    ChannelFilter ranks channels independently using a scoring function and
    selects a proportion of the highest-scoring channels. The scoring function
    can be specified either by name or as a callable. An optional redundancy
    pruning stage can be applied after ranking.

    Built-in scoring functions are:

    - ``"variance"``: rank channels by variance over all cases and time points.
    - ``"class_mean"``: rank channels by the sum of pairwise Euclidean
      distances between class mean series.
    - ``"mutual_information"``: rank channels by mutual information between
      class labels and summary features extracted from each channel.

    Parameters
    ----------
    score_channel : str or callable, default="variance"
        Scoring function used to rank channels. If a string, must be one of
        ``"variance"``, ``"class_mean"``, or ``"mutual_information"``.
        If a callable, it must have signature
        ``score_channel(X, y=None, **kwargs)`` and return a float, where `X`
        is a single-channel collection of shape
        ``(n_cases, 1, n_timepoints)``.
    proportion : float, default=0.4
        Proportion of channels to keep, rounded up to the nearest integer.
    score_params : dict or None, default=None
        Optional keyword arguments passed to the scoring function. For the
        built-in mutual information scorer, supported values include:

        - ``summary`` : {"mean", "mean_std", "mean_std_energy"},
          default="mean_std"
          Summary features extracted from each channel before mutual
          information is computed.
        - ``random_state`` : int or None, default=None
          Random state passed to ``sklearn.feature_selection.mutual_info_classif``.
    redundancy_method : str or None, default=None
        Optional redundancy pruning method applied after relevance ranking.
        If None, no redundancy pruning is used. Current supported option is
        ``"correlation"``.
    redundancy_threshold : float, default=0.95
        Threshold used by the redundancy pruning method. For
        ``redundancy_method="correlation"``, a candidate channel is excluded
        if its absolute Pearson correlation with any already selected channel
        exceeds this threshold.
    redundancy_params : dict or None, default=None
        Optional keyword arguments passed to the redundancy pruning method.

    Attributes
    ----------
    channels_selected_ : np.ndarray of shape (n_selected_channels,)
        Indices of the selected channels.
    scores_ : np.ndarray of shape (n_channels,)
        Score for each input channel.

    Notes
    -----
    This class is intended for cheap filter methods that score each channel
    independently. Higher scores are always assumed to be better.

    Input to the scoring function is a univariate collection of shape
    ``(n_cases, 1, n_timepoints)``.
    """

    _tags = {
        "requires_y": False,
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        score_channel: str | Callable = "variance",
        proportion: float = 0.4,
        score_params: dict | None = None,
        redundancy_method: str | None = None,
        redundancy_threshold: float = 0.95,
        redundancy_params: dict | None = None,
    ):
        self.score_channel = score_channel
        self.proportion = proportion
        self.score_params = score_params
        self.redundancy_method = redundancy_method
        self.redundancy_threshold = redundancy_threshold
        self.redundancy_params = redundancy_params
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray | TypingList | None = None):
        """Fit the channel selector.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training data.
        y : array-like or list or None, default=None
            Target values for X. Required only for supervised scoring methods.

        Returns
        -------
        self :
            Reference to self.
        """
        if self.proportion <= 0 or self.proportion > 1:
            raise ValueError("proportion must be in the range (0, 1].")

        scorer = self._resolve_scorer()
        score_params = {} if self.score_params is None else self.score_params

        n_channels = X.shape[1]
        scores = np.zeros(n_channels, dtype=float)

        for i in range(n_channels):
            Xi = X[:, i : i + 1, :]
            scores[i] = scorer(Xi, y=y, **score_params)

        self.scores_ = scores
        ranked_channels = np.argsort(-scores)
        n_keep = math.ceil(n_channels * self.proportion)

        if self.redundancy_method is None:
            self.channels_selected_ = ranked_channels[:n_keep]
        else:
            self.channels_selected_ = self._apply_redundancy_pruning(
                X=X,
                ranked_channels=ranked_channels,
                n_keep=n_keep,
            )

        return self

    def _resolve_scorer(self) -> Callable:
        """Resolve the channel scoring function."""
        if callable(self.score_channel):
            return self.score_channel

        if self.score_channel == "variance":
            return self._variance_score
        if self.score_channel == "class_mean":
            return self._class_mean_score
        if self.score_channel == "mutual_information":
            return self._mutual_information_score

        raise ValueError(
            "score_channel must be a callable or one of "
            "{'variance', 'class_mean', 'mutual_information'}."
        )

    def _apply_redundancy_pruning(
        self,
        X: np.ndarray,
        ranked_channels: np.ndarray,
        n_keep: int,
    ) -> np.ndarray:
        """Apply optional redundancy pruning to ranked channels."""
        if self.redundancy_method == "correlation":
            redundancy_params = (
                {} if self.redundancy_params is None else self.redundancy_params
            )
            return self._correlation_pruning(
                X=X,
                ranked_channels=ranked_channels,
                n_keep=n_keep,
                threshold=self.redundancy_threshold,
                **redundancy_params,
            )

        raise ValueError("redundancy_method must be None or one of {'correlation'}.")

    @staticmethod
    def _variance_score(X: np.ndarray, y=None, **kwargs) -> float:
        """Score a channel by total variance."""
        return float(np.var(X))

    @staticmethod
    def _class_mean_score(X: np.ndarray, y=None, **kwargs) -> float:
        """Score a channel by pairwise class-mean separation."""
        if y is None:
            raise ValueError("y is required for score_channel='class_mean'.")

        Xi = X[:, 0, :]
        classes = np.unique(y)
        class_means = [Xi[y == c].mean(axis=0) for c in classes]

        score = 0.0
        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                score += np.linalg.norm(class_means[i] - class_means[j])

        return float(score)

    @staticmethod
    def _mutual_information_score(
        X: np.ndarray,
        y=None,
        summary: str = "mean_std",
        random_state: int | None = None,
        **kwargs,
    ) -> float:
        """Score a channel by mutual information with class labels."""
        if y is None:
            raise ValueError("y is required for score_channel='mutual_information'.")

        Xi = X[:, 0, :]
        features = ChannelFilter._channel_summary_features(Xi, summary=summary)
        mi = mutual_info_classif(features, y, random_state=random_state)
        return float(np.sum(mi))

    @staticmethod
    def _channel_summary_features(
        X: np.ndarray,
        summary: str = "mean_std",
    ) -> np.ndarray:
        """Extract summary features for a single channel."""
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        energy = np.sum(X**2, axis=1)

        if summary == "mean":
            return mean[:, np.newaxis]
        if summary == "mean_std":
            return np.column_stack((mean, std))
        if summary == "mean_std_energy":
            return np.column_stack((mean, std, energy))

        raise ValueError(
            "summary must be one of {'mean', 'mean_std', 'mean_std_energy'}."
        )

    @staticmethod
    def _correlation_pruning(
        X: np.ndarray,
        ranked_channels: np.ndarray,
        n_keep: int,
        threshold: float = 0.95,
        method: str = "pearson",
        fallback_fill: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Greedily prune channels that are highly correlated.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training data.
        ranked_channels : np.ndarray of shape (n_channels,)
            Channel indices ranked from most to least relevant.
        n_keep : int
            Number of channels to retain.
        threshold : float, default=0.95
            Absolute correlation threshold above which a candidate channel is
            treated as redundant.
        method : str, default="pearson"
            Correlation method. Currently only ``"pearson"`` is supported.
        fallback_fill : bool, default=True
            If True and fewer than `n_keep` channels survive pruning, fill the
            remaining slots using the original ranking order.

        Returns
        -------
        np.ndarray of shape (n_selected_channels,)
            Selected channel indices.
        """
        if method != "pearson":
            raise ValueError("Only method='pearson' is currently supported.")

        if threshold < 0 or threshold > 1:
            raise ValueError("redundancy_threshold must be in the range [0, 1].")

        selected = []
        flattened = X.reshape(X.shape[0], X.shape[1], -1)
        channel_vectors = flattened.transpose(1, 0, 2).reshape(X.shape[1], -1)

        for candidate in ranked_channels:
            if len(selected) == 0:
                selected.append(candidate)
            else:
                keep = True
                candidate_vec = channel_vectors[candidate]

                for chosen in selected:
                    chosen_vec = channel_vectors[chosen]
                    corr = ChannelFilter._safe_abs_pearson_corr(
                        candidate_vec,
                        chosen_vec,
                    )
                    if corr > threshold:
                        keep = False
                        break

                if keep:
                    selected.append(candidate)

            if len(selected) == n_keep:
                break

        if fallback_fill and len(selected) < n_keep:
            for candidate in ranked_channels:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == n_keep:
                    break

        return np.asarray(selected, dtype=int)

    @staticmethod
    def _safe_abs_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
        """Compute absolute Pearson correlation with constant-vector handling."""
        x_std = np.std(x)
        y_std = np.std(y)

        if x_std == 0 or y_std == 0:
            return 0.0

        return float(abs(np.corrcoef(x, y)[0, 1]))

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> TypingDict[str, any]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        dict
            Dictionary of testing parameters.
        """
        return {
            "score_channel": "variance",
            "proportion": 0.4,
            "redundancy_method": "correlation",
            "redundancy_threshold": 0.95,
        }
