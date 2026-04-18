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
    can be specified either by name or as a callable.

    Built-in scoring functions are:

    - ``"variance"``: ranks channels by variance over all cases and time points.
    - ``"class_mean"``: ranks channels by the sum of pairwise Euclidean
      distances between class mean series.
    - ``"mutual_information"``: ranks channels by mutual information between
      class labels and summary features extracted from each channel.

    Parameters
    ----------
    score_channel : str or callable, default="variance"
        Scoring function used to rank channels. If a string, must be one of
        ``"variance"``, ``"class_mean"``, or ``"mutual_information"``.
        If a callable, it must have signature ``score_channel(X, y=None, **kwargs)``
        and return a float, where `X` is a single-channel collection of shape
        ``(n_cases, 1, n_timepoints)``.
    proportion : float, default=0.4
        Proportion of channels to keep, rounded up to the nearest integer.
    score_params : dict or None, default=None
        Optional keyword arguments passed to the scoring function. For the
        built-in mutual information scorer, supported values include:

        - ``summary`` : {"mean", "mean_std", "mean_std_energy"}, default="mean_std"
          Summary features extracted from each channel before mutual information
          is computed.
        - ``random_state`` : int or None, default=None
          Random state passed to ``sklearn.feature_selection.mutual_info_classif``.

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
    ):
        self.score_channel = score_channel
        self.proportion = proportion
        self.score_params = score_params
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

        sorted_indices = np.argsort(-scores)
        n_keep = math.ceil(n_channels * self.proportion)
        self.channels_selected_ = sorted_indices[:n_keep]

        return self

    def _resolve_scorer(self) -> Callable:
        """Resolve the channel scoring function."""
        if callable(self.score_channel):
            return self.score_channel

        if self.score_channel == "variance":
            return self._variance_score
        elif self.score_channel == "class_mean":
            return self._class_mean_score
        elif self.score_channel == "mutual_information":
            return self._mutual_information_score
        else:
            raise ValueError(
                "score_channel must be a callable or one of "
                "{'variance', 'class_mean', 'mutual_information'}."
            )

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
        X: np.ndarray, summary: str = "mean_std"
    ) -> np.ndarray:
        """Extract summary features for a single channel."""
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        energy = np.sum(X**2, axis=1)

        if summary == "mean":
            return mean[:, np.newaxis]
        elif summary == "mean_std":
            return np.column_stack((mean, std))
        elif summary == "mean_std_energy":
            return np.column_stack((mean, std, energy))
        else:
            raise ValueError(
                "summary must be one of {'mean', 'mean_std', 'mean_std_energy'}."
            )

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
        }
