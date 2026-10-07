"""Binary particle swarm optimisation channel selector."""

from __future__ import annotations

from math import ceil

import numpy as np
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

__all__ = ["BPSO"]


class BPSO(BaseChannelSelector):
    """Select a fixed-size channel subset using binary particle swarm optimisation.

    Candidate subsets are scored on a single stratified train-validation split.
    Each particle is constrained to retain ``ceil(proportion * n_channels)``
    channels.

    Parameters
    ----------
    proportion : float, default=0.25
        Proportion of channels to retain, rounded up to the nearest channel.
    n_particles : int, default=30
        Number of particles in the swarm.
    max_iter : int, default=50
        Number of optimisation iterations.
    estimator : object or None, default=None
        Cloneable classifier used to evaluate candidate channel subsets. It must
        implement ``fit`` and ``predict`` and accept 3D time-series collections.
        If None, an ``aeon`` MiniRocketClassifier is used.
    inertia : float, default=0.729
        Inertia weight used in particle velocity updates.
    cognitive : float, default=1.49445
        Cognitive coefficient used in particle velocity updates.
    social : float, default=1.49445
        Social coefficient used in particle velocity updates.
    random_state : int or None, default=None
        Random seed used for swarm initialisation, updates, and data splitting.

    Attributes
    ----------
    channels_selected_ : list of int
        Indices of the selected channels.
    best_score_ : float
        Validation accuracy of the selected subset.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:univariate": False,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "requires_y": True,
    }

    def __init__(
        self,
        proportion: float = 0.25,
        n_particles: int = 30,
        max_iter: int = 50,
        estimator=None,
        inertia: float = 0.729,
        cognitive: float = 1.49445,
        social: float = 1.49445,
        random_state: int | None = None,
    ):
        self.proportion = proportion
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.estimator = estimator
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the channel selector.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training time series.
        y : np.ndarray of shape (n_cases,)
            Class labels. Each class must contain enough cases for a stratified
            50 percent validation split.

        Returns
        -------
        self
            Fitted channel selector.
        """
        self._validate_parameters()
        rng = check_random_state(self.random_state)
        n_channels = X.shape[1]
        n_selected = ceil(self.proportion * n_channels)

        train_indices, validation_indices = train_test_split(
            np.arange(X.shape[0]),
            test_size=0.5,
            random_state=self.random_state,
            stratify=y,
        )
        particles = self._initialise_particles(
            rng, self.n_particles, n_channels, n_selected
        )
        velocities = rng.uniform(-1.0, 1.0, size=particles.shape)
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.n_particles, -np.inf)
        global_best_position = particles[0].copy()
        global_best_score = -np.inf
        score_cache = {}

        for _ in range(self.max_iter):
            for particle_index, particle in enumerate(particles):
                key = tuple(particle)
                if key not in score_cache:
                    score_cache[key] = self._score_particle(
                        X,
                        y,
                        particle,
                        train_indices,
                        validation_indices,
                    )
                score = score_cache[key]

                if score > personal_best_scores[particle_index]:
                    personal_best_scores[particle_index] = score
                    personal_best_positions[particle_index] = particle.copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.copy()

            random_cognitive = rng.random_sample(particles.shape)
            random_social = rng.random_sample(particles.shape)
            velocities = (
                self.inertia * velocities
                + self.cognitive
                * random_cognitive
                * (personal_best_positions - particles)
                + self.social * random_social * (global_best_position - particles)
            )
            probabilities = self._sigmoid(velocities)
            particles = self._sample_particles(probabilities, rng, n_selected)

        self.channels_selected_ = np.flatnonzero(global_best_position).tolist()
        self.best_score_ = float(global_best_score)
        return self

    def _score_particle(
        self,
        X,
        y,
        particle,
        train_indices,
        validation_indices,
    ):
        """Return the validation accuracy for one candidate subset.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Training time series.
        y : np.ndarray of shape (n_cases,)
            Class labels.
        particle : np.ndarray of shape (n_channels,)
            Binary channel-selection mask.
        train_indices : np.ndarray of shape (n_train_cases,)
            Indices used to fit the candidate estimator.
        validation_indices : np.ndarray of shape (n_validation_cases,)
            Indices used to score the candidate estimator.

        Returns
        -------
        float
            Classification accuracy on the validation cases.
        """
        estimator = clone(self._get_estimator())
        selected = np.flatnonzero(particle)
        estimator.fit(X[train_indices][:, selected], y[train_indices])
        predictions = estimator.predict(X[validation_indices][:, selected])
        return float(accuracy_score(y[validation_indices], predictions))

    def _get_estimator(self):
        """Return the configured estimator or construct the default.

        Returns
        -------
        estimator
            Unfitted classifier used to evaluate channel subsets.
        """
        if self.estimator is not None:
            return self.estimator

        from aeon.classification.convolution_based import MiniRocketClassifier

        return MiniRocketClassifier(random_state=self.random_state)

    def _validate_parameters(self):
        """Validate constructor parameters.

        Raises
        ------
        ValueError
            If the proportion, swarm size, iteration count, or particle update
            coefficients are outside their supported ranges.
        """
        if not 0 < self.proportion <= 1:
            raise ValueError("proportion must be in the interval (0, 1].")
        if not isinstance(self.n_particles, int) or self.n_particles < 1:
            raise ValueError("n_particles must be a positive integer.")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if self.inertia < 0 or self.cognitive < 0 or self.social < 0:
            raise ValueError(
                "inertia, cognitive, and social must be greater than or equal to 0."
            )

    @staticmethod
    def _initialise_particles(rng, n_particles, n_channels, n_selected):
        """Create random particles with the required number of channels.

        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator.
        n_particles : int
            Number of particles to create.
        n_channels : int
            Number of binary positions in each particle.
        n_selected : int
            Number of positions set to one in each particle.

        Returns
        -------
        np.ndarray of shape (n_particles, n_channels)
            Initial binary particle positions.
        """
        particles = np.zeros((n_particles, n_channels), dtype=np.int8)
        for particle in particles:
            particle[rng.choice(n_channels, size=n_selected, replace=False)] = 1
        return particles

    @staticmethod
    def _sample_particles(probabilities, rng, n_selected):
        """Sample binary positions and enforce the required subset size.

        Positions are first sampled independently from their probabilities.
        Excess selected positions with the lowest probabilities are cleared, or
        unselected positions with the highest probabilities are added, until
        each particle contains exactly ``n_selected`` channels.

        Parameters
        ----------
        probabilities : np.ndarray of shape (n_particles, n_channels)
            Probability of selecting each channel in each particle.
        rng : np.random.RandomState
            Random number generator.
        n_selected : int
            Required number of selected channels per particle.

        Returns
        -------
        np.ndarray of shape (n_particles, n_channels)
            Sampled binary particle positions.
        """
        particles = (rng.random_sample(probabilities.shape) < probabilities).astype(
            np.int8
        )
        for particle, particle_probabilities in zip(particles, probabilities):
            selected = np.flatnonzero(particle)
            if len(selected) > n_selected:
                n_to_clear = len(selected) - n_selected
                to_clear = selected[
                    np.argsort(particle_probabilities[selected])[:n_to_clear]
                ]
                particle[to_clear] = 0
            elif len(selected) < n_selected:
                unselected = np.flatnonzero(particle == 0)
                to_set = unselected[
                    np.argsort(particle_probabilities[unselected])[
                        -(n_selected - len(selected)) :
                    ]
                ]
                particle[to_set] = 1
        return particles

    @staticmethod
    def _sigmoid(values):
        """Apply a numerically stable sigmoid transform.

        Parameters
        ----------
        values : np.ndarray
            Particle velocities.

        Returns
        -------
        np.ndarray
            Selection probabilities with the same shape as ``values``.
        """
        return 1.0 / (1.0 + np.exp(-np.clip(values, -709, 709)))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a lightweight parameter set for estimator tests.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set. Currently ignored.

        Returns
        -------
        dict
            Parameters that avoid fitting the default MiniRocketClassifier.
        """
        from aeon.classification import DummyClassifier

        return {
            "proportion": 0.5,
            "n_particles": 2,
            "max_iter": 1,
            "estimator": DummyClassifier(),
            "random_state": 0,
        }
