"""Binary particle swarm optimisation channel selector."""

from __future__ import annotations

import numpy as np
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import train_test_split

__all__ = ["BPSO"]


class BPSO(BaseChannelSelector):
    """Channel selector based on binary particle swarm optimisation.

    Placeholder implementation for a wrapper-style channel selector that
    will use binary particle swarm optimisation to search for an
    informative subset of channels.

    Parameters
    ----------
    proportion : float, default=0.25
        Proportion of channels to retain.
    n_particles : int, default=30
        Number of particles in the swarm.
    max_iter : int, default=50
        Maximum number of optimisation iterations.
    estimator : object or None, default=None
        Estimator used to evaluate candidate channel subsets.
    inertia : float, default=0.729
        Inertia weight used in particle velocity updates.
    cognitive : float, default=1.49445
        Cognitive coefficient used in particle velocity updates.
    social : float, default=1.49445
        Social coefficient used in particle velocity updates.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Notes
    -----
    This class is currently a placeholder and is not implemented.
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
            Input time series collection.
        y : np.ndarray of shape (n_cases,), default=None
            Class labels.

        Returns
        -------
        self : BPSOChannelSelector
            Reference to self.

        Raises
        ------
        NotImplementedError
            Always, since this is a placeholder.
        """

        n_channels = X.shape[1]  
        particles = np.random.randint(2, size=(self.n_particles, n_channels))
        velocities = np.random.rand(self.n_particles, n_channels) 
        pbest_positions = particles.copy()
        pbest_scores = np.zeros(self.n_particles)  
        gbest_position = None 
        gbest_score = -float('inf')  
        for iteration in range(self.max_iter):
            for i in range( self.n_particles):

                fitness = self.fitness_function(X, y, particles[i])
                
                if fitness > pbest_scores[i]:
                    pbest_scores[i] = fitness
                    pbest_positions[i] = particles[i].copy()
                

                if fitness > gbest_score:
                    gbest_score = fitness
                    gbest_position = particles[i].copy()

            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.inertia * velocities[i]
                                + self.cognitive * r1 * (pbest_positions[i] - particles[i])
                                + self.social * r2 * (gbest_position - particles[i]))
                

                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = np.where(sigmoid > 0.5, 1, 0)  
        self.channels_selected_=list(gbest_position)
        return self

    def fitness_function(self,X, y, selected_channels):
        from aeon.classification.convolution_based import MiniRocketClassifier
        classifier = MiniRocketClassifier()
        X_selected = X[:, self.pick_channels_id(selected_channels)]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.50, random_state=23)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        num_selected_channels = sum(selected_channels)
        fitness = accuracy - 0.01 * num_selected_channels
        return fitness
    
    def pick_channels_id(self,ids):
        c = []
        for i in range(len(ids)):
            if ids[i]:
                c.append(i)
        return c



    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {},
            {
                "proportion": 0.5,
                "n_particles": 10,
                "max_iter": 5,
                "random_state": 0,
            },
        ]
