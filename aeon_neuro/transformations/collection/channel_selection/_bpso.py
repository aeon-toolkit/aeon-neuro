"""Binary particle swarm optimisation channel selector."""

from __future__ import annotations

from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

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
        raise NotImplementedError("BPSOChannelSelector is not implemented yet.")

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
