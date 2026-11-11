"""Tests for the RiemannianSpatialFilter transformer."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from scipy.linalg import eigh
from scipy.optimize import minimize

from aeon_neuro.transformations.collection.channel_creation import (
    RiemannianSpatialFilter,
)

N_CHANNELS = 6
N_TIMEPOINTS = 100


def _data(n_classes=2, n_per_class=20):
    """Return data where each class has extra power on a different channel."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(n_classes), n_per_class)
    X = rng.normal(size=(len(y), N_CHANNELS, N_TIMEPOINTS))
    for c in range(n_classes):
        X[y == c, c] *= 3
    return X, y


def _class_means(X, y):
    """Return the Riemannian mean covariance of each of the two classes."""
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_covariance

    X_cov = Covariances(estimator="scm").transform(X)
    return (
        mean_covariance(X_cov[y == 0], metric="riemann"),
        mean_covariance(X_cov[y == 1], metric="riemann"),
    )


def _squared_distance(W, mean_a, mean_b):
    """Return the squared Riemannian distance between the filtered means.

    This is the objective maximised over W by Pan et al. (2025), written
    directly from its definition rather than through
    ``RiemannianSpatialFilter``.
    """
    evals = eigh(W.T @ mean_a @ W, W.T @ mean_b @ W, eigvals_only=True)
    return np.sum(np.log(evals) ** 2)


@pytest.mark.parametrize("n_components", [1, 2, 4, N_CHANNELS])
def test_rsf_returns_spatially_filtered_collection(n_components):
    """Test output is (n_cases, n_components, n_timepoints) and uses filters_."""
    X, y = _data()
    rsf = RiemannianSpatialFilter(n_components=n_components)
    Xt = rsf.fit_transform(X, y)

    assert Xt.shape == (len(y), n_components, N_TIMEPOINTS)
    assert Xt.dtype == np.float64
    assert rsf.filters_.shape == (N_CHANNELS, N_CHANNELS)
    assert rsf.evals_.shape == (N_CHANNELS,)
    expected = np.einsum("kc,nct->nkt", rsf.filters_[:n_components], X)
    np.testing.assert_allclose(Xt, expected)


def test_rsf_transform_matches_fit_transform():
    """Test transform after fit gives the same output as fit_transform."""
    X, y = _data()

    rsf = RiemannianSpatialFilter(n_components=2)
    Xt1 = rsf.fit_transform(X, y)
    Xt2 = rsf.transform(X)

    np.testing.assert_allclose(Xt1, Xt2)


def test_rsf_solves_generalised_eigenproblem():
    """Test filters satisfy M_a w = lambda M_b w, ordered by |log(lambda)|."""
    X, y = _data()
    rsf = RiemannianSpatialFilter().fit(X, y)

    mean_a, mean_b = _class_means(X, y)
    for w, lam in zip(rsf.filters_, rsf.evals_):
        np.testing.assert_allclose(mean_a @ w, lam * mean_b @ w, atol=1e-10)
    assert np.all(np.diff(np.abs(np.log(rsf.evals_))) <= 1e-12)
    # the most discriminative filters pick out the channels with extra power
    assert set(np.abs(rsf.filters_[:2]).argmax(axis=1)) == {0, 1}


@pytest.mark.parametrize("n_components", [1, 2, 4, N_CHANNELS])
def test_rsf_distance_between_filtered_means(n_components):
    """Test the filtered class means are sum(log(evals) ** 2) apart."""
    from pyriemann.utils.distance import distance_riemann

    X, y = _data()
    rsf = RiemannianSpatialFilter(n_components=n_components).fit(X, y)

    mean_a, mean_b = _class_means(X, y)
    W = rsf.filters_[:n_components].T
    expected = np.sum(np.log(rsf.evals_[:n_components]) ** 2)

    np.testing.assert_allclose(_squared_distance(W, mean_a, mean_b), expected)
    np.testing.assert_allclose(
        distance_riemann(W.T @ mean_a @ W, W.T @ mean_b @ W) ** 2, expected
    )
    # keeping every filter loses nothing of the distance between the means
    if n_components == N_CHANNELS:
        np.testing.assert_allclose(distance_riemann(mean_a, mean_b) ** 2, expected)


@pytest.mark.parametrize("n_components", [1, 2, 4])
def test_rsf_matches_numerical_optimisation(n_components):
    """Test the closed form reaches the optimum found by numerical optimisation.

    The reference method of Pan et al. (2025) finds W by numerically maximising
    the squared Riemannian distance between the filtered class means, whereas
    ``RiemannianSpatialFilter`` uses the closed form solution. The objective is
    unchanged by any invertible mixing of the columns of W, so the two versions
    are compared on the objective value they reach rather than on W itself.
    """
    X, y = _data()
    rsf = RiemannianSpatialFilter(n_components=n_components).fit(X, y)

    mean_a, mean_b = _class_means(X, y)
    closed_form = _squared_distance(rsf.filters_[:n_components].T, mean_a, mean_b)

    def negative_distance(w_flat):
        W = w_flat.reshape(N_CHANNELS, n_components)
        return -_squared_distance(W, mean_a, mean_b)

    optimised = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        W0 = np.linalg.qr(rng.normal(size=(N_CHANNELS, n_components)))[0]
        optimised.append(-minimize(negative_distance, W0.ravel(), method="BFGS").fun)

    # no starting point beats the closed form, and the best one reaches it
    assert max(optimised) <= closed_form * (1 + 1e-6)
    np.testing.assert_allclose(max(optimised), closed_form, rtol=1e-3)


def test_rsf_is_invariant_to_class_order():
    """Test swapping the class labels inverts evals_ but keeps the filters."""
    X, y = _data()
    rsf = RiemannianSpatialFilter().fit(X, y)
    swapped = RiemannianSpatialFilter().fit(X, 1 - y)

    np.testing.assert_allclose(swapped.evals_, 1 / rsf.evals_)
    # each filter is the same direction, up to sign and scale
    for w, w_swapped in zip(rsf.filters_, swapped.filters_):
        cosine = w @ w_swapped / (np.linalg.norm(w) * np.linalg.norm(w_swapped))
        np.testing.assert_allclose(abs(cosine), 1)


def test_rsf_accepts_non_integer_labels():
    """Test string labels give the same filters as integer labels."""
    X, y = _data()
    rsf = RiemannianSpatialFilter().fit(X, y)
    rsf_str = RiemannianSpatialFilter().fit(X, np.where(y == 0, "left", "right"))

    np.testing.assert_array_equal(rsf_str.classes_, ["left", "right"])
    np.testing.assert_allclose(rsf_str.filters_, rsf.filters_)


def test_rsf_single_class_raises():
    """Test fit with only one class in y raises a ValueError."""
    X, y = _data()

    with pytest.raises(ValueError, match="only one class"):
        RiemannianSpatialFilter().fit(X, np.zeros_like(y))


@pytest.mark.parametrize("n_classes", [3, 4])
def test_rsf_more_than_two_classes_raises(n_classes):
    """Test fit with more than two classes in y raises a ValueError."""
    X, y = _data(n_classes)

    with pytest.raises(ValueError, match="only supports binary classification"):
        RiemannianSpatialFilter().fit(X, y)


@pytest.mark.parametrize("n_components", [0, -1, N_CHANNELS + 1])
def test_rsf_n_components_out_of_range_raises(n_components):
    """Test n_components outside 1 to n_channels raises a ValueError."""
    X, y = _data()

    with pytest.raises(ValueError, match="n_components must be between"):
        RiemannianSpatialFilter(n_components=n_components).fit(X, y)


@pytest.mark.parametrize("n_components", [2.0, "2", None, True])
def test_rsf_n_components_not_integer_raises(n_components):
    """Test a non-integer n_components raises a ValueError."""
    X, y = _data()

    with pytest.raises(ValueError, match="n_components must be an integer"):
        RiemannianSpatialFilter(n_components=n_components).fit(X, y)


def test_rsf_singular_covariance_raises():
    """Test fewer time points than channels raises unless shrinkage is used."""
    X, y = _data()
    X = X[:, :, : N_CHANNELS - 2]

    with pytest.raises(ValueError, match="positive definite"):
        RiemannianSpatialFilter().fit(X, y)

    Xt = RiemannianSpatialFilter(covariance_estimator="lwf").fit_transform(X, y)
    assert Xt.shape == (len(y), 4, N_CHANNELS - 2)
    assert np.all(np.isfinite(Xt))


def test_rsf_requires_y():
    """Test fit without y raises a ValueError."""
    X, _ = make_example_3d_numpy(
        n_cases=20, n_channels=4, n_timepoints=30, random_state=0
    )

    with pytest.raises(ValueError, match="requires_y"):
        RiemannianSpatialFilter().fit(X)


@pytest.mark.parametrize("init", ["random", "csp"])
def test_rsf_numerical_solver_matches_closed_form(init):
    """Test the numerical solver recovers the closed form filters."""
    X, y = _data()
    closed_form = RiemannianSpatialFilter(n_components=2).fit(X, y)
    rsf = RiemannianSpatialFilter(
        n_components=2, solver="trust-constr", init=init, random_state=0
    )
    Xt = rsf.fit_transform(X, y)

    assert Xt.shape == (len(y), 2, N_TIMEPOINTS)
    assert rsf.filters_.shape == (2, N_CHANNELS)
    assert rsf.evals_.shape == (2,)
    assert np.all(np.diff(np.abs(np.log(rsf.evals_))) <= 1e-12)
    np.testing.assert_allclose(rsf.evals_, closed_form.evals_[:2], rtol=1e-3)
    # each filter is the same direction as the closed form one, up to sign
    for w, w_closed in zip(rsf.filters_, closed_form.filters_):
        cosine = w @ w_closed / (np.linalg.norm(w) * np.linalg.norm(w_closed))
        np.testing.assert_allclose(abs(cosine), 1, atol=1e-3)


@pytest.mark.parametrize("n_components", [1, 2, 4])
@pytest.mark.parametrize("init", ["random", "csp"])
def test_rsf_numerical_solver_never_beats_closed_form(n_components, init):
    """Test the optimised distance is at most the closed form distance."""
    X, y = _data()
    closed_form = RiemannianSpatialFilter(n_components=n_components).fit(X, y)
    rsf = RiemannianSpatialFilter(
        n_components=n_components, solver="trust-constr", init=init, random_state=0
    ).fit(X, y)

    mean_a, mean_b = _class_means(X, y)
    optimised = _squared_distance(rsf.filters_.T, mean_a, mean_b)
    best = np.sum(np.log(closed_form.evals_[:n_components]) ** 2)

    np.testing.assert_allclose(optimised, np.sum(np.log(rsf.evals_) ** 2))
    assert optimised <= best * (1 + 1e-6)


def test_rsf_csp_init_is_never_worse_than_csp():
    """Test filters started from CSP separate the means at least as well as CSP."""
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP

    X, y = _data()
    mean_a, mean_b = _class_means(X, y)
    X_cov = Covariances(estimator="scm").transform(X)
    W_csp = CSP(nfilter=2, metric="euclid").fit(X_cov, y).filters_.T

    # even a single iteration must not end below its starting point
    rsf = RiemannianSpatialFilter(
        n_components=2, solver="trust-constr", init="csp", max_iter=1
    ).fit(X, y)

    assert _squared_distance(rsf.filters_.T, mean_a, mean_b) >= _squared_distance(
        W_csp, mean_a, mean_b
    ) * (1 - 1e-10)


def test_rsf_random_state_makes_random_init_reproducible():
    """Test the same random_state gives the same filters."""
    X, y = _data()
    params = {"n_components": 2, "solver": "trust-constr", "max_iter": 3}

    rsf1 = RiemannianSpatialFilter(random_state=0, **params).fit(X, y)
    rsf2 = RiemannianSpatialFilter(random_state=0, **params).fit(X, y)
    rsf3 = RiemannianSpatialFilter(random_state=1, **params).fit(X, y)

    np.testing.assert_array_equal(rsf1.filters_, rsf2.filters_)
    assert not np.allclose(rsf1.filters_, rsf3.filters_)


def test_rsf_invalid_solver_raises():
    """Test an unknown solver raises a ValueError."""
    X, y = _data()

    with pytest.raises(ValueError, match="solver must be one of"):
        RiemannianSpatialFilter(solver="nelder-mead").fit(X, y)


def test_rsf_invalid_init_raises():
    """Test an unknown init raises a ValueError for the numerical solver."""
    X, y = _data()

    with pytest.raises(ValueError, match="init must be one of"):
        RiemannianSpatialFilter(solver="trust-constr", init="identity").fit(X, y)


@pytest.mark.parametrize("max_iter", [0, -5, 10.0, None, True])
def test_rsf_invalid_max_iter_raises(max_iter):
    """Test a max_iter that is not a positive integer raises a ValueError."""
    X, y = _data()

    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        RiemannianSpatialFilter(solver="trust-constr", max_iter=max_iter).fit(X, y)


# Results of the authors' implementation of Pan et al. (2025) (``RSF`` in ``rsf.py`` of
# https://github.com/PLC-TJU/RSF, rsf.py at commit dc033a6), run once offline on
# ``_data()`` with ``RSF(dim=n_components, method="default")`` after
# ``np.random.seed(0)``. Its code cannot be imported here because it is not
# packaged and its licence is non-commercial, so only these numbers are stored.
# AUTHORS_DISTANCE is the squared Riemannian distance between the filtered
# Ledoit-Wolf class means, and AUTHORS_FILTERS the filters, one per column.
# Generated with pyriemann 0.12, scikit-learn 1.9.1, numpy 2.3.5, scipy 1.17.1.
AUTHORS_DISTANCE = {1: 4.4674390710, 2: 8.8568363057, 4: 8.8689566237}
AUTHORS_FILTERS = {
    1: np.array(
        [
            [-2.2149029083],
            [0.0048258971],
            [-0.0219526878],
            [-0.02245684],
            [0.0251855116],
            [0.1131464767],
        ]
    ),
    2: np.array(
        [
            [-1.2629323526, 0.6124330367],
            [-0.1708881215, -1.2754312089],
            [-0.0189834135, -0.041374877],
            [-0.0157871388, -0.015673797],
            [0.0159566619, 0.0047465399],
            [0.0714933714, 0.0199128585],
        ]
    ),
    4: np.array(
        [
            [-0.604443451, 1.1953290886, -0.3292188636, 0.2617466531],
            [-0.4042510132, -0.7194151062, -0.6459173953, 0.2392469366],
            [-0.5791184175, -0.1683883932, 0.6585372487, 0.056902532],
            [0.1936438443, -0.0024176982, 0.0556425145, 0.534768635],
            [-0.6825502527, -0.0747266635, 0.1870504308, -1.08248755],
            [0.7333985877, 0.1868768302, -0.9669233679, -0.3582363111],
        ]
    ),
}


@pytest.mark.parametrize("n_components", [1, 2, 4])
@pytest.mark.parametrize(
    "params",
    [
        {"solver": "eig"},
        {"solver": "trust-constr", "init": "random", "random_state": 0},
        {"solver": "trust-constr", "init": "csp"},
    ],
)
def test_rsf_matches_authors_implementation(n_components, params):
    """Test RSF reaches the authors' distance and filter subspace.

    The distance is unchanged by any invertible mixing of the filters, so the
    filters are compared by the largest principal angle between the subspaces
    they span, rather than entry by entry.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_covariance
    from scipy.linalg import subspace_angles

    X, y = _data()
    rsf = RiemannianSpatialFilter(
        n_components=n_components, covariance_estimator="lwf", **params
    ).fit(X, y)
    W = rsf.filters_[:n_components].T

    X_cov = Covariances(estimator="lwf").transform(X)
    mean_a = mean_covariance(X_cov[y == 0], metric="riemann")
    mean_b = mean_covariance(X_cov[y == 1], metric="riemann")

    np.testing.assert_allclose(
        _squared_distance(W, mean_a, mean_b),
        AUTHORS_DISTANCE[n_components],
        rtol=1e-7,
    )
    # With 4 components the weakest filter has an eigenvalue close to 1, so it
    # barely changes the distance and the optimisers (the authors' included)
    # stop at slightly different directions depending on the platform's
    # floating point. 0.1 degrees still rules out a different filter subspace.
    angles = subspace_angles(W, AUTHORS_FILTERS[n_components])
    assert np.degrees(angles.max()) < 0.1
