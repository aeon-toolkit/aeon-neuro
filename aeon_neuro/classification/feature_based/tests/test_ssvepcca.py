"""Tests for the SSVEP CCA classifier."""

import numpy as np
import pytest

from aeon_neuro.classification.feature_based import SSVEPCCAClassifier

sfreq, n_channels, n_timepoints = 250, 4, 500
freqs = {"a": 8, "b": 11, "c": 15, "d": 23}
t = np.arange(n_timepoints) / sfreq


def _make_ssvep_data(n_cases=80, random_state=0):
    """Noisy multichannel data where each case oscillates at its class freq."""
    rng = np.random.default_rng(random_state)
    labels = rng.choice(list(freqs), size=n_cases)
    X = rng.standard_normal((n_cases, n_channels, n_timepoints))
    for i, label in enumerate(labels):
        phase = rng.uniform(0, 2 * np.pi)
        weights = rng.standard_normal(n_channels)
        X[i] += weights[:, None] * np.sin(2 * np.pi * freqs[label] * t + phase)
    return X, labels


# MOABB SSVEP_CCA(n_harmonics=3, freq_map=freqs) on _make_ssvep_data(n_cases=20),
# generated offline by Claude (Anthropic) with moabb 1.8.0 (commit 8901391), which
# uses the QR closed form. Rows are cases, columns are classes "a", "b", "c", "d".
MOABB_PREDICTIONS = list("dccbbaaaadcdccdccccd")
MOABB_SCORES = np.array(
    [
        [0.2140858282, 0.1330024670, 0.1336409743, 0.6905267759],
        [0.1622697010, 0.1251676185, 0.8520074602, 0.1044712017],
        [0.1501000214, 0.1620732909, 0.8567823842, 0.1377154938],
        [0.1364216011, 0.7865842508, 0.1260511446, 0.1669863376],
        [0.1740210211, 0.7813370179, 0.1406678379, 0.1373528586],
        [0.7938491018, 0.1472796142, 0.1540242621, 0.1368351461],
        [0.9006862781, 0.1582116715, 0.1773500858, 0.1664124585],
        [0.8101960770, 0.1533229198, 0.1614916558, 0.1292644759],
        [0.8740318468, 0.1204086160, 0.1324830742, 0.1559195819],
        [0.1774987109, 0.1899451004, 0.1342504721, 0.6098059832],
        [0.1648223071, 0.1263911611, 0.7919013211, 0.1571221094],
        [0.1829168060, 0.1415705277, 0.1369831995, 0.7853701584],
        [0.1189962337, 0.1330194932, 0.7619838098, 0.1076181498],
        [0.1648099673, 0.1390340659, 0.8027287687, 0.1552326503],
        [0.1739479328, 0.1557568898, 0.1344118101, 0.7885210251],
        [0.1521665844, 0.1665889892, 0.8554084351, 0.1218915804],
        [0.1206967682, 0.1185203270, 0.8604285941, 0.1074882286],
        [0.1373380583, 0.1130456763, 0.7780663206, 0.1072449111],
        [0.1694233974, 0.1487836913, 0.7149735181, 0.1510681709],
        [0.2009198939, 0.1504507646, 0.1767894403, 0.8288005734],
    ]
)


@pytest.mark.parametrize("solver", ["nipals", "cf"])
def test_matches_moabb(solver):
    """Predictions and correlations match MOABB's SSVEP_CCA.

    MOABB turns correlations into probabilities differently (it subtracts each
    row's minimum), so predict_proba is compared with MOABB's correlations
    normalised the way this classifier does it: divided by their row sum.
    """
    X, y = _make_ssvep_data(n_cases=20)
    clf = SSVEPCCAClassifier(
        sfreq=sfreq, class_frequencies=freqs, n_harmonics=3, solver=solver
    ).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), MOABB_PREDICTIONS)
    expected = MOABB_SCORES / MOABB_SCORES.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(clf.predict_proba(X), expected, atol=1e-6)


def test_solvers_agree():
    """The closed form and sklearn's iterative CCA give the same results."""
    X, y = _make_ssvep_data(n_cases=40, random_state=1)
    results = {}
    for solver in ("nipals", "cf"):
        clf = SSVEPCCAClassifier(
            sfreq=sfreq, class_frequencies=freqs, n_harmonics=3, solver=solver
        ).fit(X, y)
        results[solver] = (clf.predict(X), clf.predict_proba(X))
    np.testing.assert_array_equal(results["nipals"][0], results["cf"][0])
    np.testing.assert_allclose(results["nipals"][1], results["cf"][1], atol=1e-6)


def test_predict_shape_and_labels():
    """Predictions have the right shape and only contain known labels."""
    X, y = _make_ssvep_data()
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(freqs))


def test_references_shape():
    """One reference per class, each (2 * n_harmonics, n_timepoints)."""
    X, y = _make_ssvep_data()
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs, n_harmonics=3).fit(
        X, y
    )
    references = clf._build_references(n_timepoints)
    assert len(references) == len(freqs)
    for reference in references:
        assert reference.shape == (2 * 3, n_timepoints)


def test_classifies_clean_signal():
    """A strong tone at each class frequency is classified correctly."""
    y = np.array(list(freqs) * 5)
    X = np.zeros((len(y), n_channels, n_timepoints))
    for i, label in enumerate(y):
        X[i] += 5 * np.sin(2 * np.pi * freqs[label] * t)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


def test_predict_proba_sums_to_one():
    """Class probabilities are non-negative and sum to one per case."""
    X, y = _make_ssvep_data(n_cases=20)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(freqs))
    assert np.all(proba >= 0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_references_from_length_only():
    """References depend on the series length and labels, not case values."""
    X, y = _make_ssvep_data()
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs).fit(X, y)
    expected = []
    for label in clf.classes_:
        rows = []
        for h in (1, 2, 3):
            rows.append(np.sin(2 * np.pi * h * freqs[label] * t))
            rows.append(np.cos(2 * np.pi * h * freqs[label] * t))
        expected.append(np.array(rows))
    for ref, exp in zip(clf._build_references(n_timepoints), expected):
        np.testing.assert_allclose(ref, exp)


@pytest.mark.parametrize("solver", ["nipals", "cf"])
def test_flat_case(solver):
    """A flat case does not crash and gets uniform probabilities."""
    X, y = _make_ssvep_data(n_cases=20)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=freqs, solver=solver).fit(
        X, y
    )
    X_flat = np.zeros((1, n_channels, n_timepoints))
    np.testing.assert_allclose(clf.predict_proba(X_flat), 1 / len(clf.classes_))
    assert clf.predict(X_flat)[0] in clf.classes_


def test_extra_keys_in_class_frequencies_allowed():
    """class_frequencies may list classes not present in y."""
    X, y = _make_ssvep_data()
    extra = dict(freqs, e=27)
    clf = SSVEPCCAClassifier(sfreq=sfreq, class_frequencies=extra).fit(X, y)
    assert set(clf.predict(X)).issubset(set(freqs))


@pytest.mark.parametrize(
    "params, match",
    [
        ({"sfreq": 0}, "sfreq must be finite and positive"),
        ({"sfreq": -250}, "sfreq must be finite and positive"),
        ({"sfreq": np.inf}, "sfreq must be finite and positive"),
        ({"sfreq": "250"}, "sfreq must be finite and positive"),
        ({"n_harmonics": 0}, "n_harmonics must be a positive integer"),
        ({"n_harmonics": True}, "n_harmonics must be a positive integer"),
        ({"n_harmonics": 1.5}, "n_harmonics must be a positive integer"),
        ({"solver": "cff"}, "solver must be 'cf' or 'nipals'"),
        ({"class_frequencies": {}}, "must contain at least one class"),
        (
            {"class_frequencies": {"a": 8, "b": 11, "c": 15, "d": -1}},
            "must be finite and positive",
        ),
        (
            {"class_frequencies": {"a": 8, "b": 11, "c": 15, "d": 200}},
            "below the Nyquist frequency",
        ),
        (
            {"class_frequencies": {"a": 8, "b": 8, "c": 15, "d": 23}},
            "duplicate frequencies",
        ),
        (
            {"class_frequencies": {"a": 8, "b": 11, "c": 15}},
            r"Labels \['d'\] in y are not in class_frequencies",
        ),
    ],
)
def test_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X, y = _make_ssvep_data(n_cases=20)
    kwargs = {"sfreq": sfreq, "class_frequencies": freqs, **params}
    with pytest.raises(ValueError, match=match):
        SSVEPCCAClassifier(**kwargs).fit(X, y)
