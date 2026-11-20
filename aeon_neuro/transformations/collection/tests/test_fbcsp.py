"""Tests for the FBCommonSpatialPatterns transformer."""

import numpy as np
import pytest
from aeon.testing.estimator_checking import check_estimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from aeon_neuro.transformations.collection import (
    FBCommonSpatialPatterns,
    LogVariance,
)
from aeon_neuro.transformations.collection.channel_creation import (
    CommonSpatialPatterns,
)
from aeon_neuro.transformations.series import BandPassSeriesTransformer

sfreq, n_channels, n_timepoints = 100, 6, 200
bands = [(4, 8), (8, 12), (20, 30)]


def _make_mi_data(n_cases=60, random_state=0):
    """Return noise where each class has extra 10 Hz power on its own channel.

    Only the 8-12 Hz band separates the classes, as in motor imagery, where the
    mu rhythm is stronger over one hemisphere than the other.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_cases, n_channels, n_timepoints))
    y = np.arange(n_cases) % 2
    t = np.arange(n_timepoints) / sfreq
    phases = rng.uniform(0, 2 * np.pi, n_cases)  # not phase-locked to the cue
    mu = 3 * np.sin(2 * np.pi * 10 * t[None, :] + phases[:, None])
    X[y == 0, 0] += mu[y == 0]
    X[y == 1, 1] += mu[y == 1]
    return X, y


@pytest.mark.parametrize(
    "params, n_features",
    [
        ({}, len(bands) * 4),
        ({"n_components": 2}, len(bands) * 2),
        ({"n_components": 2, "k": 3}, 3),
        ({"n_components": 2, "k": len(bands) * 2}, len(bands) * 2),
        ({"bands": [(8, 12)], "n_components": 3}, 3),
    ],
)
def test_output_shape(params, n_features):
    """Output is (n_cases, k), or (n_cases, n_bands * n_components) for all."""
    X, y = _make_mi_data()
    params = {"sfreq": sfreq, "bands": bands, **params}
    Xt = FBCommonSpatialPatterns(**params).fit_transform(X, y)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (len(y), n_features)


def test_default_bands():
    """Default bands are the nine 4 Hz wide bands of Ang et al. (2012)."""
    X, y = _make_mi_data()
    X = np.repeat(X, 3, axis=2)  # longer series so 250 Hz filters fit
    fbcsp = FBCommonSpatialPatterns(sfreq=250, n_components=2).fit(X, y)
    expected = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28),
                (28, 32), (32, 36), (36, 40)]  # fmt: skip
    assert fbcsp.bands_ == expected
    assert len(fbcsp.csps_) == 9
    assert fbcsp.transform(X).shape == (len(y), 9 * 2)


def test_matches_manual_filter_bank():
    """Features equal band-pass, then per-band CSP, then log-variance."""
    X, y = _make_mi_data()
    X_train, y_train, X_test = X[:40], y[:40], X[40:]
    fbcsp = FBCommonSpatialPatterns(sfreq=sfreq, bands=bands, n_components=2)
    fbcsp.fit(X_train, y_train)

    expected = []
    for (l_freq, h_freq), fitted_csp in zip(bands, fbcsp.csps_):
        bandpass = BandPassSeriesTransformer(
            sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method="iir"
        )
        X_band_train = np.stack([bandpass.fit_transform(x) for x in X_train])
        X_band_test = np.stack([bandpass.fit_transform(x) for x in X_test])

        csp = CommonSpatialPatterns(n_components=2).fit(X_band_train, y_train)
        np.testing.assert_allclose(fitted_csp.filters_, csp.filters_)
        expected.append(LogVariance().fit_transform(csp.transform(X_band_test)))

    np.testing.assert_allclose(fbcsp.transform(X_test), np.hstack(expected))


def test_selects_discriminative_band():
    """With small k, the selected features come from the 8-12 Hz band."""
    X, y = _make_mi_data()
    fbcsp = FBCommonSpatialPatterns(
        sfreq=sfreq, bands=bands, n_components=2, k=2, random_state=0
    ).fit(X, y)
    # columns 2 and 3 are the two CSP components of bands[1] = (8, 12)
    np.testing.assert_array_equal(np.flatnonzero(fbcsp.selector_.get_support()), [2, 3])
    scores = fbcsp.selector_.scores_
    assert scores[2:4].min() > max(scores[:2].max(), scores[4:].max())


def test_transform_applies_selection_from_fit():
    """Transform keeps the columns chosen in fit, without refitting."""
    X, y = _make_mi_data()
    fbcsp = FBCommonSpatialPatterns(
        sfreq=sfreq, bands=bands, n_components=2, k=3, random_state=0
    )
    Xt = fbcsp.fit_transform(X, y)
    support = fbcsp.selector_.get_support().copy()
    np.testing.assert_allclose(fbcsp.transform(X), Xt)
    np.testing.assert_array_equal(fbcsp.selector_.get_support(), support)
    np.testing.assert_allclose(Xt, fbcsp._band_features(X)[:, support])


def test_random_state_reproducible():
    """The same random_state gives the same selection scores."""
    X, y = _make_mi_data()
    params = {"sfreq": sfreq, "bands": bands, "k": 5, "random_state": 42}
    a = FBCommonSpatialPatterns(**params).fit(X, y)
    b = FBCommonSpatialPatterns(**params).fit(X, y)
    np.testing.assert_array_equal(a.selector_.scores_, b.selector_.scores_)
    np.testing.assert_allclose(a.transform(X), b.transform(X))


def test_string_labels():
    """String class labels give the same features as integer labels."""
    X, y = _make_mi_data()
    y_str = np.where(y == 1, "right", "left")
    params = {"sfreq": sfreq, "bands": bands, "n_components": 2}
    expected = FBCommonSpatialPatterns(**params).fit_transform(X, y)
    Xt = FBCommonSpatialPatterns(**params).fit_transform(X, y_str)
    np.testing.assert_allclose(Xt, expected)


def test_requires_y():
    """Fitting without labels raises an error."""
    X, _ = _make_mi_data()
    with pytest.raises(ValueError, match="requires_y"):
        FBCommonSpatialPatterns(sfreq=sfreq, bands=bands).fit(X)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"bands": []}, "bands must be None or a non-empty list"),
        ({"bands": (8, 12)}, "Each band must be a pair of numbers"),
        ({"bands": [(8,)]}, "Each band must be a pair of numbers"),
        ({"bands": [(8, 12, 16)]}, "Each band must be a pair of numbers"),
        ({"bands": [("8", 12)]}, "Each band must be a pair of numbers"),
        ({"bands": [(True, 12)]}, "Each band must be a pair of numbers"),
        ({"bands": [(12, 8)]}, "0 < l_freq < h_freq"),
        ({"bands": [(0, 8)]}, "0 < l_freq < h_freq"),
        ({"bands": [(8, 8)]}, "0 < l_freq < h_freq"),
        ({"bands": [(30, 60)]}, "h_freq must be less than 50.0 Hz"),
        ({"k": 0}, "k must be 'all' or a positive int"),
        ({"k": -1}, "k must be 'all' or a positive int"),
        ({"k": True}, "k must be 'all' or a positive int"),
        ({"k": 1.5}, "k must be 'all' or a positive int"),
        ({"k": "best"}, "k must be 'all' or a positive int"),
        ({"k": 100}, r"k=100 exceeds the number of features \(12\)"),
        ({"sfreq": -1}, "sfreq must be a finite positive number"),
        ({"method": "butter"}, "method must be 'fir' or 'iir'"),
    ],
)
def test_invalid_params(params, match):
    """Invalid parameters raise a ValueError."""
    X, y = _make_mi_data()
    params = {"sfreq": sfreq, "bands": bands, **params}
    with pytest.raises(ValueError, match=match):
        FBCommonSpatialPatterns(**params).fit(X, y)


def test_lda_pipeline():
    """FBCSP with LDA separates classes that differ in 8-12 Hz power."""
    X, y = _make_mi_data(n_cases=200)
    X_train, y_train, X_test, y_test = X[:140], y[:140], X[140:], y[140:]
    pipeline = make_pipeline(
        FBCommonSpatialPatterns(
            sfreq=sfreq, bands=bands, n_components=2, k=2, random_state=0
        ),
        LinearDiscriminantAnalysis(),
    )
    pipeline.fit(X_train, y_train)
    assert pipeline.score(X_test, y_test) > 0.9


def test_aeon_estimator_checks():
    """Run aeon's estimator checks on the transformer."""
    check_estimator(FBCommonSpatialPatterns, raise_exceptions=True)
