"""Tests for the NeuralBench experiment wrappers."""

import numpy as np
import pytest
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.estimator_checking import check_estimator

from aeon_neuro.classification.experimental import (
    MOABBTSLRClassifier,
    NeuralBenchCTNetClassifier,
    NeuralBenchPreprocessor,
    NeuralBenchTSLRClassifier,
)


def _tones(sfreq, n_timepoints, freqs, n_cases=4, n_channels=2):
    t = np.arange(n_timepoints) / sfreq
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return np.tile(signal, (n_cases, n_channels, 1))


def _power_at(X, sfreq, freq):
    spectrum = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], 1 / sfreq)
    return spectrum[..., np.argmin(np.abs(freqs - freq))].mean()


def test_preprocessor_resamples_to_120_hz():
    """Check the output length matches resampling to 120 Hz."""
    X = np.random.default_rng(0).standard_normal((3, 2, 640))
    Xt = NeuralBenchPreprocessor(sfreq=160.0).fit_transform(X)
    assert Xt.shape == (3, 2, 480)


def test_preprocessor_notches_line_noise():
    """Check 50 Hz is removed and lower frequencies are kept."""
    sfreq = 250.0
    X = _tones(sfreq, 1000, [10.0, 50.0])
    pre = NeuralBenchPreprocessor(
        sfreq=sfreq, target_sfreq=None, scaling=None, clamp=None
    )
    Xt = pre.fit_transform(X)

    assert _power_at(Xt, sfreq, 50.0) < 1e-2 * _power_at(X, sfreq, 50.0)
    assert _power_at(Xt, sfreq, 10.0) > 0.9 * _power_at(X, sfreq, 10.0)


def test_preprocessor_demeans_each_case():
    """Check each channel's mean is removed in each case."""
    X = np.random.default_rng(0).standard_normal((3, 2, 200)) + 5.0
    pre = NeuralBenchPreprocessor(sfreq=120.0, notch_freqs=(), scaling=None, clamp=None)
    np.testing.assert_allclose(pre.fit_transform(X).mean(axis=2), 0, atol=1e-10)


def test_preprocessor_scales_with_training_statistics():
    """Check training data get median 0 and IQR 1, and test data reuse them."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 2, 240)) * [[3.0], [1e-6]]
    pre = NeuralBenchPreprocessor(sfreq=120.0, clamp=None)
    Xt = pre.fit_transform(X_train)

    pooled = np.moveaxis(Xt, 1, 0).reshape(2, -1)
    q25, median, q75 = np.percentile(pooled, [25, 50, 75], axis=1)
    np.testing.assert_allclose(median, 0, atol=1e-10)
    np.testing.assert_allclose(q75 - q25, 1, atol=1e-10)

    # test data are scaled with the training statistics, not their own, so
    # doubling the input doubles the interquartile range of the output
    np.testing.assert_allclose(pre.transform(X_train), Xt)
    pooled = np.moveaxis(pre.transform(2 * X_train), 1, 0).reshape(2, -1)
    q25, q75 = np.percentile(pooled, [25, 75], axis=1)
    np.testing.assert_allclose(q75 - q25, 2, rtol=1e-6)


def test_preprocessor_clamps():
    """Check scaled values are clamped to 20."""
    X = np.random.default_rng(0).standard_normal((5, 2, 240))
    X[0, 0, 100] = 1e3
    Xt = NeuralBenchPreprocessor(sfreq=120.0).fit_transform(X)
    assert np.abs(Xt).max() == 20.0


def test_preprocessor_needs_sfreq():
    """Check a missing sampling frequency raises."""
    X = np.zeros((2, 2, 50))
    with pytest.raises(ValueError, match="sfreq"):
        NeuralBenchPreprocessor().fit(X)


def _covariance_data(n_cases=60, n_channels=4, n_timepoints=240, random_state=0):
    """Classes that differ in the variance of their first channel."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_cases, n_channels, n_timepoints))
    y = np.arange(n_cases) % 2
    X[y == 1, 0] *= 3.0
    return X, y


def test_tslr_matches_pyriemann_pipeline():
    """Check TS-LR equals the plain pyriemann and sklearn pipeline."""
    from pyriemann.estimation import Covariances, Shrinkage
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _covariance_data()
    clf = NeuralBenchTSLRClassifier(preprocess=False, random_state=0).fit(X, y)
    expected = make_pipeline(
        Covariances(estimator="oas"),
        Shrinkage(shrinkage=0.1),
        TangentSpace(metric="riemann"),
        StandardScaler(),
        LogisticRegressionCV(
            Cs=[10.0**k for k in range(-3, 4)],
            cv=5,
            max_iter=1000,
            class_weight="balanced",
            scoring="accuracy",
        ),
    ).fit(X, y)

    np.testing.assert_allclose(clf.predict_proba(X), expected.predict_proba(X))
    assert clf.metric_ == "riemann"


def test_tslr_separates_variance_classes():
    """Check TS-LR with preprocessing learns a variance difference."""
    X, y = _covariance_data(n_cases=120)
    clf = NeuralBenchTSLRClassifier(sfreq=120.0, random_state=0)
    clf.fit(X[:80], y[:80])
    assert clf.score(X[80:], y[80:]) > 0.9


def test_tslr_drops_flat_training_cases_and_handles_flat_test_cases():
    """Check flat cases are dropped in fit and get finite probabilities."""
    X, y = _covariance_data()
    X[0] = 0.0
    clf = NeuralBenchTSLRClassifier(preprocess=False, random_state=0).fit(X, y)
    assert clf.n_dropped_ == 1

    proba = clf.predict_proba(X[:1])
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_moabb_tslr_matches_moabb_pipeline():
    """Check MOABB TS-LR equals MNE's IIR band-pass then the TSLR.yml steps."""
    import mne
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    X, y = _covariance_data()
    clf = MOABBTSLRClassifier(sfreq=120.0, random_state=0).fit(X, y)
    X_f = mne.filter.filter_data(X, 120.0, 8.0, 32.0, method="iir", verbose=False)
    expected = make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(C=1.0),
    ).fit(X_f, y)

    np.testing.assert_allclose(clf.predict_proba(X), expected.predict_proba(X_f))


def test_moabb_tslr_uses_only_the_band():
    """Check a class difference outside 8-32 Hz is filtered out."""
    sfreq, n_timepoints = 120.0, 480
    t = np.arange(n_timepoints) / sfreq
    rng = np.random.default_rng(0)
    y = np.arange(120) % 2
    X = rng.standard_normal((120, 4, n_timepoints))
    in_band = X.copy()
    in_band[y == 1, 0] += 3 * np.sin(2 * np.pi * 12 * t)
    out_band = X.copy()
    out_band[y == 1, 0] += 3 * np.sin(2 * np.pi * 50 * t)

    def accuracy(data):
        clf = MOABBTSLRClassifier(sfreq=sfreq, random_state=0)
        return clf.fit(data[:80], y[:80]).score(data[80:], y[80:])

    assert accuracy(in_band) > 0.9
    assert accuracy(out_band) < 0.75


def test_moabb_tslr_needs_sfreq_to_filter():
    """Check filtering without a sampling frequency raises."""
    X, y = _covariance_data()
    with pytest.raises(ValueError, match="sfreq"):
        MOABBTSLRClassifier().fit(X, y)


def test_ctnet_uses_neuralbench_settings():
    """Check the CTNet wrapper passes NeuralBench's training recipe."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=100, random_state=0
    )
    clf = NeuralBenchCTNetClassifier(**NeuralBenchCTNetClassifier._get_test_params())
    clf.fit(X, y)

    params = clf.classifier_.get_params()
    assert params["model"] == "CTNet"
    assert params["optimizer"] == "adamw"
    assert params["lr_scheduler"] == "onecycle"
    assert params["gradient_clip"] == 1.0
    assert params["class_weight"] == "balanced"
    assert params["label_smoothing"] == 0.1
    assert params["sfreq"] == 120.0
    defaults = NeuralBenchCTNetClassifier()
    assert (defaults.n_epochs, defaults.early_stopping_patience) == (40, 5)
    assert (defaults.lr, defaults.weight_decay) == (1e-4, 0.05)
    assert defaults.validation_size == 0.25


def test_ctnet_validation_by_subject():
    """Check subject IDs passed to the wrapper give a subject-wise split."""
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=2, n_timepoints=100, random_state=0
    )
    params = NeuralBenchCTNetClassifier._get_test_params()
    params["validation_size"] = 0.25
    groups = np.repeat(np.arange(8), 2)
    clf = NeuralBenchCTNetClassifier(train_groups=groups, **params).fit(X, y)
    assert len(clf.classifier_.validation_groups_) == 2

    clf = NeuralBenchCTNetClassifier(**params).fit(X, y)
    assert clf.classifier_.validation_groups_ is None


def test_ctnet_pads_short_cases():
    """Check cases shorter than CTNet's minimum length are padded to 64."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=40, random_state=0
    )
    clf = NeuralBenchCTNetClassifier(**NeuralBenchCTNetClassifier._get_test_params())
    clf.fit(X, y)
    assert clf.classifier_.metadata_["n_timepoints"] == 64
    assert clf.predict_proba(X).shape == (12, 2)


@pytest.mark.parametrize(
    "estimator",
    [
        NeuralBenchPreprocessor,
        NeuralBenchTSLRClassifier,
        NeuralBenchCTNetClassifier,
        MOABBTSLRClassifier,
    ],
)
def test_aeon_estimator_checks(estimator):
    """Run aeon's estimator checks."""
    check_estimator(estimator, raise_exceptions=True)
