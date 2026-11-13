"""Test EEGNet classifier."""

import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy

from aeon_neuro.classification.deep_learning._eegnet import EEGNetClassifier


def test_eeg_net_classifier_multivariate_equal_length():
    """Basic EEGNet test placeholder."""
    X, y = make_example_3d_numpy(
        n_cases=10,
        n_channels=3,
        n_timepoints=256,
        random_state=0,
    )

    clf = EEGNetClassifier(
        n_epochs=1,
        batch_size=4,
        random_state=0,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (10, len(np.unique(y)))
    assert preds.shape == (10,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_eegnet_lightweight_test_parameters():
    """Check that aeon receives the intended lightweight test settings."""
    params = EEGNetClassifier._get_test_params()

    assert params.get("n_epochs") == 1
    assert params.get("batch_size") == 4


def test_eegnet_subject_validation_and_optional_info_predictions():
    """Real training keeps validation subjects outside optimiser batches."""
    from aeon_neuro.datasets import EEGDatasetInfo

    X = np.random.default_rng(1).normal(size=(12, 2, 128))
    y = np.tile([0, 1], 6)
    info = EEGDatasetInfo.from_arrays(
        X,
        sfreq=128,
        channel_names=["C3", "C4"],
        units=["V", "V"],
        subjects=[f"person-{i//2}" for i in range(12)],
    )
    clf = EEGNetClassifier(
        n_epochs=2,
        batch_size=4,
        random_state=2,
        validation_size=0.34,
        patience=1,
        device="cpu",
    )
    clf.fit(X[:10], y[:10], info=info[:10])
    train, val = clf.training_indices_, clf.validation_indices_
    subjects = np.array(info[:10].participant_ids)
    assert set(subjects[train]).isdisjoint(set(subjects[val]))
    assert set(train) | set(val) == set(range(10))
    assert clf.validation_policy_ == "subject"
    assert len(clf.validation_history_) == len(clf.history_) <= 2
    assert 1 <= clf.best_epoch_ <= len(clf.history_)
    probs = clf.predict_proba(X[10:], info=info[10:])
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert clf.network_.sfreq == 128


def test_eegnet_validation_without_info_preserves_stratified_fallback():
    """Explicit early stopping remains usable through ordinary fit(X, y)."""
    X = np.random.default_rng(3).normal(size=(12, 2, 128))
    y = np.tile([0, 1], 6)
    clf = EEGNetClassifier(
        n_epochs=1, batch_size=4, random_state=1, validation_size=0.25, device="cpu"
    ).fit(X, y)
    assert clf.info_ is None and clf.validation_policy_ == "stratified cases"
    assert set(y[clf.training_indices_]) == set(y[clf.validation_indices_]) == {0, 1}


def test_eegnet_early_stopping_restores_best_validation_weights(monkeypatch):
    """Worsening validation stops training and restores the first best state."""
    import torch
    from sklearn.model_selection import GroupShuffleSplit

    from aeon_neuro.datasets import EEGDatasetInfo

    class _OneWeight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.validation_weights = []

        def forward(self, values):
            if not self.training:
                self.validation_weights.append(float(self.weight.detach()))
            score = self.weight * values[:, 0, 0]
            return torch.stack([-score, score], dim=1)

    y = np.tile([0, 1], 6)
    subjects = [f"p-{i//2}" for i in range(12)]
    X = np.zeros((12, 2, 128))
    X[:, 0, 0] = 2 * y - 1
    seed = int(np.random.RandomState(1).randint(np.iinfo(np.int32).max))
    _, val = next(
        GroupShuffleSplit(n_splits=1, test_size=0.34, random_state=seed).split(
            X, y, subjects
        )
    )
    X[val, 0, 0] *= -1
    info = EEGDatasetInfo.from_arrays(
        X, sfreq=128, channel_names=["C3", "C4"], subjects=subjects
    )
    monkeypatch.setattr(
        EEGNetClassifier, "_build_network", lambda self, **kwargs: _OneWeight()
    )
    clf = EEGNetClassifier(
        n_epochs=5,
        batch_size=12,
        lr=0.1,
        optimizer="sgd",
        validation_size=0.34,
        patience=1,
        random_state=1,
        device="cpu",
    )
    clf.fit(X, y, info=info)
    assert len(clf.history_) == 2 and clf.best_epoch_ == 1
    weights = clf.network_.validation_weights
    assert weights[-1] > weights[0]
    assert float(clf.network_.weight.detach()) == weights[0]
