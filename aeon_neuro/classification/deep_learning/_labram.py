"""Braindecode LaBraM wrapper for aeon."""

from __future__ import annotations

__all__ = ["LabramClassifier"]

from functools import cache

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

_MAX_PATCHES = 16  # temporal embedding slots in the pretrained checkpoint


class LabramClassifier(BaseClassifier):
    """Frozen LaBraM feature extraction followed by a scikit-learn classifier.

    The pretrained LaBraM encoder [1]_ from Braindecode is used as a fixed feature
    extractor: it is never trained. Each series is embedded by the encoder, the
    patch embeddings are averaged over time patches, and ``estimator`` is fitted
    on the resulting ``n_channels * 200`` features.

    This classifier is for equal-length collection data and accepts multivariate
    time series in aeon's standard shape: (n_cases, n_channels, n_timepoints).

    LaBraM splits each channel into patches of 200 samples, so series are
    zero-padded at the end to the next multiple of 200. The pretrained weights
    expect data sampled at 200 Hz; resampling is left to the user. Inputs of at
    most 16 patches (16 s at 200 Hz) are supported, and pretraining used at most
    256 patches in total (``n_channels * n_patches``).

    The authors' data was band-pass filtered at 0.1-75 Hz, notch filtered at
    50 Hz, and scaled to units of 0.1 mV (microvolts divided by 100), following
    their preprocessing
    (``dataset_maker/make_TUAB.py`` and ``engine_for_finetuning.py`` in the
    official LaBraM repository). Data loaded with MNE is in volts, so multiply it
    by ``1e4``. No filtering or scaling is applied here.

    LaBraM learns one embedding per electrode, so the channel names must be given
    and must be standard 10-20 system names in LaBraM's 128-electrode list (e.g.
    ``"FP1"``, ``"CZ"``, ``"O2"``; case-insensitive).

    The LaBraM authors report that frozen features (linear probing) perform
    noticeably worse than fine-tuning, especially on multi-class tasks [1]_.

    Parameters
    ----------
    ch_names : list of str or None, default=None
        Electrode name of each channel, in order. Required when
        ``pretrained=True``. If None and ``pretrained=False``, the first
        ``n_channels`` names of LaBraM's electrode list are used.
    estimator : sklearn classifier or None, default=None
        Classifier fitted on the extracted features. If None, uses a
        ``StandardScaler`` followed by ``RidgeClassifierCV``.
    pretrained : bool, default=True
        If True, load the encoder weights from ``"braindecode/labram-pretrained"``
        on the Hugging Face Hub. Requires network access the first time; set
        ``HF_HUB_OFFLINE=1`` to reuse a cached copy. If False, use randomly
        initialised encoder weights, mainly for ablation and offline testing.
    batch_size : int, default=64
        Mini-batch size used for feature extraction.
    device : str or None, default=None
        Torch device string. If None, uses "cuda" when available, otherwise "cpu".
    random_state : int or None, default=None
        Random seed for the encoder initialisation (when ``pretrained=False``)
        and for ``estimator``.

    References
    ----------
    .. [1] Jiang, W.-B., Zhao, L.-M., and Lu, B.-L. (2024). Large Brain Model for
       Learning Generic Representations with Tremendous EEG Data in BCI.
       International Conference on Learning Representations (ICLR).
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        # braindecode's Labram cannot be pickled (lambda inside its MLP blocks)
        "cant_pickle": True,
        "python_dependencies": ["braindecode", "torch"],
    }

    def __init__(
        self,
        ch_names: list[str] | None = None,
        estimator=None,
        pretrained: bool = True,
        batch_size: int = 64,
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.ch_names = ch_names
        self.estimator = estimator
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit LabramClassifier."""
        import torch

        rng = check_random_state(self.random_state)
        torch.manual_seed(int(rng.randint(np.iinfo(np.int32).max)))

        X = self._pad(X)
        self.device_ = self._resolve_device()
        self.network_ = (
            self._build_network(n_channels=X.shape[1], n_timepoints=X.shape[2])
            .to(self.device_)
            .eval()
        )

        estimator = self.estimator
        if estimator is None:
            estimator = make_pipeline(
                StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            )
        self.estimator_ = _clone_estimator(estimator, self.random_state)
        self.estimator_.fit(self._extract_features(X), y)

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        return self.estimator_.predict(self._extract_features(self._pad(X)))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        X_t = self._extract_features(self._pad(X))
        if callable(getattr(self.estimator_, "predict_proba", None)):
            return self.estimator_.predict_proba(X_t)

        dists = np.zeros((len(X_t), self.n_classes_))
        for i, pred in enumerate(self.estimator_.predict(X_t)):
            dists[i, self._class_dictionary[pred]] = 1
        return dists

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Embed X with the frozen encoder and average over time patches."""
        import torch

        features = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[start : start + self.batch_size])
                tokens = self.network_(
                    xb.to(self.device_), return_features=True, ch_names=self.ch_names_
                )["features"]
                # (batch, n_channels * n_patches, 200) -> (batch, n_channels * 200)
                emb = tokens.unflatten(1, (xb.shape[1], -1)).mean(dim=2)
                features.append(emb.flatten(start_dim=1).cpu().numpy())
        return np.concatenate(features)

    @staticmethod
    def _pad(X: np.ndarray) -> np.ndarray:
        """Zero-pad the time axis to a multiple of 200, cast to float32."""
        X = np.asarray(X, dtype=np.float32, order="C")
        n_pad = -X.shape[-1] % 200
        return np.pad(X, ((0, 0), (0, 0), (0, n_pad))) if n_pad else X

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode LaBraM encoder."""
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        n_patches = n_timepoints // 200
        if n_patches > _MAX_PATCHES:
            raise ValueError(
                f"LabramClassifier supports at most {_MAX_PATCHES} patches of 200 "
                f"samples, got {n_patches} (n_timepoints={n_timepoints} after "
                "padding)."
            )

        if self.ch_names is not None:
            self.ch_names_ = list(self.ch_names)
        elif self.pretrained:
            raise ValueError("ch_names is required when pretrained=True.")
        else:
            self.ch_names_ = LABRAM_CHANNEL_ORDER[:n_channels]

        labram = _labram_class()
        if self.pretrained:
            network = labram.from_pretrained("braindecode/labram-pretrained")
        else:
            network = labram(  # same geometry as the pretrained checkpoint
                n_times=3000,
                n_outputs=0,
                chs_info=[{"ch_name": name} for name in LABRAM_CHANNEL_ORDER],
            )
        network.patch_embed[0].n_patchs = n_patches
        return network

    def _resolve_device(self):
        """Resolve the torch device to use."""
        import torch

        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is not available.")

        return device

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the aeon estimator checks."""
        return {"pretrained": False, "batch_size": 4, "random_state": 0}


@cache
def _labram_class():
    """Braindecode Labram that sizes its embeddings from the input length.

    braindecode 1.8.1 fixes the number of patches when the model is built, so the
    pretrained checkpoint only accepts 15-patch (3000-sample) input. Like the
    original LaBraM implementation, use the first ``n_patches`` temporal
    embeddings, with ``n_patches`` set on ``patch_embed[0].n_patchs`` (which also
    sizes the spatial embedding). Verified to match the original LaBraM code
    exactly for 1-16 patches.
    """
    from braindecode.models import Labram

    # ponytail: overrides braindecode internals (braindecode issue #1227); drop
    # once a release sizes LaBraM's embeddings from the input.
    class _Labram(Labram):
        def _adj_temporal_embedding(self, num_ch, batch_size, dim_embed=None):
            n_patches = self.patch_embed[0].n_patchs
            time_embed = self.temporal_embedding[:, 0:n_patches, :]
            return (
                time_embed.unsqueeze(1).expand(batch_size, num_ch, -1, -1).flatten(1, 2)
            )

    return _Labram
