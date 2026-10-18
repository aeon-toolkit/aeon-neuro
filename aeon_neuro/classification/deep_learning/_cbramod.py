"""Braindecode CBraMod wrapper for aeon."""

from __future__ import annotations

__all__ = ["CBraModClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


class CBraModClassifier(BaseClassifier):
    """Frozen CBraMod feature extraction followed by a scikit-learn classifier.

    The pretrained CBraMod encoder [1]_ from Braindecode is used as a fixed
    feature extractor: it is never trained. Each series is embedded by the
    encoder, the embeddings are averaged over time patches, and ``estimator`` is
    fitted on the resulting ``n_channels * 200`` features.

    This classifier is for equal-length collection data and accepts multivariate
    time series in aeon's standard shape: (n_cases, n_channels, n_timepoints).

    CBraMod splits each channel into patches of 200 samples, so series are
    zero-padded at the end to the next multiple of 200. The pretrained weights
    expect data sampled at 200 Hz; resampling is left to the user.

    CBraMod has no per-electrode embeddings (it uses convolutional positional
    encoding), so channel names are not needed and any number of channels is
    accepted. It was pretrained on 19 channels of the 10-20 system.

    The authors' data was band-pass filtered at 0.3-75 Hz, notch filtered at
    60 Hz, and scaled to units of 0.1 mV (microvolts divided by 100), following
    their preprocessing (``preprocessing/preprocessing_tueg_for_pretraining.py``
    and ``pretrain_trainer.py`` in the official CBraMod repository). Data loaded
    with MNE is in volts, so multiply it by ``1e4``. No filtering or scaling is
    applied here.

    Parameters
    ----------
    estimator : sklearn classifier or None, default=None
        Classifier fitted on the extracted features. If None, uses a
        ``StandardScaler`` followed by ``RidgeClassifierCV``.
    pretrained : bool, default=True
        If True, load the encoder weights from ``"braindecode/cbramod-pretrained"``
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
    .. [1] Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., and
       Pan, G. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG
       Decoding. International Conference on Learning Representations (ICLR).
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": ["braindecode", "torch"],
    }

    def __init__(
        self,
        estimator=None,
        pretrained: bool = True,
        batch_size: int = 64,
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit CBraModClassifier."""
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
                # (batch, n_channels, n_patches, 200) -> (batch, n_channels * 200)
                emb = self.network_(xb.to(self.device_)).mean(dim=2)
                features.append(emb.flatten(start_dim=1).cpu().numpy())
        return np.concatenate(features)

    @staticmethod
    def _pad(X: np.ndarray) -> np.ndarray:
        """Zero-pad the time axis to a multiple of 200, cast to float32."""
        X = np.asarray(X, dtype=np.float32, order="C")
        n_pad = -X.shape[-1] % 200
        return np.pad(X, ((0, 0), (0, 0), (0, n_pad))) if n_pad else X

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode CBraMod encoder."""
        from braindecode.models import CBraMod

        kwargs = dict(
            n_outputs=self.n_classes_,
            n_chans=n_channels,
            n_times=n_timepoints,
            return_encoder_output=True,
        )
        if self.pretrained:
            return CBraMod.from_pretrained("braindecode/cbramod-pretrained", **kwargs)
        return CBraMod(**kwargs)

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
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the aeon estimator checks."""
        return {"pretrained": False, "batch_size": 4, "random_state": 0}
