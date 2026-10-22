"""Braindecode EEGPT wrapper for aeon."""

from __future__ import annotations

__all__ = ["EEGPTClassifier"]

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

_PATCH_SIZE = 64  # samples per patch in the pretrained checkpoint


class EEGPTClassifier(BaseClassifier):
    """Frozen EEGPT feature extraction followed by a scikit-learn classifier.

    The pretrained EEGPT encoder [1]_ from Braindecode is used as a fixed feature
    extractor: it is never trained. Each series is embedded by the encoder into
    4 summary tokens per time patch, the summary tokens are averaged over time
    patches, and ``estimator`` is fitted on the resulting ``4 * 512`` features.
    This matches the linear probing evaluation used in the EEGPT paper.

    This classifier is for equal-length collection data and accepts multivariate
    time series in aeon's standard shape: (n_cases, n_channels, n_timepoints).

    EEGPT splits each channel into patches of 64 samples with a stride of 32.
    Series shorter than 64 samples are zero-padded. The pretrained weights expect
    data sampled at 256 Hz (pretraining used 4 s windows); resampling is left to
    the user.

    The pretraining data was in millivolts and average-referenced (the mean over
    channels subtracted at each time point), following the authors' preprocessing
    (``datasets/pretrain/prepare_pretrain_dataset.py`` in the official EEGPT
    repository). Data loaded with MNE is in volts, so multiply it by ``1e3``. No
    scaling or referencing is applied here.

    EEGPT learns one embedding per electrode, so the channel names must be given
    and must be standard 10-20 system names in EEGPT's 62-electrode list (e.g.
    ``"FP1"``, ``"CZ"``, ``"O2"``; case-insensitive).

    Parameters
    ----------
    ch_names : list of str or None, default=None
        Electrode name of each channel, in order. Required when
        ``pretrained=True``. If None and ``pretrained=False``, the first
        ``n_channels`` names of EEGPT's electrode list are used.
    estimator : sklearn classifier or None, default=None
        Classifier fitted on the extracted features. If None, uses a
        ``StandardScaler`` followed by ``RidgeClassifierCV``.
    pretrained : bool, default=True
        If True, load the encoder weights from ``"braindecode/eegpt-pretrained"``
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
    .. [1] Wang, G., Liu, W., He, Y., Xu, C., Ma, L., and Li, H. (2024). EEGPT:
       Pretrained Transformer for Universal and Reliable Representation of EEG
       Signals. Advances in Neural Information Processing Systems (NeurIPS).
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
        """Fit EEGPTClassifier."""
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
                # (batch, n_patches, 4, 512) -> (batch, 4 * 512)
                emb = self.network_(xb.to(self.device_)).mean(dim=1)
                features.append(emb.flatten(start_dim=1).cpu().numpy())
        return np.concatenate(features)

    @staticmethod
    def _pad(X: np.ndarray) -> np.ndarray:
        """Zero-pad the time axis to at least one patch, cast to float32."""
        X = np.asarray(X, dtype=np.float32, order="C")
        n_pad = max(_PATCH_SIZE - X.shape[-1], 0)
        return np.pad(X, ((0, 0), (0, 0), (0, n_pad))) if n_pad else X

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode EEGPT encoder."""
        from braindecode.models import EEGPT
        from braindecode.models.eegpt import EEGPT_CHANNELS

        if self.ch_names is not None:
            self.ch_names_ = list(self.ch_names)
            unknown = [n for n in self.ch_names_ if n.upper() not in EEGPT_CHANNELS]
            if unknown:
                raise ValueError(
                    f"Unknown channel names {unknown}; EEGPT supports: "
                    f"{EEGPT_CHANNELS}."
                )
        elif self.pretrained:
            raise ValueError("ch_names is required when pretrained=True.")
        else:
            self.ch_names_ = EEGPT_CHANNELS[:n_channels]

        # chan_proj_type="none": the default channel projection layer is not in
        # the checkpoint, so it would be a random layer in front of the encoder.
        network = EEGPT(
            n_chans=n_channels,
            n_times=n_timepoints,
            chs_info=[{"ch_name": name} for name in self.ch_names_],
            chan_proj_type="none",
            return_encoder_output=True,
        )
        if self.pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            state = load_file(
                hf_hub_download("braindecode/eegpt-pretrained", "model.safetensors")
            )
            # Only the encoder: the checkpoint's chans_id is sized for its own 62
            # channels and is rebuilt above from ch_names.
            network.load_state_dict(
                {k: v for k, v in state.items() if k.startswith("target_encoder.")},
                strict=False,
            )
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
