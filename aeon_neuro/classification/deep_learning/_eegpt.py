"""Braindecode EEGPT wrapper for aeon."""

from __future__ import annotations

__all__ = ["EEGPTClassifier"]

import numpy as np

from aeon_neuro.classification.deep_learning._foundation import (
    _BaseFoundationClassifier,
)

# Older 10-20 names of four electrodes, which EEGPT's electrode list omits.
_OLD_1020_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_PATCH_SIZE = 64  # samples per patch in the pretrained checkpoint


class EEGPTClassifier(_BaseFoundationClassifier):
    """Fine-tuned EEGPT classifier.

    The EEGPT encoder [1]_ from Braindecode is loaded with pretrained weights, a
    new linear classification head is added on its summary tokens, and the whole
    network is fine-tuned.

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
    ``"FP1"``, ``"CZ"``, ``"O2"``; case-insensitive). The older names T3, T4, T5
    and T6 are accepted as T7, T8, P7 and P8.

    Parameters
    ----------
    ch_names : list of str or None, default=None
        Electrode name of each channel, in order. Required when
        ``pretrained=True``. If None and ``pretrained=False``, the first
        ``n_channels`` names of EEGPT's electrode list are used.
    pretrained : bool, default=True
        If True, load the encoder weights from ``"braindecode/eegpt-pretrained"``
        on the Hugging Face Hub. Requires network access the first time; set
        ``HF_HUB_OFFLINE=1`` to reuse a cached copy. If False, use randomly
        initialised encoder weights, mainly for ablation and offline testing.
    batch_size : int, default=64
        Mini-batch size used for training and inference.
    n_epochs : int, default=20
        Number of training epochs.
    lr : float, default=1e-4
        Learning rate.
    optimizer : {"adam", "adamw", "sgd"}, default="adam"
        Optimiser used for training.
    weight_decay : float, default=0.0
        L2 penalty passed to the optimiser.
    device : str or None, default=None
        Torch device string. If None, uses "cuda" when available, otherwise "cpu".
    verbose : bool, default=False
        If True, print epoch losses during training.
    random_state : int or None, default=None
        Random seed.
    freeze_encoder : bool, default=False
        If True, train only the new head on the frozen encoder's output (linear
        probing). If False, fine-tune the whole network.
    head_lr : float or None, default=None
        Learning rate of the head. If None, ``lr`` is used.
    layer_decay : float, default=1.0
        Layer-wise learning rate decay over the transformer blocks.
    lr_schedule : {None, "cosine"}, default=None
        Learning rate schedule: linear warmup, then cosine decay to zero.
    warmup_epochs : int, default=0
        Number of warmup epochs of the "cosine" schedule.
    clip_grad : float or None, default=None
        If set, clip the gradient norm to this value.
    label_smoothing : float, default=0.0
        Label smoothing of the cross-entropy loss.
    validation_data : tuple of (np.ndarray, np.ndarray) or None, default=None
        Validation series and labels. If given, the weights of the epoch with the
        best validation balanced accuracy are kept.

    See ``_BaseFoundationClassifier`` for details of the training options. The
    authors keep the encoder frozen and train only the head (linear probing,
    ``downstream/linear_probe_EEGPT_*.py`` in the official repository): 100
    epochs with AdamW, a one-cycle schedule peaking at ``lr=4e-4`` after 20% of
    training, and ``weight_decay=0.01``. In this class that is
    ``freeze_encoder=True``, ``n_epochs=100``, ``lr_schedule="cosine"`` and
    ``warmup_epochs=20``.

    References
    ----------
    .. [1] Wang, G., Liu, W., He, Y., Xu, C., Ma, L., and Li, H. (2024). EEGPT:
       Pretrained Transformer for Universal and Reliable Representation of EEG
       Signals. Advances in Neural Information Processing Systems (NeurIPS).
    """

    def __init__(
        self,
        ch_names: list[str] | None = None,
        pretrained: bool = True,
        batch_size: int = 64,
        n_epochs: int = 20,
        lr: float = 1e-4,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
        freeze_encoder: bool = False,
        head_lr: float | None = None,
        layer_decay: float = 1.0,
        lr_schedule: str | None = None,
        warmup_epochs: int = 0,
        clip_grad: float | None = None,
        label_smoothing: float = 0.0,
        validation_data: tuple | None = None,
    ):
        self.ch_names = ch_names
        self.pretrained = pretrained

        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            random_state=random_state,
            freeze_encoder=freeze_encoder,
            head_lr=head_lr,
            layer_decay=layer_decay,
            lr_schedule=lr_schedule,
            warmup_epochs=warmup_epochs,
            clip_grad=clip_grad,
            label_smoothing=label_smoothing,
            validation_data=validation_data,
        )

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """Zero-pad the time axis to at least one patch, cast to float32."""
        X = super()._pad(X)
        n_pad = max(_PATCH_SIZE - X.shape[-1], 0)
        return np.pad(X, ((0, 0), (0, 0), (0, n_pad))) if n_pad else X

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode EEGPT model."""
        from braindecode.models import EEGPT
        from braindecode.models.eegpt import EEGPT_CHANNELS

        if self.ch_names is not None:
            self.ch_names_ = [_OLD_1020_NAMES.get(n.upper(), n) for n in self.ch_names]
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
            n_outputs=self.n_classes_,
        )
        if self.pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            state = load_file(
                hf_hub_download("braindecode/eegpt-pretrained", "model.safetensors")
            )
            # Only the encoder: the checkpoint's chans_id is sized for its own 62
            # channels and is rebuilt above from ch_names, and the head is new.
            network.load_state_dict(
                {k: v for k, v in state.items() if k.startswith("target_encoder.")},
                strict=False,
            )
        return network

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the aeon estimator checks."""
        return {"pretrained": False, "batch_size": 4, "n_epochs": 1, "random_state": 0}
