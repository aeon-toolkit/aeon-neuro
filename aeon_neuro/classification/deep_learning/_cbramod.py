"""Braindecode CBraMod wrapper for aeon."""

from __future__ import annotations

__all__ = ["CBraModClassifier"]

import numpy as np

from aeon_neuro.classification.deep_learning._foundation import (
    _BaseFoundationClassifier,
)


class CBraModClassifier(_BaseFoundationClassifier):
    """Fine-tuned CBraMod classifier.

    The CBraMod encoder [1]_ from Braindecode is loaded with pretrained weights,
    a new classification head is added, and the whole network is fine-tuned.

    This classifier is for equal-length collection data and accepts multivariate
    time series in aeon's standard shape: (n_cases, n_channels, n_timepoints).

    CBraMod splits each channel into patches of ``patch_size`` samples, so series
    are zero-padded at the end to the next multiple of ``patch_size``. The
    pretrained weights expect data sampled at 200 Hz with ``patch_size=200``;
    resampling is left to the user.

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
    pretrained : bool, default=True
        If True, load the encoder weights from ``"braindecode/cbramod-pretrained"``
        on the Hugging Face Hub. Requires network access the first time; set
        ``HF_HUB_OFFLINE=1`` to reuse a cached copy. If False, train the same
        architecture from random weights, mainly for ablation and offline testing.
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
    authors fine-tune (``finetune_main.py`` and ``finetune_trainer.py`` in the
    official repository) for 50 epochs with AdamW, ``lr=1e-4``,
    ``head_lr=1e-3 * (batch_size / 256) ** 0.5``, ``weight_decay=0.05``, a cosine
    schedule without warmup, ``clip_grad=1`` and ``label_smoothing=0.1``, and keep
    the epoch with the best validation score.

    The remaining parameters mirror the main architectural parameters of
    `braindecode.models.CBraMod`. They are ignored when ``pretrained=True``.

    patch_size : int, default=200
        Samples per patch. Must be 200 when ``pretrained=True``.
    dim_feedforward : int, default=800
        Feedforward dimension of each transformer layer.
    n_layer : int, default=12
        Number of transformer layers.
    nhead : int, default=8
        Number of attention heads.
    emb_dim : int, default=200
        Embedding dimension.
    drop_prob : float, default=0.1
        Dropout probability.

    References
    ----------
    .. [1] Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., and
       Pan, G. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG
       Decoding. International Conference on Learning Representations (ICLR).
    """

    def __init__(
        self,
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
        patch_size: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        emb_dim: int = 200,
        drop_prob: float = 0.1,
    ):
        self.pretrained = pretrained
        self.patch_size = patch_size
        self.dim_feedforward = dim_feedforward
        self.n_layer = n_layer
        self.nhead = nhead
        self.emb_dim = emb_dim
        self.drop_prob = drop_prob

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
        """Zero-pad the time axis to a multiple of patch_size, cast to float32."""
        X = super()._pad(X)
        n_pad = -X.shape[-1] % self.patch_size
        return np.pad(X, ((0, 0), (0, 0), (0, n_pad))) if n_pad else X

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode CBraMod model."""
        from braindecode.models import CBraMod

        if self.pretrained:
            if self.patch_size != 200:
                raise ValueError(
                    "pretrained=True requires patch_size=200, "
                    f"got patch_size={self.patch_size}."
                )
            return CBraMod.from_pretrained(
                "braindecode/cbramod-pretrained",
                n_outputs=self.n_classes_,
                n_chans=n_channels,
                n_times=n_timepoints,
            )

        return CBraMod(
            n_chans=n_channels,
            n_outputs=self.n_classes_,
            n_times=n_timepoints,
            patch_size=self.patch_size,
            dim_feedforward=self.dim_feedforward,
            n_layer=self.n_layer,
            nhead=self.nhead,
            emb_dim=self.emb_dim,
            drop_prob=self.drop_prob,
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the aeon estimator checks."""
        return {
            "pretrained": False,
            "batch_size": 4,
            "n_epochs": 1,
            "random_state": 0,
            "dim_feedforward": 16,
            "n_layer": 1,
            "nhead": 2,
        }
