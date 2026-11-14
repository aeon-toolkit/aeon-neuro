"""Braindecode LaBraM wrapper for aeon."""

from __future__ import annotations

__all__ = ["LabramClassifier"]

from functools import cache

import numpy as np

from aeon_neuro.classification.deep_learning._foundation import (
    _BaseFoundationClassifier,
)

_MAX_PATCHES = 16  # temporal embedding slots in the pretrained checkpoint


class LabramClassifier(_BaseFoundationClassifier):
    """Fine-tuned LaBraM classifier.

    The LaBraM encoder [1]_ from Braindecode is loaded with pretrained weights, a
    new linear classification head is added, and the whole network is fine-tuned.

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

    Parameters
    ----------
    ch_names : list of str or None, default=None
        Electrode name of each channel, in order. Required when
        ``pretrained=True``. If None and ``pretrained=False``, the first
        ``n_channels`` names of LaBraM's electrode list are used.
    pretrained : bool, default=True
        If True, load the encoder weights from ``"braindecode/labram-pretrained"``
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
    drop_path : float, default=0.0
        Stochastic depth: during training, each sample skips transformer block
        ``i`` with a probability rising linearly from 0 for the first block to
        ``drop_path`` for the last.

    See ``_BaseFoundationClassifier`` for details of the training options. The
    authors fine-tune (``run_class_finetuning.py`` in the official repository)
    for 30 epochs with AdamW, ``lr=5e-4``, ``weight_decay=0.05``,
    ``layer_decay=0.9``, a cosine schedule with 5 warmup epochs,
    ``label_smoothing=0.1`` and ``drop_path=0.1``, and keep the epoch with the
    best validation accuracy.

    References
    ----------
    .. [1] Jiang, W.-B., Zhao, L.-M., and Lu, B.-L. (2024). Large Brain Model for
       Learning Generic Representations with Tremendous EEG Data in BCI.
       International Conference on Learning Representations (ICLR).
    """

    _tags = {
        # braindecode's Labram cannot be pickled (lambda inside its MLP blocks)
        "cant_pickle": True,
    }

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
        drop_path: float = 0.0,
    ):
        self.ch_names = ch_names
        self.pretrained = pretrained
        self.drop_path = drop_path

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
        """Zero-pad the time axis to a multiple of 200, cast to float32."""
        X = super()._pad(X)
        n_pad = -X.shape[-1] % 200
        return np.pad(X, ((0, 0), (0, 0), (0, n_pad))) if n_pad else X

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode LaBraM model with a new head."""
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
            network = labram.from_pretrained(
                "braindecode/labram-pretrained", drop_path_prob=self.drop_path
            )
        else:
            network = labram(  # same geometry as the pretrained checkpoint
                n_times=3000,
                n_outputs=0,
                chs_info=[{"ch_name": name} for name in LABRAM_CHANNEL_ORDER],
                drop_path_prob=self.drop_path,
            )
        network.patch_embed[0].n_patchs = n_patches
        network.ch_names = self.ch_names_
        network.reset_classifier(self.n_classes_)
        return network

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the aeon estimator checks."""
        return {"pretrained": False, "batch_size": 4, "n_epochs": 1, "random_state": 0}


@cache
def _labram_class():
    """Braindecode Labram defaulting ``ch_names`` in ``forward`` to the network's.

    The base training loop calls ``network(x)``, but Labram needs the channel
    names to pick its spatial embeddings. ``n_patches`` is set on
    ``patch_embed[0].n_patchs``, from which braindecode (>=2.0) uses the first
    ``n_patches`` temporal embeddings, as in the original LaBraM code.
    """
    from braindecode.models import Labram

    class _Labram(Labram):
        ch_names = None

        def forward(self, x, *args, ch_names=None, **kwargs):
            ch_names = self.ch_names if ch_names is None else ch_names
            return super().forward(x, *args, ch_names=ch_names, **kwargs)

    return _Labram
