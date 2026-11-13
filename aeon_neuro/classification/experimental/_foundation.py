"""Fine-tuned EEG foundation models (CBraMod, LaBraM) with NeuralBench's recipe.

``CBraModClassifier`` and ``LabramClassifier`` use the pretrained encoders as
frozen feature extractors (linear probing). The classifiers here fine-tune all
the weights instead, as NeuralBench (Banville et al., 2026, arXiv:2605.08495)
does in its full fine-tuning set-up (the ``finetune_mean`` preset in
``neuralbench/defaults/downstream_wrappers.yaml``, version 0.4.0):

- the pretrained encoder from Braindecode, without its own task head;
- the encoder's output tokens averaged over channels and time patches, then a
  single linear layer from the 200-dimensional embedding to the classes;
- every weight trained, with the recipe of ``NeuralBenchDeepClassifier``:
  AdamW (learning rate 1e-4, weight decay 0.05), one-cycle schedule with 10%
  warmup and cosine annealing stepped per batch, gradient norm clipping at 1.0,
  batch size 64, up to 40 epochs, early stopping on validation balanced
  accuracy (patience 5, keeping the best epoch), and cross-entropy with
  balanced class weights and label smoothing 0.1.

The input is preprocessed as in NeuralBench's model configs
(``neuralbench/models/cbramod.yaml`` and ``labram.yaml``), adapted to pre-cut
cases in the same way as ``NeuralBenchPreprocessor``:

- each channel's mean is removed in each case (in place of the high-pass at
  0.3 Hz for CBraMod and 0.1 Hz for LaBraM);
- a notch at 60 Hz (CBraMod) or 50 Hz (LaBraM) and its harmonics below the
  Nyquist frequency, and a low-pass at 75 Hz;
- resampling to 200 Hz;
- scaling to units of 0.1 mV (100 microvolts), the units both models were
  pretrained in, with no further scaling or clamping;
- zero padding at the end to a whole number of 200-sample (1 s) patches.

Setting ``pretrained=False`` gives the random-weight control: the same
architecture, head, preprocessing and training, from random initial weights.

The original papers fine-tune differently, and those settings are not used, so
that all deep learners are trained alike:

- CBraMod (Wang et al., 2025): AdamW, learning rate 1e-4, weight decay 0.05,
  cosine annealing, 50 epochs, label smoothing 0.1, and a head that flattens
  all patch embeddings into a multi-layer perceptron with dropout 0.1.
- LaBraM (Jiang et al., 2024): AdamW, learning rate 5e-4, weight decay 0.05,
  cosine schedule with 5 warmup epochs, 50 epochs, batch size 512, layer-wise
  learning rate decay 0.65, drop path 0.1, label smoothing 0.1, and mean
  pooling of the patch tokens into a linear head.
"""

from __future__ import annotations

__all__ = ["CBraModFineTuneClassifier", "LabramFineTuneClassifier"]

import os

import numpy as np

from aeon_neuro.classification.deep_learning._braindecode import (
    _BaseBraindecodeClassifier,
)

_PATCH = 200  # samples per patch, at 200 Hz
_TARGET_SFREQ = 200.0
_EMBED_DIM = 200
# LaBraM's pretrained checkpoint has 16 temporal embedding slots
_LABRAM_MAX_PATCHES = 16
# multiplier from each unit to the 0.1 mV (100 microvolt) pretraining units
_UNIT_SCALE = {"V": 1e4, "mV": 10.0, "uV": 1e-2}


def _mean_token_head(encoder, forward, n_classes):
    """Return encoder -> mean over tokens -> linear layer, as a torch module."""
    from torch import nn

    class _MeanTokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(_EMBED_DIM, n_classes)

        def forward(self, x):
            tokens = forward(self.encoder, x)
            # average every token axis between batch and embedding
            return self.head(tokens.flatten(1, -2).mean(dim=1))

    return _MeanTokenClassifier()


class _FoundationFineTuneClassifier(_BaseBraindecodeClassifier):
    """Shared code for fine-tuning a pretrained Braindecode encoder.

    Subclasses set ``_NOTCH``, ``_HF_NAME`` and implement ``_build_encoder``
    and ``_encode``.
    """

    _NOTCH: float = 60.0
    _HF_NAME: str = ""

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": ["braindecode", "torch", "mne"],
    }

    def __init__(
        self,
        sfreq: float | None = None,
        input_units: str | float = "uV",
        pretrained: bool = True,
        weights_path: str | None = None,
        n_epochs: int = 40,
        early_stopping_patience: int = 5,
        validation_size: float = 0.25,
        train_groups=None,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.sfreq = sfreq
        self.input_units = input_units
        self.pretrained = pretrained
        self.weights_path = weights_path
        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            optimizer="adamw",
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            random_state=random_state,
            lr_scheduler="onecycle",
            gradient_clip=1.0,
            class_weight="balanced",
            label_smoothing=0.1,
            validation_size=validation_size,
            early_stopping_patience=early_stopping_patience,
            train_groups=train_groups,
        )

    # preprocessing

    def _fit(self, X, y):
        from aeon_neuro.classification.experimental._neuralbench import (
            NeuralBenchPreprocessor,
        )

        self.unit_scale_ = self._unit_scale()
        self.preprocessor_ = NeuralBenchPreprocessor(
            sfreq=self.sfreq,
            target_sfreq=_TARGET_SFREQ,
            demean=True,
            notch_freqs=(self._NOTCH,),
            h_freq=75.0,
            scaling=None,
            clamp=None,
        )
        X = self.preprocessor_.fit_transform(X)
        self.n_times_ = X.shape[2] + (-X.shape[2] % _PATCH)
        return super()._fit(self._scale_and_pad(X), y)

    def _predict_proba(self, X):
        X = self._scale_and_pad(self.preprocessor_.transform(X))
        return super()._predict_proba(X)

    def _balanced_accuracy(self, X, y_int):
        """Return balanced accuracy on already preprocessed validation cases."""
        from sklearn.metrics import balanced_accuracy_score

        probs = super()._predict_proba(X)
        return balanced_accuracy_score(y_int, probs.argmax(axis=1))

    def _scale_and_pad(self, X):
        """Scale to 0.1 mV units and zero pad to ``n_times_`` samples."""
        X = np.asarray(X * self.unit_scale_, dtype=np.float32)
        n_pad = self.n_times_ - X.shape[2]
        if n_pad:
            X = np.pad(X, ((0, 0), (0, 0), (0, n_pad)))
        return np.ascontiguousarray(X)

    def _unit_scale(self):
        units = self.input_units
        if isinstance(units, str):
            if units not in _UNIT_SCALE:
                raise ValueError(
                    f"input_units must be one of {sorted(_UNIT_SCALE)} or a "
                    f"positive number, got {units!r}."
                )
            return _UNIT_SCALE[units]
        if isinstance(units, bool) or not isinstance(
            units, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"input_units must be one of {sorted(_UNIT_SCALE)} or a positive "
                f"number, got {units!r}."
            )
        if not units > 0:
            raise ValueError(f"input_units must be positive, got {units!r}.")
        return float(units)

    # network

    def _build_network(self, n_channels, n_timepoints, load_weights=True):
        """Build the encoder plus linear head for preprocessed input.

        The base class passes the shape of the preprocessed data, which is
        200 Hz and padded to whole patches.
        """
        encoder = self._build_encoder(n_channels, n_timepoints)
        if self.pretrained and load_weights:
            self._load_weights(encoder)
        return _mean_token_head(encoder, self._encode, self.n_classes_)

    def _load_weights(self, encoder):
        """Load the pretrained encoder weights into ``encoder``.

        From ``weights_path`` if given (a directory with ``model.safetensors``
        or ``pytorch_model.bin``, as downloaded from the Hugging Face Hub), else
        from the Hugging Face cache (``HF_HOME``), which needs
        ``huggingface_hub`` and, unless ``HF_HUB_OFFLINE=1``, network access the
        first time.
        """
        import torch

        if self.weights_path is not None:
            state = _read_state_dict(self.weights_path)
        else:
            from huggingface_hub import hf_hub_download

            try:
                path = hf_hub_download(self._HF_NAME, "model.safetensors")
            except Exception:  # noqa: BLE001, older uploads have only .bin
                path = hf_hub_download(self._HF_NAME, "pytorch_model.bin")
            state = _read_state_dict(os.path.dirname(path))
        state = {k: v.to(torch.float32) for k, v in state.items()}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        # the encoders are built without a task head; anything else must match
        missing = [k for k in missing if not k.startswith("final_layer.")]
        unexpected = [k for k in unexpected if not k.startswith("final_layer.")]
        if missing or unexpected:
            raise ValueError(
                f"The pretrained weights do not match {type(encoder).__name__}: "
                f"missing {missing}, unexpected {unexpected}."
            )

    def _build_encoder(self, n_channels, n_timepoints):
        raise NotImplementedError

    @staticmethod
    def _encode(encoder, x):
        raise NotImplementedError

    def __setstate__(self, state):
        """Rebuild the network from its pickled weights, without reloading."""
        network_state = state.pop("_network_state", None)
        self.__dict__.update(state)
        if network_state is not None:
            import torch

            if self.device_.type == "cuda" and not torch.cuda.is_available():
                self.device_ = torch.device("cpu")
            self.network_ = self._build_network(
                self.metadata_["n_channels"], self.n_times_, load_weights=False
            )
            self.network_.load_state_dict(network_state)
            self.network_.to(self.device_).eval()

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "sfreq": 100.0,
            "pretrained": False,
            "n_epochs": 1,
            "early_stopping_patience": 1,
            "validation_size": 0.3,
            "batch_size": 4,
            "random_state": 0,
        }


def _read_state_dict(directory):
    """Read ``model.safetensors`` (preferred) or ``pytorch_model.bin``."""
    import torch

    safetensors_path = os.path.join(directory, "model.safetensors")
    bin_path = os.path.join(directory, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError:
            load_file = None
        if load_file is not None:
            return load_file(safetensors_path)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    if os.path.exists(safetensors_path):
        raise ImportError(
            f"{directory} has only model.safetensors, which needs the "
            "safetensors package."
        )
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin in weights_path {directory!r}."
    )


class CBraModFineTuneClassifier(_FoundationFineTuneClassifier):
    """CBraMod fine-tuned end to end with NeuralBench's recipe.

    The pretrained CBraMod encoder [1]_ from Braindecode, with its output
    tokens averaged over channels and patches and a linear layer to the
    classes; all weights are trained. See the module docstring of
    ``aeon_neuro.classification.experimental._foundation`` for the head,
    training recipe and preprocessing, and how they differ from the paper.
    ``CBraModClassifier`` is the frozen-encoder (linear probe) version.

    CBraMod needs no channel names and accepts any number of channels.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the input in Hz. Required; the data is resampled
        to 200 Hz.
    input_units : {"uV", "mV", "V"} or float, default="uV"
        Units of the input, used to scale it to the 0.1 mV units of
        pretraining. A number is used as the multiplier directly.
    pretrained : bool, default=True
        If True, start from the pretrained weights. If False, start from
        random weights (the random-weight control).
    weights_path : str or None, default=None
        Directory holding the pretrained weights (``model.safetensors`` or
        ``pytorch_model.bin``, as in ``braindecode/cbramod-pretrained`` on the
        Hugging Face Hub). Needs no network. If None, the weights are taken
        from the Hugging Face cache (``HF_HOME``) through ``huggingface_hub``.
    n_epochs : int, default=40
        Maximum number of epochs.
    early_stopping_patience : int, default=5
        Epochs without improvement in validation balanced accuracy before
        stopping.
    validation_size : float, default=0.25
        Fraction of training subjects (or, without ``train_groups``, training
        cases) held out for early stopping.
    train_groups : array-like or None, default=None
        Subject ID of each training case, in the order passed to ``fit``. If
        given, whole subjects are held out for validation.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Peak learning rate.
    weight_decay : float, default=0.05
        AdamW weight decay.
    device : str or None, default=None
        Torch device. If None, uses CUDA when available.
    verbose : bool, default=False
        If True, print training progress.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    network_ : torch.nn.Module
        The fine-tuned network: ``network_.encoder`` and ``network_.head``.
    preprocessor_ : NeuralBenchPreprocessor
        The fitted preprocessing, before unit scaling and padding.
    n_times_ : int
        Length of the network input, in samples at 200 Hz.
    best_epoch_ : int or None
        Zero-based epoch whose weights were kept.

    References
    ----------
    .. [1] Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., and
       Pan, G. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG
       Decoding. International Conference on Learning Representations (ICLR).
    """

    _NOTCH = 60.0
    _HF_NAME = "braindecode/cbramod-pretrained"

    def _build_encoder(self, n_channels, n_timepoints):
        from braindecode.models import CBraMod

        return CBraMod(
            n_outputs=self.n_classes_,
            n_chans=n_channels,
            n_times=n_timepoints,
            return_encoder_output=True,
        )

    @staticmethod
    def _encode(encoder, x):
        # (batch, n_channels, n_patches, 200)
        return encoder(x)


class LabramFineTuneClassifier(_FoundationFineTuneClassifier):
    """LaBraM fine-tuned end to end with NeuralBench's recipe.

    The pretrained LaBraM encoder [1]_ from Braindecode, with its output
    tokens averaged and a linear layer to the classes; all weights are
    trained. See the module docstring of
    ``aeon_neuro.classification.experimental._foundation`` for the head,
    training recipe and preprocessing, and how they differ from the paper.
    ``LabramClassifier`` is the frozen-encoder (linear probe) version.

    LaBraM learns one embedding per electrode, so ``ch_names`` must give the
    standard 10-20 name of each channel, from LaBraM's 128-electrode list
    (e.g. ``"FP1"``, ``"CZ"``, ``"O2"``; case-insensitive). Cases of at most
    16 s (16 patches after resampling and padding) are supported.

    Parameters
    ----------
    ch_names : list of str or None, default=None
        Electrode name of each channel, in order. Required when
        ``pretrained=True``. If None and ``pretrained=False``, the first
        ``n_channels`` names of LaBraM's electrode list are used.
    sfreq : float
        Sampling frequency of the input in Hz. Required; the data is resampled
        to 200 Hz.
    input_units : {"uV", "mV", "V"} or float, default="uV"
        Units of the input, used to scale it to the 0.1 mV units of
        pretraining. A number is used as the multiplier directly.
    pretrained : bool, default=True
        If True, start from the pretrained weights. If False, start from
        random weights (the random-weight control).
    weights_path : str or None, default=None
        Directory holding the pretrained weights (``model.safetensors`` or
        ``pytorch_model.bin``, as in ``braindecode/labram-pretrained`` on the
        Hugging Face Hub). Needs no network. If None, the weights are taken
        from the Hugging Face cache (``HF_HOME``) through ``huggingface_hub``.
    n_epochs : int, default=40
        Maximum number of epochs.
    early_stopping_patience : int, default=5
        Epochs without improvement in validation balanced accuracy before
        stopping.
    validation_size : float, default=0.25
        Fraction of training subjects (or, without ``train_groups``, training
        cases) held out for early stopping.
    train_groups : array-like or None, default=None
        Subject ID of each training case, in the order passed to ``fit``. If
        given, whole subjects are held out for validation.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Peak learning rate.
    weight_decay : float, default=0.05
        AdamW weight decay.
    device : str or None, default=None
        Torch device. If None, uses CUDA when available.
    verbose : bool, default=False
        If True, print training progress.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    network_ : torch.nn.Module
        The fine-tuned network: ``network_.encoder`` and ``network_.head``.
    preprocessor_ : NeuralBenchPreprocessor
        The fitted preprocessing, before unit scaling and padding.
    n_times_ : int
        Length of the network input, in samples at 200 Hz.
    ch_names_ : list of str
        The channel names used.
    best_epoch_ : int or None
        Zero-based epoch whose weights were kept.

    References
    ----------
    .. [1] Jiang, W.-B., Zhao, L.-M., and Lu, B.-L. (2024). Large Brain Model for
       Learning Generic Representations with Tremendous EEG Data in BCI.
       International Conference on Learning Representations (ICLR).
    """

    _NOTCH = 50.0
    _HF_NAME = "braindecode/labram-pretrained"

    _tags = {
        # braindecode's Labram cannot be pickled (lambda inside its MLP blocks)
        "cant_pickle": True,
    }

    def __init__(
        self,
        ch_names: list[str] | None = None,
        sfreq: float | None = None,
        input_units: str | float = "uV",
        pretrained: bool = True,
        weights_path: str | None = None,
        n_epochs: int = 40,
        early_stopping_patience: int = 5,
        validation_size: float = 0.25,
        train_groups=None,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.ch_names = ch_names
        super().__init__(
            sfreq=sfreq,
            input_units=input_units,
            pretrained=pretrained,
            weights_path=weights_path,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            validation_size=validation_size,
            train_groups=train_groups,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )

    def _fit(self, X, y):
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        n_channels = X.shape[1]
        if self.ch_names is not None:
            if len(self.ch_names) != n_channels:
                raise ValueError(
                    f"ch_names has {len(self.ch_names)} names but X has "
                    f"{n_channels} channels."
                )
            known = {name.upper() for name in LABRAM_CHANNEL_ORDER}
            unknown = [n for n in self.ch_names if str(n).upper() not in known]
            if unknown:
                raise ValueError(
                    f"ch_names {unknown} are not in LaBraM's electrode list "
                    "(braindecode.models.labram.LABRAM_CHANNEL_ORDER)."
                )
            self.ch_names_ = [str(n).upper() for n in self.ch_names]
        elif self.pretrained:
            raise ValueError("ch_names is required when pretrained=True.")
        else:
            self.ch_names_ = list(LABRAM_CHANNEL_ORDER[:n_channels])
        return super()._fit(X, y)

    def _build_encoder(self, n_channels, n_timepoints):
        """Build LaBraM with the pretrained checkpoint's geometry.

        Setting ``patch_embed[0].n_patchs`` makes braindecode use the first
        ``n_patches`` temporal embeddings, as in the original LaBraM code.
        """
        from braindecode.models import Labram
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        n_patches = n_timepoints // _PATCH
        if n_patches > _LABRAM_MAX_PATCHES:
            raise ValueError(
                f"LabramFineTuneClassifier supports at most {_LABRAM_MAX_PATCHES} "
                f"patches of 200 samples (16 s at 200 Hz), got {n_patches}."
            )
        encoder = Labram(
            n_times=3000,
            n_outputs=0,
            chs_info=[{"ch_name": name} for name in LABRAM_CHANNEL_ORDER],
        )
        encoder.patch_embed[0].n_patchs = n_patches
        # a plain attribute, so _encode (a static method) can read it
        encoder.aeon_ch_names = list(self.ch_names_)
        return encoder

    @staticmethod
    def _encode(encoder, x):
        # (batch, 1 + n_channels * n_patches, 200), the class token first, as
        # NeuralBench's LaBraM wrapper returns
        return encoder(x, ch_names=encoder.aeon_ch_names, return_all_tokens=True)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "sfreq": 100.0,
            "pretrained": False,
            "n_epochs": 1,
            "early_stopping_patience": 1,
            "validation_size": 0.3,
            "batch_size": 4,
            "random_state": 0,
        }
