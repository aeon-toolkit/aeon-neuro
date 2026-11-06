"""aeon compatible adaptor for braindecode models."""

from __future__ import annotations

__all__ = ["BraindecodeClassifier", "SUPPORTED_MODELS"]

import inspect
from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state

from aeon_neuro.classification.base import BaseEEGClassifier

# braindecode models tested with BraindecodeClassifier: each builds, fits and
# predicts one label per case with the default settings, given ``sfreq`` and,
# where marked, ``channel_names``. Other braindecode model names are accepted,
# but may need extra inputs or give outputs that are not one score per class.
SUPPORTED_MODELS = (
    # convolutional networks, mainly for motor imagery and general BCI decoding
    "ShallowFBCSPNet",
    "Deep4Net",
    "EEGNet",
    "EEGInceptionMI",
    "EEGITNet",
    "EEGNeX",
    "EEGSimpleConv",
    "EEGSym",  # needs channel_names
    "EEGTCNet",
    "SCCNet",
    "TIDNet",
    "SincShallowNet",
    # filter-bank networks
    "FBCNet",
    "FBMSNet",
    "FBLightConvNet",
    "IFNet",
    # convolution with attention or transformer layers
    "ATCNet",
    "EEGConformer",
    "CTNet",
    "MSVTNet",
    "AttentionBaseNet",
    "SSTDPN",
    "MSCFormer",
    "TMSANet",
    "TCFormer",
    # graph neural networks (DGCNN needs channel_names)
    "GCNsNet",
    "DGCNN",
    # other tasks: ERP, emotion, pathology, seizures, speech, sleep
    "EEGInceptionERP",
    "TSception",
    "EEGMiner",
    "SyncNet",
    "BDTCN",
    "SPARCNet",
    "BrainModule",
    "SleepStagerChambon2018",
    "ContraWR",
)

# Supported models that need electrode positions, given through channel_names.
_NEEDS_CHANNEL_NAMES = ("DGCNN", "EEGSym")

# Constructor arguments set by the adaptor, not through ``model_params``.
_RESERVED_PARAMS = ("n_chans", "n_outputs", "n_times", "sfreq", "chs_info")


class _BaseBraindecodeClassifier(BaseEEGClassifier):
    """Shared training and prediction code for braindecode based classifiers.

    Subclasses implement ``_build_network``, which also checks any parameters
    that depend on the data.

    Parameters
    ----------
    batch_size : int, default=64
        Mini-batch size used for training and inference.
    n_epochs : int, default=200
        Number of training epochs.
    lr : float, default=1e-3
        Learning rate.
    optimizer : {"adam", "sgd"}, default="adam"
        Optimiser used for training.
    weight_decay : float, default=0.0
        L2 penalty passed to the optimiser.
    device : str or None, default=None
        Torch device string. If None, uses "cuda" when available, otherwise "cpu".
    verbose : bool, default=False
        If True, print epoch losses during training.
    random_state : int or None, default=None
        Random seed.
    validation_size : float or None, default=None
        Optional inner validation fraction. With information, holds out entire
        subjects; without it, stratifies cases. None trains without validation.
    patience : int, default=5
        Stop after this many epochs without improved validation loss and restore
        the best weights. Used only when validation_size is set.
    validation_strategy : {"auto", "subject", "stratified"}, default="auto"
        Auto uses subjects with information and stratifies cases without it.
        Stratified explicitly permits within-person validation with metadata.
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
        batch_size: int = 64,
        n_epochs: int = 200,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
        validation_size: float | None = None,
        patience: int = 5,
        validation_strategy: str = "auto",
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.validation_size = validation_size
        self.patience = patience
        self.validation_strategy = validation_strategy

        super().__init__()

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the torch network for data of the given shape."""
        raise NotImplementedError("Subclasses must implement _build_network.")

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Train on fitting rows and optionally validate on a separate partition."""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32, order="C")
        y_int = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.training_indices_ = np.arange(len(X))
        self.validation_indices_ = np.array([], dtype=int)
        self.validation_history_ = []
        self.best_epoch_ = None
        if self.validation_size is not None:
            if (
                isinstance(self.patience, bool)
                or not isinstance(self.patience, Integral)
                or self.patience < 1
            ):
                raise ValueError("patience must be a positive integer")
            self.training_indices_, self.validation_indices_ = self.validation_split(
                y,
                validation_size=self.validation_size,
                random_state=seed,
                strategy=self.validation_strategy,
            )
        self.validation_policy_ = (
            "none"
            if self.validation_size is None
            else (
                "subject"
                if self.validation_strategy == "subject"
                or (self.validation_strategy == "auto" and self.info_ is not None)
                else "stratified cases"
            )
        )

        self.device_ = self._resolve_device()
        self.network_ = self._build_network(
            n_channels=X.shape[1], n_timepoints=X.shape[2]
        ).to(self.device_)
        self.optimizer_ = self._build_optimizer(self.network_.parameters())
        self.loss_ = nn.CrossEntropyLoss()

        train = self.training_indices_
        dataset = TensorDataset(
            torch.from_numpy(X[train]), torch.from_numpy(y_int[train])
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            # Preserve the shared adapter's training batch policy.
            drop_last=True,
        )

        self.history_ = []
        self.network_.train()
        best_loss, best_state, stale = float("inf"), None, 0

        for epoch in range(self.n_epochs):
            epoch_loss, n_seen = 0.0, 0
            self.network_.train()

            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                self.optimizer_.zero_grad(set_to_none=True)
                logits = self._coerce_logits(self.network_(xb))
                loss = self.loss_(logits, yb)
                loss.backward()
                self.optimizer_.step()

                epoch_loss += loss.item() * xb.shape[0]
                n_seen += len(xb)

            epoch_loss /= n_seen
            self.history_.append(epoch_loss)

            if self.validation_size is not None:
                validation = TensorDataset(
                    torch.from_numpy(X[self.validation_indices_]),
                    torch.from_numpy(y_int[self.validation_indices_]),
                )
                validation_loader = DataLoader(
                    validation,
                    batch_size=min(self.batch_size, len(validation)),
                    shuffle=False,
                    drop_last=False,
                )
                self.network_.eval()
                total = 0.0
                with torch.no_grad():
                    for xb, yb in validation_loader:
                        logits = self._coerce_logits(self.network_(xb.to(self.device_)))
                        loss = self.loss_(logits, yb.to(self.device_))
                        total += loss.item() * len(xb)
                validation_loss = total / len(validation)
                if not np.isfinite(validation_loss):
                    raise ValueError("Validation loss is not finite")
                self.validation_history_.append(validation_loss)
                if validation_loss < best_loss:
                    best_loss, stale = validation_loss, 0
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.network_.state_dict().items()
                    }
                    self.best_epoch_ = epoch + 1
                else:
                    stale += 1
                if stale >= self.patience:
                    break

            if self.verbose:
                print(  # noqa: T201
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"training loss: {epoch_loss:.6f}"
                )

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.eval()
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32, order="C")
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=False,
            drop_last=False,
        )

        probs = np.zeros((X.shape[0], self.n_classes_), dtype=np.float32)

        self.network_.eval()
        start = 0

        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_)
                logits = self._coerce_logits(self.network_(xb))
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()

                end = start + batch_probs.shape[0]
                probs[start:end] = batch_probs
                start = end

        return probs

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        probs = self._predict_proba(X)
        rng = check_random_state(self.random_state)

        return np.asarray(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def summary(self):
        """Return the training losses from fit.

        Matches ``summary`` of aeon's deep learning classifiers.

        Returns
        -------
        history : dict or None
            ``{"loss": [...]}`` with the mean training loss for each epoch, or
            None if the classifier has not been fitted.
        """
        history = getattr(self, "history_", None)
        return {"loss": list(history)} if history is not None else None

    def __getstate__(self):
        """Pickle the network as its weights.

        Some braindecode layers (e.g. EEGNet's max-norm constraint) use torch
        parametrizations, which torch can only save through ``state_dict``.
        """
        state = self.__dict__.copy()
        network = state.pop("network_", None)
        if network is not None:
            state["_network_state"] = {
                key: value.detach().cpu() for key, value in network.state_dict().items()
            }
            # the optimiser refers to the network's parameters; it is only
            # needed during fit
            state["optimizer_"] = None
        return state

    def __setstate__(self, state):
        """Rebuild the network from its pickled weights."""
        network_state = state.pop("_network_state", None)
        self.__dict__.update(state)
        if network_state is not None:
            import torch

            if self.device_.type == "cuda" and not torch.cuda.is_available():
                self.device_ = torch.device("cpu")
            self.network_ = self._build_network(
                n_channels=self.metadata_["n_channels"],
                n_timepoints=self.metadata_["n_timepoints"],
            )
            self.network_.load_state_dict(network_state)
            self.network_.to(self.device_).eval()

    def _build_optimizer(self, parameters):
        """Construct the torch optimiser."""
        import torch

        optimizer = self.optimizer.lower()
        if optimizer == "adam":
            return torch.optim.Adam(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        if optimizer == "sgd":
            return torch.optim.SGD(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )

        raise ValueError(
            f"Unknown optimizer='{self.optimizer}'. "
            "Supported values are {'adam', 'sgd'}."
        )

    def _resolve_device(self):
        """Resolve the torch device to use."""
        import torch

        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is not available.")

        return device

    def _coerce_logits(self, logits):
        """Coerce model output to shape (batch_size, n_classes).

        Trailing singleton dimensions are removed. Any other output shape, such
        as one score per time point or an embedding, raises a ValueError because
        it is not one score per class.
        """
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        while logits.ndim > 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        if logits.ndim != 2 or logits.shape[1] != self.n_classes_:
            raise ValueError(
                f"The network output has shape {tuple(logits.shape)}, but a "
                f"classifier needs one score per class, shape (n_cases, "
                f"{self.n_classes_}). This model does not produce a single "
                f"prediction per case, so it cannot be used as a classifier."
            )

        return logits


class BraindecodeClassifier(_BaseBraindecodeClassifier):
    """aeon classifier for any braindecode model.

    Builds the braindecode model named by ``model`` for the shape of the
    training data and trains it with a standard cross-entropy loop. Expects
    equal-length collections of shape (n_cases, n_channels, n_timepoints).

    The network is built for the number of channels and the series length seen
    in ``fit``, so ``predict`` needs data with the same shape; a different shape
    raises a ``ValueError``.

    ``SUPPORTED_MODELS`` lists the braindecode models tested with this adaptor.
    Each builds, fits and predicts with its default settings:

    - **Convolutional, motor imagery and general BCI:** ``"ShallowFBCSPNet"``
      and ``"Deep4Net"`` (Schirrmeister et al., 2017), ``"EEGNet"`` (Lawhern
      et al., 2018), ``"EEGInceptionMI"`` (Zhang et al., 2021), ``"EEGITNet"``
      (Salami et al., 2022), ``"EEGNeX"`` (Chen et al., 2024),
      ``"EEGSimpleConv"`` (El Ouahidi et al., 2023), ``"EEGSym"``
      (Pérez-Velasco et al., 2022; needs ``channel_names``), ``"EEGTCNet"``
      (Ingolfsson et al., 2020), ``"SCCNet"`` (Wei et al., 2019), ``"TIDNet"``
      (Kostas and Rudzicz, 2020), ``"SincShallowNet"`` (Borra et al., 2020).
    - **Filter-bank:** ``"FBCNet"`` (Mane et al., 2021), ``"FBMSNet"`` (Liu et
      al., 2022), ``"FBLightConvNet"`` (Ma et al., 2023), ``"IFNet"`` (Wang et
      al., 2023).
    - **Convolution with attention or transformers:** ``"ATCNet"`` (Altaheri
      et al., 2023), ``"EEGConformer"`` (Song et al., 2023), ``"CTNet"`` (Zhao
      et al., 2024), ``"MSVTNet"`` (Liu et al., 2024), ``"AttentionBaseNet"``
      (Wimpff et al., 2023), ``"SSTDPN"`` (Han et al., 2025), ``"MSCFormer"``
      (Zhao et al., 2025), ``"TMSANet"`` (Zhao and Zhu, 2025), ``"TCFormer"``
      (Altaheri et al., 2025).
    - **Graph networks:** ``"GCNsNet"`` (Hou et al., 2022), ``"DGCNN"`` (Song et
      al., 2018; needs ``channel_names``).
    - **Other tasks:** ``"EEGInceptionERP"`` (ERP; Santamaría-Vázquez et al.,
      2020), ``"TSception"`` (emotion; Ding et al., 2020), ``"EEGMiner"``
      (emotion; Ludwig et al., 2024), ``"SyncNet"`` (Li et al., 2017),
      ``"BDTCN"`` (pathology; Gemein et al., 2020), ``"SPARCNet"`` (seizures;
      Jing et al., 2023), ``"BrainModule"`` (speech; Défossez et al., 2023),
      ``"SleepStagerChambon2018"`` (sleep; Chambon et al., 2018),
      ``"ContraWR"`` (sleep; Yang et al., 2023).

    Full references are in the braindecode documentation for each model. Most
    models size their layers for a particular sampling rate, so pass ``sfreq``.

    braindecode names the model inputs differently from aeon. The adaptor sets
    them from the training data and the arguments below:

    =====================  ========================================================
    braindecode argument   Set from
    =====================  ========================================================
    ``n_chans``            number of channels, ``n_channels`` (``X.shape[1]``)
    ``n_times``            series length, ``n_timepoints`` (``X.shape[2]``)
    ``n_outputs``          number of classes in ``y`` (``n_classes_``)
    ``sfreq``              aligned information or the ``sfreq`` argument
    ``chs_info``           aligned channel names, with explicit template positions
    =====================  ========================================================

    Other braindecode model names, or a braindecode model class, are accepted
    but not tested. Foundation models (e.g. ``"Labram"``, ``"REVE"``) build and
    train from random weights here; loading pretrained weights is not
    supported yet. Models that produce embeddings, sequences or per time point
    outputs raise a ``ValueError`` because they do not give one prediction per
    case.

    Parameters
    ----------
    model : str or type, default="EEGNet"
        Name of a class in ``braindecode.models``, e.g. ``"ShallowFBCSPNet"``,
        or a braindecode model class.
    model_params : dict or None, default=None
        Extra keyword arguments for the model constructor, e.g.
        ``{"drop_prob": 0.25}``. The inputs in the table above (``n_chans``,
        ``n_times``, ``n_outputs``, ``sfreq`` and ``chs_info``) are set by the
        adaptor, so they cannot be given here.
    sfreq : float or None, default=None
        Sampling frequency of the data in Hz. Passed to models that accept it.
        Some models (e.g. FBCNet, IFNet, TSception) need it to size their
        filters; for others braindecode uses a default.
    channel_names : list of str or None, default=None
        Channel names in the order of the channels in X, using standard 10-05
        names. Models that take electrode positions (``chs_info``) get the
        positions of the ``standard_1005`` montage for these names.
    batch_size : int, default=64
        Mini-batch size used for training and inference.
    n_epochs : int, default=200
        Maximum number of training epochs. With validation_size=None every
        epoch is run. 200 is a compromise between the
        several hundred epochs typical in EEG deep learning papers and the
        training time on small EEG datasets; tune it for each problem.
    lr : float, default=1e-3
        Learning rate.
    optimizer : {"adam", "sgd"}, default="adam"
        Optimiser used for training.
    weight_decay : float, default=0.0
        L2 penalty passed to the optimiser.
    device : str or None, default=None
        Torch device string. If None, uses "cuda" when available, otherwise "cpu".
    verbose : bool, default=False
        If True, print epoch losses during training.
    random_state : int or None, default=None
        Random seed.
    validation_size : float or None, default=None
        Optional inner validation fraction. With information, holds out entire
        subjects; without it, stratifies cases. None trains without validation.
    patience : int, default=5
        Stop after this many epochs without improved validation loss and restore
        the best weights. Used only when validation_size is set.
    validation_strategy : {"auto", "subject", "stratified"}, default="auto"
        Auto uses subjects with information and stratifies cases without it.
        Stratified explicitly permits within-person validation with metadata.

    Attributes
    ----------
    network_ : torch.nn.Module
        The trained braindecode network.
    history_ : list of float
        Mean training loss for each epoch.

    Notes
    -----
    Optional ``info`` in fit/predict supplies current rate and ordered channel
    names without changing constructor parameters. Explicit conflicting
    parameters raise. Information-derived names alone do not assert measured
    positions. For spatial models requiring coordinates, or when channel_names
    is explicitly set, positions use the standard_1005 template, never measured
    source electrodes. ``channel_geometry_`` records names_only or standard_1005
    when the model accepts chs_info. Unknown template names raise.

    Optional validation holds out subjects with information, or stratifies cases
    without it. An explicit stratified strategy supports within-person questions.
    Training retains the shared incomplete-batch policy; validation and prediction
    include every row. Pickling restores the selected weights and information.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from aeon_neuro.classification.deep_learning import BraindecodeClassifier
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=4, n_timepoints=200,
    ...                              random_state=0)
    >>> clf = BraindecodeClassifier(model="ShallowFBCSPNet", n_epochs=1,
    ...                             random_state=0)
    >>> clf.fit(X, y)
    BraindecodeClassifier(...)
    >>> clf.predict_proba(X).shape
    (10, 2)
    """

    def __init__(
        self,
        model: str | type = "EEGNet",
        model_params: dict | None = None,
        sfreq: float | None = None,
        channel_names: list[str] | None = None,
        batch_size: int = 64,
        n_epochs: int = 200,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
        validation_size: float | None = None,
        patience: int = 5,
        validation_strategy: str = "auto",
    ):
        self.model = model
        self.model_params = model_params
        self.sfreq = sfreq
        self.channel_names = channel_names

        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            random_state=random_state,
            validation_size=validation_size,
            patience=patience,
            validation_strategy=validation_strategy,
        )

    def _resolve_model(self):
        """Return the braindecode model class named by ``model``."""
        from torch import nn

        if isinstance(self.model, str):
            import braindecode.models

            model_class = getattr(braindecode.models, self.model, None)
            is_model = inspect.isclass(model_class) and issubclass(
                model_class, nn.Module
            )
            if not is_model:
                raise ValueError(
                    f"Unknown braindecode model {self.model!r}. Use the name of a "
                    f"class in braindecode.models, e.g. one of {SUPPORTED_MODELS}."
                )
            return model_class

        if inspect.isclass(self.model) and issubclass(self.model, nn.Module):
            return self.model

        raise ValueError(
            "model must be the name of a braindecode model or a torch.nn.Module "
            f"subclass, got {self.model!r}."
        )

    def _channel_info(self):
        """Return braindecode ``chs_info`` for ``channel_names``."""
        import mne

        info = mne.create_info(
            list(self.channel_names_), sfreq=self.sfreq_ or 1.0, ch_types="eeg"
        )
        model_name = self.model if isinstance(self.model, str) else self.model.__name__
        # Names carried by data do not establish measured positions. Preserve
        # the existing template policy only when explicitly requested through
        # channel_names or required by a spatial model.
        if self.channel_names is None and model_name not in _NEEDS_CHANNEL_NAMES:
            self.channel_geometry_ = "names_only"
            return info["chs"]
        self.channel_geometry_ = "standard_1005"
        try:
            info.set_montage(mne.channels.make_standard_montage("standard_1005"))
        except ValueError as exc:
            raise ValueError(
                "Some channel_names are not in the standard_1005 montage, so their "
                f"electrode positions are unknown: {exc}"
            ) from exc
        return info["chs"]

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the braindecode model for data of the given shape."""
        if self.model_params is not None:
            if not isinstance(self.model_params, dict):
                raise ValueError(
                    f"model_params must be a dict or None, got {self.model_params!r}."
                )
            reserved = sorted(set(self.model_params) & set(_RESERVED_PARAMS))
            if reserved:
                raise ValueError(
                    f"model_params cannot set {reserved}. The adaptor sets "
                    "n_chans to the number of channels in X, n_times to the "
                    "series length, n_outputs to the number of classes in y, and "
                    "sfreq and chs_info from the sfreq and channel_names "
                    "arguments."
                )
        model_class = self._resolve_model()
        model_name = model_class.__name__
        if self.channel_names_ is None and model_name in _NEEDS_CHANNEL_NAMES:
            raise ValueError(
                f"{self.model} needs electrode positions; pass channel_names with "
                "the standard 10-05 name of each channel in X."
            )
        if self.channel_names_ is not None and len(self.channel_names_) != n_channels:
            raise ValueError(
                f"channel_names has {len(self.channel_names_)} names but X has "
                f"{n_channels} channels."
            )

        accepted = inspect.signature(model_class.__init__).parameters

        kwargs = {
            "n_chans": n_channels,
            "n_outputs": self.n_classes_,
            "n_times": n_timepoints,
        }
        if self.sfreq_ is not None and "sfreq" in accepted:
            kwargs["sfreq"] = self.sfreq_
        self.channel_geometry_ = None
        if self.channel_names_ is not None and "chs_info" in accepted:
            kwargs["chs_info"] = self._channel_info()
        kwargs.update(self.model_params or {})

        return model_class(**kwargs)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        small = {"batch_size": 4, "n_epochs": 1, "lr": 1e-3, "random_state": 0}
        return [
            {
                "model": "ShallowFBCSPNet",
                "model_params": {
                    "n_filters_time": 4,
                    "filter_time_length": 5,
                    "n_filters_spat": 4,
                    "pool_time_length": 5,
                    "pool_time_stride": 2,
                },
                **small,
            },
            {
                "model": "EEGNet",
                "model_params": {
                    "kernel_length": 4,
                    "depthwise_kernel_length": 4,
                    "pool1_kernel_size": 2,
                    "pool2_kernel_size": 2,
                },
                **small,
            },
        ]
