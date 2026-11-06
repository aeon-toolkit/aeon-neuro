"""aeon compatible adaptor for braindecode models."""

from __future__ import annotations

__all__ = ["BraindecodeClassifier", "SUPPORTED_MODELS"]

import inspect

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state

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


class _BaseBraindecodeClassifier(BaseClassifier):
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
    lr_scheduler : {None, "onecycle"}, default=None
        Learning rate schedule. ``"onecycle"`` uses torch's ``OneCycleLR`` with
        ``max_lr=lr``, 10% warmup and cosine annealing, stepped every batch and
        planned over all ``n_epochs`` (early stopping may end it sooner).
    gradient_clip : float or None, default=None
        If set, clip the gradient norm to this value before each step.
    class_weight : {None, "balanced"}, default=None
        If ``"balanced"``, weight the loss by inverse class frequency in the
        training cases, as sklearn's ``compute_class_weight("balanced")``.
    label_smoothing : float, default=0.0
        Label smoothing passed to ``torch.nn.CrossEntropyLoss``.
    validation_size : float or None, default=None
        If set, hold out this fraction of the training cases, stratified by
        class, as a validation set. Its balanced accuracy is measured after
        every epoch and the weights from the best epoch are kept. The split is
        random over cases, not by subject.
    early_stopping_patience : int or None, default=None
        With ``validation_size``, stop when validation balanced accuracy has
        not improved for this many epochs. If None, train all ``n_epochs`` and
        keep the best epoch.
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
        lr_scheduler: str | None = None,
        gradient_clip: float | None = None,
        class_weight: str | None = None,
        label_smoothing: float = 0.0,
        validation_size: float | None = None,
        early_stopping_patience: int | None = None,
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.lr_scheduler = lr_scheduler
        self.gradient_clip = gradient_clip
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.validation_size = validation_size
        self.early_stopping_patience = early_stopping_patience

        super().__init__()

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the torch network for data of the given shape."""
        raise NotImplementedError("Subclasses must implement _build_network.")

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Build the network and train it on X and y."""
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

        X_val = y_val = None
        if self.validation_size is not None:
            X, X_val, y_int, y_val = self._validation_split(X, y_int, rng)

        self.device_ = self._resolve_device()
        self.network_ = self._build_network(
            n_channels=X.shape[1], n_timepoints=X.shape[2]
        ).to(self.device_)
        self.optimizer_ = self._build_optimizer(self.network_.parameters())
        self.loss_ = nn.CrossEntropyLoss(
            weight=self._loss_class_weights(y_int),
            label_smoothing=self.label_smoothing,
        )

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_int))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            # As in braindecode's EEGClassifier: drop the incomplete last batch
            # in training, which avoids a noisy update from a tiny batch (and
            # batch norm failing on a batch of one case). Shuffling means a
            # different few cases sit out each epoch.
            drop_last=True,
        )

        self.scheduler_ = self._build_scheduler(len(loader))

        self.history_ = []
        self.validation_history_ = []
        self.best_epoch_ = None
        best_score, best_state, stale = -np.inf, None, 0

        for epoch in range(self.n_epochs):
            self.network_.train()
            epoch_loss = 0.0
            n_seen = 0

            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                self.optimizer_.zero_grad(set_to_none=True)
                logits = self._coerce_logits(self.network_(xb))
                loss = self.loss_(logits, yb)
                loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.network_.parameters(), self.gradient_clip
                    )
                self.optimizer_.step()
                if self.scheduler_ is not None:
                    self.scheduler_.step()

                epoch_loss += loss.item() * xb.shape[0]
                n_seen += xb.shape[0]

            epoch_loss /= n_seen
            self.history_.append(epoch_loss)

            message = f"Epoch {epoch + 1}/{self.n_epochs} - training loss: "
            message += f"{epoch_loss:.6f}"
            stop = False
            if X_val is not None:
                score = self._balanced_accuracy(X_val, y_val)
                self.validation_history_.append(score)
                message += f" - validation balanced accuracy: {score:.4f}"
                if score > best_score:
                    best_score, stale = score, 0
                    self.best_epoch_ = epoch
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.network_.state_dict().items()
                    }
                else:
                    stale += 1
                    stop = (
                        self.early_stopping_patience is not None
                        and stale >= self.early_stopping_patience
                    )

            if self.verbose:
                print(message)  # noqa: T201
            if stop:
                break

        if best_state is not None:
            self.network_.load_state_dict(best_state)

        # Leave the network in eval mode so predict does not mutate fitted state.
        self.network_.eval()

        return self

    def _validation_split(self, X, y_int, rng):
        """Hold out ``validation_size`` of the cases, stratified if possible."""
        from sklearn.model_selection import train_test_split

        if not 0 < self.validation_size < 1:
            raise ValueError(
                f"validation_size must be between 0 and 1, got {self.validation_size}."
            )
        seed = int(rng.randint(np.iinfo(np.int32).max))
        try:
            return train_test_split(
                X,
                y_int,
                test_size=self.validation_size,
                random_state=seed,
                stratify=y_int,
            )
        except ValueError:
            # too few cases of some class to stratify; split at random instead
            return train_test_split(
                X, y_int, test_size=self.validation_size, random_state=seed
            )

    def _loss_class_weights(self, y_int):
        """Return the cross-entropy class weights, or None."""
        import torch

        if self.class_weight is None:
            return None
        if self.class_weight != "balanced":
            raise ValueError(
                f"class_weight must be None or 'balanced', got {self.class_weight!r}."
            )
        counts = np.bincount(y_int, minlength=self.n_classes_).astype(np.float64)
        weights = np.ones(self.n_classes_)
        present = counts > 0
        # sklearn's "balanced": n_cases / (n_classes * count), over the classes
        # present in the training cases; absent classes keep weight one
        weights[present] = counts.sum() / (present.sum() * counts[present])
        return torch.tensor(weights, dtype=torch.float32, device=self.device_)

    def _build_scheduler(self, steps_per_epoch):
        """Construct the learning rate scheduler, or None."""
        import torch

        if self.lr_scheduler is None:
            return None
        if self.lr_scheduler != "onecycle":
            raise ValueError(
                f"lr_scheduler must be None or 'onecycle', got {self.lr_scheduler!r}."
            )
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer_,
            max_lr=self.lr,
            total_steps=self.n_epochs * steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    def _balanced_accuracy(self, X, y_int):
        """Balanced accuracy of the current network on X."""
        from sklearn.metrics import balanced_accuracy_score

        probs = self._predict_proba(X)
        return balanced_accuracy_score(y_int, probs.argmax(axis=1))

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
            # the optimiser and scheduler refer to the network's parameters;
            # they are only needed during fit
            state["optimizer_"] = None
            state["scheduler_"] = None
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
        if optimizer == "adamw":
            return torch.optim.AdamW(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        if optimizer == "sgd":
            return torch.optim.SGD(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )

        raise ValueError(
            f"Unknown optimizer='{self.optimizer}'. "
            "Supported values are {'adam', 'adamw', 'sgd'}."
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
    ``sfreq``              the ``sfreq`` argument
    ``chs_info``           the ``channel_names`` argument, with montage positions
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
        Number of training epochs. By default there is no early stopping or
        learning rate schedule, so every epoch is run. 200 is a compromise
        between the several hundred epochs typical in EEG deep learning papers
        and the training time on small EEG datasets; tune it for each problem.
    lr : float, default=1e-3
        Learning rate (the peak rate with ``lr_scheduler="onecycle"``).
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
    lr_scheduler : {None, "onecycle"}, default=None
        Learning rate schedule. ``"onecycle"`` uses torch's ``OneCycleLR`` with
        ``max_lr=lr``, 10% warmup and cosine annealing, stepped every batch and
        planned over all ``n_epochs`` (early stopping may end it sooner).
    gradient_clip : float or None, default=None
        If set, clip the gradient norm to this value before each step.
    class_weight : {None, "balanced"}, default=None
        If ``"balanced"``, weight the loss by inverse class frequency in the
        training cases.
    label_smoothing : float, default=0.0
        Label smoothing passed to ``torch.nn.CrossEntropyLoss``.
    validation_size : float or None, default=None
        If set, hold out this fraction of the training cases, stratified by
        class, as a validation set. Its balanced accuracy is measured after
        every epoch and the weights from the best epoch are kept. The split is
        random over cases, not by subject.
    early_stopping_patience : int or None, default=None
        With ``validation_size``, stop when validation balanced accuracy has
        not improved for this many epochs. If None, train all ``n_epochs`` and
        keep the best epoch.

    Attributes
    ----------
    network_ : torch.nn.Module
        The trained braindecode network.
    history_ : list of float
        Mean training loss for each epoch.
    validation_history_ : list of float
        Validation balanced accuracy for each epoch; empty without
        ``validation_size``.
    best_epoch_ : int or None
        Zero-based epoch whose weights were kept, or None without
        ``validation_size``.

    Notes
    -----
    ``sfreq`` and ``channel_names`` describe the recordings rather than the
    model. They are plain parameters for now. Once the sidecar information
    class from issue #283 is available, they will be taken from it instead.

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
        lr_scheduler: str | None = None,
        gradient_clip: float | None = None,
        class_weight: str | None = None,
        label_smoothing: float = 0.0,
        validation_size: float | None = None,
        early_stopping_patience: int | None = None,
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
            lr_scheduler=lr_scheduler,
            gradient_clip=gradient_clip,
            class_weight=class_weight,
            label_smoothing=label_smoothing,
            validation_size=validation_size,
            early_stopping_patience=early_stopping_patience,
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
            list(self.channel_names), sfreq=self.sfreq or 1.0, ch_types="eeg"
        )
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
        if self.channel_names is None and self.model in _NEEDS_CHANNEL_NAMES:
            raise ValueError(
                f"{self.model} needs electrode positions; pass channel_names with "
                "the standard 10-05 name of each channel in X."
            )
        if self.channel_names is not None and len(self.channel_names) != n_channels:
            raise ValueError(
                f"channel_names has {len(self.channel_names)} names but X has "
                f"{n_channels} channels."
            )

        model_class = self._resolve_model()
        accepted = inspect.signature(model_class.__init__).parameters

        kwargs = {
            "n_chans": n_channels,
            "n_outputs": self.n_classes_,
            "n_times": n_timepoints,
        }
        if self.sfreq is not None and "sfreq" in accepted:
            kwargs["sfreq"] = self.sfreq
        if self.channel_names is not None and "chs_info" in accepted:
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
                "n_epochs": 2,
                "optimizer": "adamw",
                "lr_scheduler": "onecycle",
                "gradient_clip": 1.0,
                "class_weight": "balanced",
                "label_smoothing": 0.1,
                "validation_size": 0.3,
                "early_stopping_patience": 1,
            },
        ]
