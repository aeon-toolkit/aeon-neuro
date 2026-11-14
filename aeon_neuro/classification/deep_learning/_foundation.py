"""Shared base class for the pretrained foundation model classifiers."""

from __future__ import annotations

__all__ = ["_BaseFoundationClassifier"]

import re
from math import cos, pi

import numpy as np
from sklearn.utils import check_random_state

from aeon_neuro.classification.deep_learning._braindecode import (
    _BaseBraindecodeClassifier,
)


class _BaseFoundationClassifier(_BaseBraindecodeClassifier):
    """Base class fine-tuning a pretrained Braindecode foundation model.

    Extends ``_BaseBraindecodeClassifier`` with the training options used to
    fine-tune pretrained models. Subclasses implement ``_build_network`` and may
    override ``_pad`` to reshape the input to what the network accepts. The
    network is trained with cross-entropy loss. Its classification head must be
    the submodule ``final_layer`` (as in Braindecode), taking the encoder output
    as its only input.

    Unlike the base class, the incomplete last batch is kept in training: these
    models use layer norm, so a batch of one case is not a problem.

    By default the whole network is trained, which for pretrained models is full
    fine-tuning. With ``freeze_encoder=True`` it is linear probing instead: the
    encoder is frozen in eval mode, its outputs are computed once, and only the
    head is trained on them.

    With ``validation_data``, the network is scored on it after every epoch, and
    the weights of the epoch with the best balanced accuracy are
    kept (the earliest, on ties) instead of those after the last epoch.

    Parameters
    ----------
    batch_size : int, default=64
        Mini-batch size used for training and inference.
    n_epochs : int, default=20
        Number of training epochs.
    lr : float, default=1e-3
        Learning rate.
    optimizer : {"adam", "adamw", "sgd"}, default="adam"
        Optimiser used for training.
    weight_decay : float, default=0.0
        Weight decay passed to the optimiser.
    freeze_encoder : bool, default=False
        If True, train only the head on the frozen encoder's output (linear
        probing). If False, train the whole network.
    head_lr : float or None, default=None
        Learning rate of the head. If None, ``lr`` is used.
    layer_decay : float, default=1.0
        Layer-wise learning rate decay: the learning rate of transformer block
        ``i`` of ``L`` (parameters named ``blocks.i.`` or ``layers.i.``) is
        multiplied by ``layer_decay ** (L - i)``, and that of the parameters
        before the first block (the embeddings) by ``layer_decay ** (L + 1)``.
    lr_schedule : {None, "cosine"}, default=None
        If "cosine", the learning rate warms up linearly over ``warmup_epochs``,
        then decays to zero with a cosine curve, updated every batch. If None, it
        is constant.
    warmup_epochs : int, default=0
        Number of warmup epochs of the "cosine" schedule.
    clip_grad : float or None, default=None
        If set, clip the gradient norm to this value.
    label_smoothing : float, default=0.0
        Label smoothing of the cross-entropy loss.
    validation_data : tuple of (np.ndarray, np.ndarray) or None, default=None
        Validation series, shaped like the training series, and their labels,
        used to choose the epoch whose weights are kept. It should not share
        subjects with the training data.
    device : str or None, default=None
        Torch device string. If None, uses "cuda" when available, otherwise "cpu".
    verbose : bool, default=False
        If True, print epoch losses during training.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    history_ : list of float
        Mean training loss of every epoch.
    validation_scores_ : list of float
        Balanced accuracy on ``validation_data`` after every epoch, or empty.
    best_epoch_ : int
        Epoch (1-based) whose weights were kept.
    """

    def __init__(
        self,
        batch_size: int = 64,
        n_epochs: int = 20,
        lr: float = 1e-3,
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
        self.freeze_encoder = freeze_encoder
        self.head_lr = head_lr
        self.layer_decay = layer_decay
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.clip_grad = clip_grad
        self.label_smoothing = label_smoothing
        self.validation_data = validation_data

        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the network."""
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset

        X = self._pad(X)
        y_int = torch.from_numpy(
            np.asarray([self._class_dictionary[label] for label in y], dtype=np.int64)
        )

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device_ = self._resolve_device()
        self.network_ = self._build_network(
            n_channels=X.shape[1], n_timepoints=X.shape[2]
        ).to(self.device_)
        self.loss_ = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.history_ = []
        self.validation_scores_ = []
        self._best_state = None

        X_val = y_val = None
        if self.validation_data is not None:
            X_val = self._pad(self.validation_data[0])
            y_val = np.asarray(
                [self._class_dictionary[label] for label in self.validation_data[1]]
            )

        if self.freeze_encoder:
            module = self.network_.final_layer
            X_t = self._head_inputs(X)
            val = None if X_val is None else (self._head_inputs(X_val), y_val)
        else:
            module = self.network_
            X_t = torch.from_numpy(X)
            val = None if X_val is None else (torch.from_numpy(X_val), y_val)
        module.train()
        self._train(
            module,
            TensorDataset(X_t, y_int),
            self._param_groups(head_only=self.freeze_encoder),
            self.n_epochs,
            val,
        )

        self.best_epoch_ = len(self.history_)
        if self._best_state is not None:
            self.best_epoch_ = 1 + int(np.argmax(self.validation_scores_))
            self.network_.load_state_dict(self._best_state)
        del self._best_state
        self.network_.eval()
        return self

    def _train(self, module, dataset, param_groups, n_epochs, val=None):
        """Train ``module`` on ``dataset`` for ``n_epochs``.

        ``val`` is a pair of module inputs and integer labels to score after
        every epoch, keeping a copy of the best weights.
        """
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )
        self.optimizer_ = self._build_optimizer(param_groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_, self._lr_factor(n_epochs * len(loader), len(loader))
        )
        parameters = [p for group in param_groups for p in group["params"]]

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                self.optimizer_.zero_grad(set_to_none=True)
                logits = self._coerce_logits(module(xb))
                loss = self.loss_(logits, yb)
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.clip_grad)
                self.optimizer_.step()
                scheduler.step()

                epoch_loss += loss.item() * xb.shape[0]

            epoch_loss /= len(dataset)
            self.history_.append(epoch_loss)
            message = f"Epoch {epoch + 1}/{n_epochs} - training loss: {epoch_loss:.6f}"

            if val is not None:
                score = self._validation_score(module, *val)
                if score > max(self.validation_scores_, default=-np.inf):
                    self._best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.network_.state_dict().items()
                    }
                self.validation_scores_.append(score)
                module.train()
                message += f" - validation balanced accuracy: {score:.4f}"

            if self.verbose:
                print(message)  # noqa: T201

    def _validation_score(self, module, X, y):
        """Return the balanced accuracy of ``module`` on inputs X."""
        import torch
        from sklearn.metrics import balanced_accuracy_score
        from torch.utils.data import DataLoader, TensorDataset

        module.eval()
        with torch.no_grad():
            preds = [
                self._coerce_logits(module(xb.to(self.device_))).argmax(1).cpu()
                for (xb,) in DataLoader(TensorDataset(X), batch_size=self.batch_size)
            ]
        return balanced_accuracy_score(y, torch.cat(preds).numpy())

    def _head_inputs(self, X: np.ndarray):
        """Return the input of ``final_layer`` for X, from the frozen encoder."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        inputs = []
        hook = self.network_.final_layer.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].cpu())
        )
        self.network_.eval()
        try:
            with torch.no_grad():
                loader = DataLoader(
                    TensorDataset(torch.from_numpy(X)), batch_size=self.batch_size
                )
                for (xb,) in loader:
                    self.network_(xb.to(self.device_))
        finally:
            hook.remove()
        return torch.cat(inputs)

    def _param_groups(self, head_only: bool):
        """Optimiser parameter groups with per-layer learning rates.

        With ``head_only``, the encoder is frozen and only the head is returned.
        """
        named = list(self.network_.named_parameters())
        blocks = [re.search(r"(?:blocks|layers)\.(\d+)\.", n) for n, _ in named]
        n_layers = 1 + max((int(b.group(1)) for b in blocks if b), default=-1)

        groups = {}
        before_blocks = True
        for (name, param), block in zip(named, blocks):
            before_blocks = before_blocks and block is None
            is_head = name.startswith("final_layer.")
            param.requires_grad_(is_head or not head_only)
            if not param.requires_grad:
                continue
            if is_head and self.head_lr is not None:
                lr = self.head_lr
            elif block:
                lr = self.lr * self.layer_decay ** (n_layers - int(block.group(1)))
            elif before_blocks:
                lr = self.lr * self.layer_decay ** (n_layers + 1)
            else:  # final norm, projections and head
                lr = self.lr
            groups.setdefault(lr, []).append(param)

        return [{"params": params, "lr": lr} for lr, params in groups.items()]

    def _lr_factor(self, n_steps: int, steps_per_epoch: int):
        """Return the learning rate multiplier as a function of the step."""
        if self.lr_schedule is None:
            return lambda step: 1.0
        if self.lr_schedule != "cosine":
            raise ValueError(
                f"Unknown lr_schedule='{self.lr_schedule}'. "
                "Supported values are {None, 'cosine'}."
            )
        n_warmup = min(self.warmup_epochs * steps_per_epoch, n_steps)

        def factor(step):
            if step < n_warmup:
                return (step + 1) / n_warmup
            progress = (step - n_warmup) / max(n_steps - n_warmup, 1)
            return 0.5 * (1 + cos(pi * min(progress, 1.0)))

        return factor
