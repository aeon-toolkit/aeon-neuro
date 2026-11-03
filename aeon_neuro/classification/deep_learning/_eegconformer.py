"""aeon compatible wrapper for Braindecode EEGConformer classifier."""

from __future__ import annotations

__all__ = ["EEGConformerClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state


class EEGConformerClassifier(BaseClassifier):
    """aeon wrapper around Braindecode's EEGConformer model.

    This classifier is for equal-length multivariate time series and expects aeon's
    standard collection shape:
    (n_cases, n_channels, n_timepoints).

    Series must have at least ``filter_time_length + pool_time_length - 1``
    timepoints (99 with the default parameters).

    Parameters
    ----------
    batch_size : int, default=64
        Mini-batch size used for training and inference.
    n_epochs : int, default=20
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

    The remaining parameters mirror the main architectural parameters of
    `braindecode.models.EEGConformer`.

    n_filters_time : int, default=40
        Number of temporal filters, also the transformer embedding size. Must be
        divisible by ``num_heads``.
    filter_time_length : int, default=25
        Kernel length of the temporal convolution.
    pool_time_length : int, default=75
        Kernel length of the temporal average pooling.
    pool_time_stride : int, default=15
        Stride of the temporal average pooling.
    drop_prob : float, default=0.5
        Dropout probability in the convolutional patch embedding.
    num_layers : int, default=6
        Number of transformer encoder layers.
    num_heads : int, default=10
        Number of attention heads.
    att_drop_prob : float, default=0.5
        Dropout probability in the transformer layers.
    final_fc_length : int or str, default="auto"
        Input length of the final fully connected layer.

    References
    ----------
    .. [1] Song, Y., Zheng, Q., Liu, B., and Gao, X. (2023). EEG Conformer:
       Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, 710-719. DOI: 10.1109/TNSRE.2022.3230250
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
        n_epochs: int = 20,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        device: str | None = None,
        verbose: bool = False,
        random_state: int | None = None,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        num_layers: int = 6,
        num_heads: int = 10,
        att_drop_prob: float = 0.5,
        final_fc_length: int | str = "auto",
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.att_drop_prob = att_drop_prob
        self.final_fc_length = final_fc_length

        self.network_ = None
        self.optimizer_ = None
        self.loss_ = None
        self.device_ = None
        self.history_ = []

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit EEGConformerClassifier."""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        min_timepoints = self.filter_time_length + self.pool_time_length - 1
        if X.shape[2] < min_timepoints:
            raise ValueError(
                f"EEGConformerClassifier requires at least {min_timepoints} "
                f"timepoints (filter_time_length + pool_time_length - 1), "
                f"got {X.shape[2]}."
            )

        X = np.asarray(X, dtype=np.float32, order="C")
        y_int = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.n_timepoints_ = X.shape[2]
        self.device_ = self._resolve_device()
        self.network_ = self._build_network(
            n_channels=X.shape[1], n_timepoints=X.shape[2]
        ).to(self.device_)
        self.optimizer_ = self._build_optimizer(self.network_.parameters())
        self.loss_ = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_int))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        self.history_ = []
        self.network_.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0

            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                self.optimizer_.zero_grad(set_to_none=True)
                logits = self._coerce_logits(self.network_(xb))
                loss = self.loss_(logits, yb)
                loss.backward()
                self.optimizer_.step()

                epoch_loss += loss.item() * xb.shape[0]

            epoch_loss /= len(dataset)
            self.history_.append(epoch_loss)

            if self.verbose:
                print(  # noqa: T201
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"training loss: {epoch_loss:.6f}"
                )

        # Leave the network in eval mode so predict does not mutate fitted state.
        self.network_.eval()

        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if X.shape[2] != self.n_timepoints_:
            raise ValueError(
                f"EEGConformerClassifier was fitted on series of length "
                f"{self.n_timepoints_}, got {X.shape[2]}. The network's final "
                f"layer depends on the series length used in fit."
            )

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

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode EEGConformer model."""
        from braindecode.models import EEGConformer

        return EEGConformer(
            n_chans=n_channels,
            n_outputs=self.n_classes_,
            n_times=n_timepoints,
            n_filters_time=self.n_filters_time,
            filter_time_length=self.filter_time_length,
            pool_time_length=self.pool_time_length,
            pool_time_stride=self.pool_time_stride,
            drop_prob=self.drop_prob,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            att_drop_prob=self.att_drop_prob,
            final_fc_length=self.final_fc_length,
        )

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

    @staticmethod
    def _coerce_logits(logits):
        """Defensively coerce model output to shape (batch_size, n_classes)."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        while logits.ndim > 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        if logits.ndim != 2:
            logits = logits.reshape(logits.shape[0], -1)

        return logits

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        return {
            "batch_size": 4,
            "n_epochs": 1,
            "lr": 1e-3,
            "verbose": False,
            "random_state": 0,
            "num_layers": 1,
            "num_heads": 2,
            "filter_time_length": 5,
            "pool_time_length": 5,
            "pool_time_stride": 2,
        }
