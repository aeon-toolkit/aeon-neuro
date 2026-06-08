"""aeon compatible wrapper for nraindecode EEGNet classifier."""

from __future__ import annotations

__all__ = ["EEGNetClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state


class EEGNetClassifier(BaseClassifier):
    """aeon wrapper around Braindecode's EEGNet model.

    This classifier is for equal-length multivariate time series and expects aeon's
    standard collection shape:
    (n_cases, n_channels, n_timepoints).

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
    `braindecode.models.EEGNetv4`.

    final_conv_length : int or str, default="auto"
        Length of the final convolution layer.
    pool_mode : {"mean", "max"}, default="mean"
        Pooling mode.
    F1 : int, default=8
        Number of temporal filters.
    D : int, default=2
        Depth multiplier for depthwise convolution.
    F2 : int or None, default=None
        Number of pointwise filters. If None, Braindecode derives it internally.
    kernel_length : int, default=64
        Temporal kernel length of the first convolution.
    depthwise_kernel_length : int, default=16
        Kernel length of the depthwise temporal stage.
    pool1_kernel_size : int, default=4
        First pooling kernel size.
    pool2_kernel_size : int, default=8
        Second pooling kernel size.
    conv_spatial_max_norm : int, default=1
        Maximum norm for spatial convolution weights.
    batch_norm_momentum : float, default=0.01
        Batch normalisation momentum.
    batch_norm_affine : bool, default=True
        Whether batch norm has learnable affine parameters.
    batch_norm_eps : float, default=1e-3
        Batch normalisation epsilon.
    drop_prob : float, default=0.25
        Dropout probability.
    final_layer_with_constraint : bool, default=False
        Whether to apply the final layer norm constraint.
    norm_rate : float, default=0.25
        Norm constraint rate.

    References
    ----------
    .. [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
       Hung, C. P., and Lance, B. J. (2018). EEGNet: a compact convolutional
       neural network for EEG-based brain-computer interfaces.
       Journal of Neural Engineering, 15(5), 056013.
       DOI: 10.1088/1741-2552/aace8c
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
        final_conv_length: int | str = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        conv_spatial_max_norm: int = 1,
        batch_norm_momentum: float = 0.01,
        batch_norm_affine: bool = True,
        batch_norm_eps: float = 1e-3,
        drop_prob: float = 0.25,
        final_layer_with_constraint: bool = False,
        norm_rate: float = 0.25,
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.depthwise_kernel_length = depthwise_kernel_length
        self.pool1_kernel_size = pool1_kernel_size
        self.pool2_kernel_size = pool2_kernel_size
        self.conv_spatial_max_norm = conv_spatial_max_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_eps = batch_norm_eps
        self.drop_prob = drop_prob
        self.final_layer_with_constraint = final_layer_with_constraint
        self.norm_rate = norm_rate

        self.network_ = None
        self.optimizer_ = None
        self.loss_ = None
        self.device_ = None
        self.history_ = []

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit EEGNetClassifier."""
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

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode EEGNetv4 model."""
        from braindecode.models import EEGNetv4

        return EEGNetv4(
            n_chans=n_channels,
            n_outputs=self.n_classes_,
            n_times=n_timepoints,
            final_conv_length=self.final_conv_length,
            pool_mode=self.pool_mode,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernel_length,
            depthwise_kernel_length=self.depthwise_kernel_length,
            pool1_kernel_size=self.pool1_kernel_size,
            pool2_kernel_size=self.pool2_kernel_size,
            conv_spatial_max_norm=self.conv_spatial_max_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_affine=self.batch_norm_affine,
            batch_norm_eps=self.batch_norm_eps,
            drop_prob=self.drop_prob,
            final_layer_with_constraint=self.final_layer_with_constraint,
            norm_rate=self.norm_rate,
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
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        return {
            "batch_size": 4,
            "n_epochs": 1,
            "lr": 1e-3,
            "verbose": False,
            "random_state": 0,
        }
