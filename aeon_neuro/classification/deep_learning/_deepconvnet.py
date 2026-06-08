# aeon_neuro/classification/deep_learning/_deep_conv_net.py

"""Braindecode Deep ConvNet wrapper for aeon."""

from __future__ import annotations

__all__ = ["DeepConvNetClassifier"]

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.utils import check_random_state


class DeepConvNetClassifier(BaseClassifier):
    """aeon wrapper around Braindecode's Deep ConvNet (Deep4Net).

    This classifier is for equal-length collection data and accepts multivariate
    time series in aeon's standard shape:
    (n_cases, n_channels, n_timepoints). Braindecode also uses this shape.

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
    `braindecode.models.Deep4Net`.

    References
    ----------
    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., and
       Ball, T. (2017). Deep learning with convolutional neural networks
       for EEG decoding and visualization. Human Brain Mapping.
       DOI: 10.1002/hbm.23730
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
        n_filters_time: int = 25,
        n_filters_spat: int = 25,
        filter_time_length: int = 10,
        pool_time_length: int = 3,
        pool_time_stride: int = 3,
        n_filters_2: int = 50,
        filter_length_2: int = 10,
        n_filters_3: int = 100,
        filter_length_3: int = 10,
        n_filters_4: int = 200,
        filter_length_4: int = 10,
        drop_prob: float = 0.5,
        split_first_layer: bool = True,
        batch_norm: bool = True,
        batch_norm_alpha: float = 0.1,
        stride_before_pool: bool = False,
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
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        self.network_ = None
        self.optimizer_ = None
        self.loss_ = None
        self.device_ = None
        self.history_ = []

        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit DeepConvNetClassifier."""
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
        """Construct the Braindecode Deep4Net model."""
        from braindecode.models import Deep4Net

        return Deep4Net(
            n_chans=n_channels,
            n_outputs=self.n_classes_,
            n_times=n_timepoints,
            final_conv_length=self.final_conv_length,
            n_filters_time=self.n_filters_time,
            n_filters_spat=self.n_filters_spat,
            filter_time_length=self.filter_time_length,
            pool_time_length=self.pool_time_length,
            pool_time_stride=self.pool_time_stride,
            n_filters_2=self.n_filters_2,
            filter_length_2=self.filter_length_2,
            n_filters_3=self.n_filters_3,
            filter_length_3=self.filter_length_3,
            n_filters_4=self.n_filters_4,
            filter_length_4=self.filter_length_4,
            drop_prob=self.drop_prob,
            split_first_layer=self.split_first_layer,
            batch_norm=self.batch_norm,
            batch_norm_alpha=self.batch_norm_alpha,
            stride_before_pool=self.stride_before_pool,
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
        """Return testing parameter settings for the aeon estimator checks."""
        return {
            "batch_size": 4,
            "n_epochs": 1,
            "lr": 1e-3,
            "verbose": False,
            "random_state": 0,
        }
