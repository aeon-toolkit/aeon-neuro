"""aeon compatible wrapper for Braindecode EEGConformer classifier."""

from __future__ import annotations

__all__ = ["EEGConformerClassifier"]

from aeon_neuro.classification.deep_learning._braindecode import (
    _BaseBraindecodeClassifier,
)


class EEGConformerClassifier(_BaseBraindecodeClassifier):
    """aeon wrapper around Braindecode's EEGConformer model.

    This classifier is for equal-length multivariate time series and expects aeon's
    standard collection shape:
    (n_cases, n_channels, n_timepoints).

    The network is built for the number of channels and series length seen in
    ``fit``, so ``predict`` needs data of the same shape; aeon raises a
    ``ValueError`` otherwise.

    Series must have at least ``filter_time_length + pool_time_length - 1``
    timepoints (99 with the default parameters).

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
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.att_drop_prob = att_drop_prob
        self.final_fc_length = final_fc_length

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

    def _build_network(self, n_channels: int, n_timepoints: int):
        """Construct the Braindecode EEGConformer model."""
        min_timepoints = self.filter_time_length + self.pool_time_length - 1
        if n_timepoints < min_timepoints:
            raise ValueError(
                f"EEGConformerClassifier requires at least {min_timepoints} "
                f"timepoints (filter_time_length + pool_time_length - 1), "
                f"got {n_timepoints}."
            )

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
