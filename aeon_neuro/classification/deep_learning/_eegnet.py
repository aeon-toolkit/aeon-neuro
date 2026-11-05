"""aeon compatible wrapper for Braindecode EEGNet classifier."""

from __future__ import annotations

__all__ = ["EEGNetClassifier"]

from aeon_neuro.classification.deep_learning._braindecode import (
    _BaseBraindecodeClassifier,
)


class EEGNetClassifier(_BaseBraindecodeClassifier):
    """aeon wrapper around Braindecode's EEGNet model.

    This classifier is for equal-length multivariate time series and expects aeon's
    standard collection shape:
    (n_cases, n_channels, n_timepoints).

    The network is built for the number of channels and series length seen in
    ``fit``, so ``predict`` needs data of the same shape; aeon raises a
    ``ValueError`` otherwise.

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
    `braindecode.models.EEGNet`.

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
        """Construct the Braindecode EEGNet model."""
        from braindecode.models import EEGNet

        return EEGNet(
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

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for aeon estimator checks."""
        return {
            "batch_size": 4,
            "n_epochs": 1,
            "lr": 1e-3,
            "verbose": False,
            "random_state": 0,
        }
