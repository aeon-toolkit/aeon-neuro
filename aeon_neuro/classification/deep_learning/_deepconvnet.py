"""Braindecode Deep ConvNet wrapper for aeon."""

from __future__ import annotations

__all__ = ["DeepConvNetClassifier"]

from aeon_neuro.classification.deep_learning._braindecode import (
    _BaseBraindecodeClassifier,
)


class DeepConvNetClassifier(_BaseBraindecodeClassifier):
    """aeon wrapper around Braindecode's Deep ConvNet (Deep4Net).

    This classifier is for equal-length collection data and accepts multivariate
    time series in aeon's standard shape:
    (n_cases, n_channels, n_timepoints). Braindecode also uses this shape.

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
    `braindecode.models.Deep4Net`.

    References
    ----------
    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., and
       Ball, T. (2017). Deep learning with convolutional neural networks
       for EEG decoding and visualization. Human Brain Mapping.
       DOI: 10.1002/hbm.23730
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

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the aeon estimator checks."""
        return {
            "batch_size": 4,
            "n_epochs": 1,
            "lr": 1e-3,
            "verbose": False,
            "random_state": 0,
        }
