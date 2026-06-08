"""EEG Network (EEGNet)."""

__all__ = ["EEGNetNetwork"]
__maintainer__ = ["hadifawaz1999"]

from aeon_neuro.networks.base import BaseDeepLearningNetwork


class EEGNetNetwork(BaseDeepLearningNetwork):
    """Establish the network structure of EEGNet.

    EEGNet [1]_ is a Convolutional Neural Network (CNN) that uses
    DepthWise Seprabrable Convolutions in order to avoid high
    number of trainable parameters. EEGNet uses two dimensional
    convolutions in order to take into consideration channel
    correlation. The first phase of this model is a temporal
    standard convolution, in 2D to avoid aggregating channel
    wise outputs, this is followed by spatial 2D convolution
    to learn correlations between channels. This is followed by
    multiple 2D Separable Convolutions (default to 1 layer)
    with auto-calculated parameters. The final layer applies
    a dropout and flattens all axes to a single vector.

    Parameters
    ----------
    n_temporal_conv_filters : int, default = 8
        The number of standard convolution filters learned.
    kernel_size : int, default = 64
        The length of every convolution filter applied
        on the temporal axis.
    depth_multiplier : int, default = 2
        The number of filters to learn per channel in the
        spatial convolution phase.
    dropout_rate : float, default = 0.5
        The dropout rate to turn off a percentage of neurons
        to avoid overfitting.
    n_separable_convolution_layers: int, default = 1,
        The number of Depthwise Separable Convolution layers
        applied after the first spatio-temporal convolution
        phase. Number of filters of these layers is set
        to the n_temporal_conv_filters*depth_multiplier
        and the kernel size to kernel_size/4.
    activation : str, default = "elu"
        The default activation function used after every
        convolution block.
    pool_size : int, default = 4
        The size of the temporal average pooling operation.
        In the separable convolution layers, the pool size
        value is doubled.
    """

    def __init__(
        self,
        n_temporal_conv_filters: int = 8,
        kernel_size: int = 64,
        depth_multiplier: int = 2,
        dropout_rate: float = 0.5,
        n_separable_convolution_layers: int = 1,
        activation: str = "elu",
        pool_size: int = 4,
    ):

        self.n_temporal_conv_filters = n_temporal_conv_filters
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.dropout_rate = dropout_rate
        self.n_separable_convolution_layers = n_separable_convolution_layers
        self.activation = activation
        self.pool_size = pool_size

        super().__init__()

    def _spatio_temporal_convolution_layer(
        self,
        input_tensor,
        n_input_channels,
        n_filters,
        kernel_size,
        strides,
        padding,
        activation,
        use_bias,
        depth_multiplier,
    ):
        import tensorflow as tf

        temporal_conv = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, 1),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
        )(input_tensor)

        temporal_conv = tf.keras.layers.BatchNormalization()(temporal_conv)

        spatial_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, n_input_channels),
            use_bias=use_bias,
            depth_multiplier=depth_multiplier,
            depthwise_constraint=tf.keras.constraints.max_norm(1.0),
        )(temporal_conv)

        spatial_conv = tf.keras.layers.BatchNormalization()(spatial_conv)
        spatial_conv = tf.keras.layers.Activation(activation=activation)(spatial_conv)

        return spatial_conv

    def _separable_convolution(
        self,
        input_tensor,
        n_filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        activation,
        pool_size,
        dropout_rate,
    ):
        import tensorflow as tf

        conv = tf.keras.layers.SeparableConv2D(
            filters=n_filters,
            kernel_size=(kernel_size, 1),
            padding=padding,
            strides=strides,
            use_bias=use_bias,
        )(input_tensor)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation(activation=activation)(conv)

        pool = tf.keras.layers.AveragePooling2D(pool_size=(pool_size, 1))(conv)

        dropout = tf.keras.layers.Dropout(dropout_rate)(pool)

        return dropout

    def build_network(self, input_shape, **kwargs):
        """Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer. This function
            assumes the input_shape is (n_timepoints, n_channels) as it
            is the tensorflow-keras expectation.

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(input_shape)

        n_timepoints = input_shape[0]
        n_channels = input_shape[1]

        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(n_timepoints, n_channels, 1)
        )(input_layer)

        spatio_temporal_convolution = self._spatio_temporal_convolution_layer(
            input_tensor=reshape_layer,
            n_input_channels=n_channels,
            n_filters=self.n_temporal_conv_filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation=self.activation,
            use_bias=False,
            depth_multiplier=self.depth_multiplier,
        )

        average_pooling = tf.keras.layers.AveragePooling2D(
            pool_size=(self.pool_size, 1)
        )(spatio_temporal_convolution)

        dropout = tf.keras.layers.Dropout(self.dropout_rate)(average_pooling)

        new_n_channels = self.n_temporal_conv_filters * self.depth_multiplier

        x = dropout

        for _ in range(self.n_separable_convolution_layers):
            x = self._separable_convolution(
                input_tensor=x,
                n_filters=new_n_channels,
                kernel_size=self.kernel_size // 4,
                strides=1,
                padding="same",
                use_bias=False,
                activation=self.activation,
                pool_size=self.pool_size * 2,
                dropout_rate=self.dropout_rate,
            )

        flatten = tf.keras.layers.Flatten()(x)

        output_layer = tf.keras.layers.Dropout(self.dropout_rate)(flatten)

        return input_layer, output_layer
