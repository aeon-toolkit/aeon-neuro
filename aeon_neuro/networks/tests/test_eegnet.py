"""Tests for EEGNet."""

import pytest
from aeon.utils.validation._dependencies import _check_soft_dependencies

from aeon_neuro.networks import EEGNetNetwork


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_output_shapes_eegnet():
    """Testing output shapes of EEGNet."""
    input_shape = (10, 2)
    network = EEGNetNetwork(kernel_size=4, pool_size=1)

    # in the case of input shape (10, 2) and kernel_size=4
    # and pool_size=1, the expected "theoretical" output shape
    # of this netwroks should be a vector of length 80

    input_layer, output_layer = network.build_network(input_shape=input_shape)

    assert input_layer is not None
    assert output_layer is not None
    assert input_layer.shape[1] == input_shape[0]
    assert input_layer.shape[2] == input_shape[1]
    assert len(output_layer.shape[1:]) == 1
    assert int(output_layer.shape[1]) == 80
