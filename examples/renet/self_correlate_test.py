import keras
import numpy as np
from keras import ops

from self_correlate import correlate


def test_output_shape_default():
    B, H, W, C = 2, 10, 10, 4
    input_tensor = keras.random.uniform(shape=(B, H, W, C))
    correlation_map = correlate(input_tensor)
    expected_shape = (B, H, W, 5, 5, C)
    assert ops.shape(correlation_map) == expected_shape


def test_output_shape_custom_params():
    B, H, W, C = 1, 8, 8, 3
    custom_kernel = (3, 3)
    input_tensor = keras.random.uniform(shape=(B, H, W, C))

    correlation_map = correlate(
        input_tensor,
        kernel_size=custom_kernel,
    )

    expected_shape = (B, H, W, *custom_kernel, C)
    assert ops.shape(correlation_map) == expected_shape


def test_correlation_values_simple_case():
    input_tensor = ops.convert_to_tensor([[[[2.0]]]], dtype="float32")
    correlation_map = correlate(
        input_tensor,
        kernel_size=(3, 3),
    )
    expected_patch = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    ).reshape(1, 1, 1, 3, 3, 1)

    np.testing.assert_allclose(
        ops.convert_to_numpy(correlation_map), expected_patch, rtol=1e-6
    )


def test_relu_effect():
    input_tensor = ops.convert_to_tensor([[[[-2.0]]]], dtype="float32")
    correlation_map = correlate(
        input_tensor,
        kernel_size=(3, 3),
    )
    expected_output = np.zeros(shape=(1, 1, 1, 3, 3, 1))
    np.testing.assert_allclose(
        ops.convert_to_numpy(correlation_map), expected_output, atol=1e-7
    )
