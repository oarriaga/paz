import numpy as np
import pytest
from keras.layers import Input
from keras.models import Model
from paz.models.layers import Conv2DNormalization


@pytest.fixture
def common_params():
    return {
        "input_shape": (4, 4, 3),
        "scale": 2.0,
        "axis": -1,
    }


def test_initialization(common_params):
    """Test if the layer initializes correctly."""
    scale = common_params["scale"]
    axis = common_params["axis"]
    layer = Conv2DNormalization(scale=scale, axis=axis)
    assert layer.scale == scale
    assert layer.axis == axis


def test_normalization_and_scaling(common_params):
    """Test if the layer normalizes and scales the input correctly."""
    input_shape = common_params["input_shape"]
    scale = common_params["scale"]
    axis = common_params["axis"]

    input_tensor = Input(shape=input_shape)
    layer = Conv2DNormalization(scale=scale, axis=axis)
    model = Model(inputs=input_tensor, outputs=layer(input_tensor))

    # Create a random input tensor
    x = np.random.rand(1, *input_shape).astype(np.float32)

    # Compute the L2 norm manually
    l2_norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True) + np.finfo(np.float32).eps)
    normalized_x = x / l2_norm
    expected_output = scale * normalized_x

    # Get the output from the layer
    output = model.predict(x)

    # Check if the output matches the expected result
    np.testing.assert_allclose(output, expected_output, atol=1e-6)


def test_output_shape(common_params):
    """Test if the output shape matches the input shape."""
    input_shape = common_params["input_shape"]
    scale = common_params["scale"]
    axis = common_params["axis"]

    input_tensor = Input(shape=input_shape)
    layer = Conv2DNormalization(scale=scale, axis=axis)
    model = Model(inputs=input_tensor, outputs=layer(input_tensor))

    # Create a random input tensor
    x = np.random.rand(1, *input_shape).astype(np.float32)

    # Get the output shape
    output_shape = model.predict(x).shape

    # Check if the output shape matches the input shape
    assert output_shape == (1, *input_shape)


if __name__ == "__main__":
    pytest.main([__file__])
