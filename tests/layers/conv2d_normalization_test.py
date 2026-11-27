import numpy as np
import pytest
from keras.layers import Input
from keras.models import Model
import paz


@pytest.fixture
def parameters():
    return {
        "input_shape": (4, 4, 3),
        "scale": 2.0,
        "axis": -1,
    }


def test_normalization_and_scaling(parameters):
    """Test if the layer normalizes and scales the input correctly."""
    input_shape = parameters["input_shape"]
    scale = parameters["scale"]
    axis = parameters["axis"]

    input_tensor = Input(shape=input_shape)
    layer = paz.layers.Conv2DNormalization(scale=scale, axis=axis)
    model = Model(inputs=input_tensor, outputs=layer(input_tensor))

    # Create a random input tensor
    random_state = np.random.RandomState(0)
    x = random_state.rand(1, *input_shape).astype(np.float32)

    # Compute the L2 norm manually
    l2_norm = np.sqrt(
        np.sum(x**2, axis=axis, keepdims=True) + np.finfo(np.float32).eps
    )
    normalized_x = x / l2_norm
    expected_output = scale * normalized_x

    # Get the output from the layer
    output = model.predict(x)

    # Check if the output matches the expected result
    np.testing.assert_allclose(output, expected_output, atol=1e-6)
