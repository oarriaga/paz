import unittest
import numpy as np
from keras.layers import Input
from keras.models import Model
from paz.models.layers import Conv2DNormalization


class TestConv2DNormalization(unittest.TestCase):
    def setUp(self):
        """Set up common variables for all tests."""
        self.input_shape = (4, 4, 3)  # Example input shape
        self.scale = 2.0  # Example scale value
        self.axis = -1  # Default axis for channels

    def test_initialization(self):
        """Test if the layer initializes correctly."""
        layer = Conv2DNormalization(scale=self.scale, axis=self.axis)
        self.assertEqual(layer.scale, self.scale)
        self.assertEqual(layer.axis, self.axis)

    def test_normalization_and_scaling(self):
        """Test if the layer normalizes and scales the input correctly."""
        input_tensor = Input(shape=self.input_shape)
        layer = Conv2DNormalization(scale=self.scale, axis=self.axis)
        model = Model(inputs=input_tensor, outputs=layer(input_tensor))

        # Create a random input tensor
        x = np.random.rand(1, *self.input_shape).astype(np.float32)

        # Compute the L2 norm manually
        l2_norm = np.sqrt(
            np.sum(x**2, axis=self.axis, keepdims=True) + np.finfo(np.float32).eps
        )
        normalized_x = x / l2_norm
        expected_output = self.scale * normalized_x

        # Get the output from the layer
        output = model.predict(x)

        # Check if the output matches the expected result
        np.testing.assert_allclose(output, expected_output, atol=1e-6)

    def test_output_shape(self):
        """Test if the output shape matches the input shape."""
        input_tensor = Input(shape=self.input_shape)
        layer = Conv2DNormalization(scale=self.scale, axis=self.axis)
        model = Model(inputs=input_tensor, outputs=layer(input_tensor))

        # Create a random input tensor
        x = np.random.rand(1, *self.input_shape).astype(np.float32)

        # Get the output shape
        output_shape = model.predict(x).shape

        # Check if the output shape matches the input shape
        self.assertEqual(output_shape, (1, *self.input_shape))


if __name__ == "__main__":
    unittest.main()
