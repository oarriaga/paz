import unittest
from paz.models.detection.ssd300 import SSD300
from keras.models import Model


class TestSSD300(unittest.TestCase):
    def test_model_creation(self):
        """Test if SSD300 model is created successfully."""
        num_classes = 21
        input_shape = (300, 300, 3)
        model = SSD300(num_classes=num_classes, input_shape=input_shape)
        output_shape = model.outputs[0].shape

        # Check if the returned object is a Keras Model
        self.assertIsInstance(model, Model)
        # Check if the model has a single concatenated output
        self.assertEqual(len(model.outputs), 1)  # Single output tensor

        # Check the shape of the output tensor
        self.assertEqual(
            output_shape[-1], num_classes + 4
        )  # Classification + Localization
        self.assertGreater(output_shape[1], 0)  # Total number of boxes

    def test_invalid_base_weights(self):
        """Test invalid base weights."""
        with self.assertRaises(ValueError):
            SSD300(base_weights="INVALID")

    def test_invalid_head_weights(self):
        """Test invalid head weights."""
        with self.assertRaises(ValueError):
            SSD300(head_weights="INVALID")

    def test_incompatible_num_classes_with_weights(self):
        """Test incompatible num_classes with pre-trained weights."""
        with self.assertRaises(ValueError):
            SSD300(num_classes=10, head_weights="VOC")


if __name__ == "__main__":
    unittest.main()
