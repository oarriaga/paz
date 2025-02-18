import unittest
import numpy as np
from keras.layers import Input
from keras.models import Model
from paz.models.detection.blocks import (
    build_multibox_head,
)  # Assuming this is the module name


class TestBuildMultiboxHead(unittest.TestCase):
    def setUp(self):
        """Set up common variables for all tests."""
        self.input_shape = (32, 32, 512)  # Example input shape
        self.num_classes = 21  # Example number of classes
        self.num_priors = [4, 6]  # Example number of priors per layer
        self.l2_loss = 0.0005
        self.num_regressions = 4  # For 2D bounding boxes

    def test_output_shape(self):
        """Test if the function produces the correct output shape."""
        # Create mock input tensors
        tensor1 = Input(shape=self.input_shape)
        tensor2 = Input(shape=self.input_shape)
        tensors = [tensor1, tensor2]

        # Build the multibox head
        outputs = build_multibox_head(
            tensors,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            l2_loss=self.l2_loss,
            num_regressions=self.num_regressions,
            l2_norm=True,
            batch_norm=True,
        )

        # Create a model to inspect the output shape
        model = Model(inputs=tensors, outputs=outputs)

        # Generate random input data
        x1 = np.random.rand(1, *self.input_shape).astype(np.float32)
        x2 = np.random.rand(1, *self.input_shape).astype(np.float32)

        # Get the output
        output = model.predict([x1, x2])

        # Calculate expected number of boxes
        total_boxes = sum(
            prior * np.prod(self.input_shape[:2]) for prior in self.num_priors
        )

        # Check the output shape
        expected_shape = (1, total_boxes, self.num_regressions + self.num_classes)
        self.assertEqual(output.shape, expected_shape)

    def test_l2_normalization(self):
        """Test if L2 normalization is applied correctly."""
        # Create mock input tensors
        tensor1 = Input(shape=self.input_shape)
        tensor2 = Input(shape=self.input_shape)
        tensors = [tensor1, tensor2]

        # Build the multibox head with L2 normalization enabled
        outputs_with_l2 = build_multibox_head(
            tensors,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            l2_loss=self.l2_loss,
            num_regressions=self.num_regressions,
            l2_norm=True,
            batch_norm=False,
        )

        # Build the multibox head with L2 normalization disabled
        outputs_without_l2 = build_multibox_head(
            tensors,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            l2_loss=self.l2_loss,
            num_regressions=self.num_regressions,
            l2_norm=False,
            batch_norm=False,
        )

        # Create models
        model_with_l2 = Model(inputs=tensors, outputs=outputs_with_l2)
        model_without_l2 = Model(inputs=tensors, outputs=outputs_without_l2)

        # Generate random input data
        x1 = np.random.rand(1, *self.input_shape).astype(np.float32)
        x2 = np.random.rand(1, *self.input_shape).astype(np.float32)

        # Get the outputs
        output_with_l2 = model_with_l2.predict([x1, x2])
        output_without_l2 = model_without_l2.predict([x1, x2])

        # Check if the outputs are different (indicating L2 normalization effect)
        self.assertFalse(np.allclose(output_with_l2, output_without_l2))

    def test_batch_normalization(self):
        """Test if batch normalization is applied correctly."""
        # Create mock input tensors
        tensor1 = Input(shape=self.input_shape)
        tensor2 = Input(shape=self.input_shape)
        tensors = [tensor1, tensor2]

        # Build the multibox head with batch normalization enabled
        outputs_with_bn = build_multibox_head(
            tensors,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            l2_loss=self.l2_loss,
            num_regressions=self.num_regressions,
            l2_norm=False,
            batch_norm=True,
        )

        # Build the multibox head with batch normalization disabled
        outputs_without_bn = build_multibox_head(
            tensors,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            l2_loss=self.l2_loss,
            num_regressions=self.num_regressions,
            l2_norm=False,
            batch_norm=False,
        )

        # Create models
        model_with_bn = Model(inputs=tensors, outputs=outputs_with_bn)
        model_without_bn = Model(inputs=tensors, outputs=outputs_without_bn)

        # Generate random input data
        x1 = np.random.rand(1, *self.input_shape).astype(np.float32)
        x2 = np.random.rand(1, *self.input_shape).astype(np.float32)

        # Get the outputs
        output_with_bn = model_with_bn.predict([x1, x2])
        output_without_bn = model_without_bn.predict([x1, x2])

        # Check if the outputs are different (indicating batch normalization effect)
        self.assertFalse(np.allclose(output_with_bn, output_without_bn))

    def test_concatenation(self):
        """Test if classification and regression outputs are concatenated correctly."""
        # Create mock input tensors
        tensor1 = Input(shape=self.input_shape)
        tensor2 = Input(shape=self.input_shape)
        tensors = [tensor1, tensor2]

        # Build the multibox head
        outputs = build_multibox_head(
            tensors,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            l2_loss=self.l2_loss,
            num_regressions=self.num_regressions,
            l2_norm=False,
            batch_norm=False,
        )

        # Create a model to inspect the output shape
        model = Model(inputs=tensors, outputs=outputs)

        # Generate random input data
        x1 = np.random.rand(1, *self.input_shape).astype(np.float32)
        x2 = np.random.rand(1, *self.input_shape).astype(np.float32)

        # Get the output
        output = model.predict([x1, x2])

        # Check if the last dimension matches the sum of regressions and classes
        total_boxes = sum(
            prior * np.prod(self.input_shape[:2]) for prior in self.num_priors
        )
        expected_last_dim = self.num_regressions + self.num_classes
        self.assertEqual(output.shape[-1], expected_last_dim)


if __name__ == "__main__":
    unittest.main()
