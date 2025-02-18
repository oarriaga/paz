import unittest
from keras.layers import Input
from keras.models import Model
import numpy as np

# Import the modules to be tested
from paz.models.detection.efficientdet.layers import (
    GetDropConnect,
    FuseFeature
)
from paz.models.detection.efficientdet.efficientnet import (
    EFFICIENTNET,
    conv_block,
    MBconv_blocks,
)
from paz.models.detection.efficientdet.efficientdet import EFFICIENTDETD0
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_detector_head,
    ClassNet,
    BoxesNet,
    EfficientNet_to_BiFPN,
    BiFPN,
)


class TestDetectionModels(unittest.TestCase):

    def test_GetDropConnect(self):
        """Test the GetDropConnect layer."""
        layer = GetDropConnect(survival_rate=0.8)
        input_tensor = np.random.rand(1, 4, 4, 3).astype("float32")

        # Test training mode (dropout applied)
        output_train = layer.call(input_tensor, training=True)
        self.assertEqual(output_train.shape, (1, 4, 4, 3))
        self.assertFalse(
            np.allclose(output_train, input_tensor)
        )  # Values should change

        # Test inference mode (no dropout)
        output_inference = layer.call(input_tensor, training=False)
        self.assertTrue(np.allclose(output_inference, input_tensor))

    def test_FuseFeature(self):
        """Test the FuseFeature layer."""
        # Test 'fast' fusion
        layer_fast = FuseFeature(fusion="fast")
        input_shape = [(1, 4, 4, 3), (1, 4, 4, 3), (1, 4, 4, 3)]
        # Manually call build to initialize weights
        layer_fast.build(input_shape)
        input_tensors = [
            np.random.rand(1, 4, 4, 3).astype("float32") for _ in range(3)
            ]
        output_fast = layer_fast.call(input_tensors, fusion="fast")
        self.assertEqual(output_fast.shape, (1, 4, 4, 3))

        # Test 'sum' fusion
        layer_sum = FuseFeature(fusion="sum")
        output_sum = layer_sum.call(input_tensors, fusion="sum")
        self.assertEqual(output_sum.shape, (1, 4, 4, 3))
        expected_sum = input_tensors[0] + input_tensors[1] + input_tensors[2]
        self.assertTrue(np.allclose(output_sum, expected_sum))

    def test_conv_block(self):
        """Test the conv_block function."""
        input_tensor = Input(shape=(32, 32, 3))
        output_tensor = conv_block(
            input_tensor,
            intro_filters=[32],
            width_coefficient=1.0,
            depth_divisor=8
        )
        model = Model(inputs=input_tensor, outputs=output_tensor)
        self.assertEqual(model.output_shape, (None, 16, 16, 32))

    def test_MBconv_blocks(self):
        """Test the MBconv_blocks function."""
        input_tensor = Input(shape=(32, 32, 32))
        scaling_coefficients = (1.0, 1.0, 0.8)
        output_features = MBconv_blocks(
            input_tensor,
            kernel_sizes=[3],
            intro_filters=[32],
            outro_filters=[16],
            W_coefficient=scaling_coefficients[0],
            D_coefficient=scaling_coefficients[1],
            D_divisor=8,
            repeats=[1],
            excite_ratio=0.25,
            survival_rate=scaling_coefficients[2],
            strides=[[1, 1]],
            expand_ratios=[6],
        )
        self.assertEqual(len(output_features), 1)
        self.assertEqual(output_features[0].shape[-1], 16)

    def test_EFFICIENTNET(self):
        """Test the EFFICIENTNET function."""
        input_tensor = Input(shape=(64, 64, 3))
        scaling_coefficients = (1.0, 1.0, 0.8)
        output_features = EFFICIENTNET(
            image=input_tensor, scaling_coefficients=scaling_coefficients
        )
        self.assertIsInstance(output_features, list)
        self.assertEqual(len(output_features), 5)  # P3-P7 outputs

    def test_build_detector_head(self):
        """Test the build_detector_head function."""
        input_tensors = [Input(shape=(16, 16, 64)) for _ in range(5)]
        output_tensor = build_detector_head(
            middles=input_tensors,
            num_classes=90,
            num_dims=4,
            aspect_ratios=[1.0, 2.0, 0.5],
            num_scales=3,
            FPN_num_filters=64,
            box_class_repeats=3,
            survival_rate=0.8,
        )
        self.assertEqual(output_tensor.shape[-1], 94)
        # Update the expected anchor count to account for all 5 levels:
        self.assertEqual(
            output_tensor.shape[1], 16 * 16 * 9 * 5
        )  # Total anchors across 5 FPN levels

    def test_ClassNet(self):
        """Test the ClassNet function."""
        input_tensors = [Input(shape=(16, 16, 64)) for _ in range(5)]
        _, class_outputs = ClassNet(
            features=input_tensors,
            num_anchors=9,
            num_filters=32,
            num_blocks=4,
            survival_rate=0.8,
            num_classes=90,
        )
        self.assertEqual(len(class_outputs), 5)
        self.assertEqual(
            class_outputs[0].shape[-1], 90 * 9
        )  # num_classes * num_anchors

    def test_BoxesNet(self):
        """Test the BoxesNet function."""
        input_tensors = [Input(shape=(16, 16, 64)) for _ in range(5)]
        _, boxes_outputs = BoxesNet(
            features=input_tensors,
            num_anchors=9,
            num_filters=32,
            num_blocks=4,
            survival_rate=0.8,
            num_dims=4,
        )
        self.assertEqual(len(boxes_outputs), 5)
        self.assertEqual(boxes_outputs[0].shape[-1], 4 * 9)

    def test_EfficientNet_to_BiFPN(self):
        """Test the EfficientNet_to_BiFPN function."""
        branches = [Input(shape=(16, 16, i * 32)) for i in range(1, 6)]
        branches, middles, skips = EfficientNet_to_BiFPN(branches,
                                                         num_filters=64)
        self.assertEqual(len(middles), 5)
        self.assertEqual(len(skips), 5)
        self.assertEqual(middles[0].shape[-1], 64)  # Check num_filters

    def test_BiFPN(self):
        """Test the BiFPN function."""
        middles = [
            Input(shape=(16 // (2**i), 16 // (2**i), 64)) for i in range(5)
            ]
        skips = [
            None if i == 0 or i == 4 else Input(
                shape=(16 // (2**i), 16 // (2**i), 64)
                )
            for i in range(5)
        ]
        new_middles, _ = BiFPN(middles, skips, num_filters=64, fusion="fast")
        self.assertEqual(len(new_middles), 5)
        self.assertEqual(new_middles[0].shape[-1], 64)  # Check num_filters

    def test_EFFICIENTDETD0(self):
        """Test the EFFICIENTDETD0 function."""
        model = EFFICIENTDETD0(
            num_classes=90,
            base_weights=None,
            head_weights=None,
            input_shape=(512, 512, 3),
            FPN_num_filters=64,
            FPN_cell_repeats=3,
            box_class_repeats=3,
            anchor_scale=4.0,
            fusion="fast",
            return_base=False,
            model_name="efficientdet-d0",
            scaling_coefficients=(1.0, 1.0, 0.8),
        )
        self.assertIsInstance(model, Model)
        self.assertIsNotNone(model.prior_boxes)


if __name__ == "__main__":
    unittest.main()
