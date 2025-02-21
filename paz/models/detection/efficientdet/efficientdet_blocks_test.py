import pytest
from keras.layers import Input
# Import the modules to be tested
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_detector_head,
    ClassNet,
    BoxesNet,
    EfficientNet_to_BiFPN,
    BiFPN,
)


def test_build_detector_head():
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
    # Last dimension should equal num_classes + num_dims (i.e., 90+4)
    assert output_tensor.shape[-1] == 94
    # Total anchors: grid size * num_anchors per cell * number of FPN levels
    expected_total_anchors = 16 * 16 * 9 * 5
    assert output_tensor.shape[1] == expected_total_anchors


def test_class_net():
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
    assert len(class_outputs) == 5
    # Each output should have last dimension equal to num_classes * num_anchors
    assert class_outputs[0].shape[-1] == 90 * 9


def test_boxes_net():
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
    assert len(boxes_outputs) == 5
    # Each output should have last dimension equal to num_dims * num_anchors
    assert boxes_outputs[0].shape[-1] == 4 * 9


def test_efficientnet_to_bifpn():
    """Test the EfficientNet_to_BiFPN function."""
    branches = [Input(shape=(16, 16, i * 32)) for i in range(1, 6)]
    branches, middles, skips = EfficientNet_to_BiFPN(branches, num_filters=64)
    assert len(middles) == 5
    assert len(skips) == 5
    # Check that the number of filters in the first middle branch is set correctly
    assert middles[0].shape[-1] == 64


def test_bifpn():
    """Test the BiFPN function."""
    middles = [Input(shape=(16 // (2**i), 16 // (2**i), 64)) for i in range(5)]
    skips = [
        None if i == 0 or i == 4 else Input(shape=(16 // (2**i), 16 // (2**i), 64))
        for i in range(5)
    ]
    new_middles, _ = BiFPN(middles, skips, num_filters=64, fusion="fast")
    assert len(new_middles) == 5
    assert new_middles[0].shape[-1] == 64


if __name__ == "__main__":
    pytest.main([__file__])
