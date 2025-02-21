import numpy as np
import pytest

# Import the modules to be tested
from paz.models.detection.efficientdet.layers import (
    GetDropConnect,
    FuseFeature
)


def test_get_drop_connect():
    """Test the GetDropConnect layer."""
    layer = GetDropConnect(survival_rate=0.8)
    input_tensor = np.random.rand(1, 4, 4, 3).astype("float32")

    # Test training mode (dropout applied)
    output_train = layer.call(input_tensor, training=True)
    assert output_train.shape == (1, 4, 4, 3)
    # Values should change in training mode due to dropout
    assert not np.allclose(output_train, input_tensor)

    # Test inference mode (no dropout)
    output_inference = layer.call(input_tensor, training=False)
    assert np.allclose(output_inference, input_tensor)


def test_fuse_feature():
    """Test the FuseFeature layer."""
    # Test 'fast' fusion
    layer_fast = FuseFeature(fusion="fast")
    input_shape = [(1, 4, 4, 3)] * 3
    # Manually call build to initialize weights
    layer_fast.build(input_shape)
    input_tensors = [
        np.random.rand(1, 4, 4, 3).astype("float32") for _ in range(3)
        ]
    output_fast = layer_fast.call(input_tensors, fusion="fast")
    assert output_fast.shape == (1, 4, 4, 3)

    # Test 'sum' fusion
    layer_sum = FuseFeature(fusion="sum")
    output_sum = layer_sum.call(input_tensors, fusion="sum")
    assert output_sum.shape == (1, 4, 4, 3)
    expected_sum = input_tensors[0] + input_tensors[1] + input_tensors[2]
    assert np.allclose(output_sum, expected_sum)


if __name__ == "__main__":
    pytest.main([__file__])
