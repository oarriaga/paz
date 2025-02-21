import pytest
from keras.layers import Input
from keras.models import Model

# Import the modules to be tested
from paz.models.detection.efficientdet.efficientnet import (
    EFFICIENTNET,
    conv_block,
    MBconv_blocks,
)


def test_conv_block():
    """Test the conv_block function."""
    input_tensor = Input(shape=(32, 32, 3))
    output_tensor = conv_block(
        input_tensor, intro_filters=[32], width_coefficient=1.0, depth_divisor=8
    )
    model = Model(inputs=input_tensor, outputs=output_tensor)
    # Expect spatial dimensions to be reduced (e.g., downsampled by 2)
    assert model.output_shape == (None, 16, 16, 32)


def test_MBconv_blocks():
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
    assert len(output_features) == 1
    assert output_features[0].shape[-1] == 16


def test_efficientnet():
    """Test the EFFICIENTNET function."""
    input_tensor = Input(shape=(64, 64, 3))
    scaling_coefficients = (1.0, 1.0, 0.8)
    output_features = EFFICIENTNET(
        image=input_tensor, scaling_coefficients=scaling_coefficients
    )
    assert isinstance(output_features, list)
    assert len(output_features) == 5  # P3-P7 outputs


if __name__ == "__main__":
    pytest.main([__file__])
