import pytest
import numpy as np
from keras.layers import Input
from keras.models import Model
from paz.models.detection.blocks import build_multibox_head


# Fixture for common variables
@pytest.fixture
def common_vars():
    return {
        "input_shape": (32, 32, 512),  # Example input shape
        "num_classes": 21,  # Example number of classes
        "num_priors": [4, 6],  # Example number of priors per layer
        "l2_loss": 0.0005,
        "num_regressions": 4,  # For 2D bounding boxes
    }


def test_output_shape(common_vars):
    """Test if the function produces the correct output shape."""
    input_shape = common_vars["input_shape"]
    num_classes = common_vars["num_classes"]
    num_priors = common_vars["num_priors"]
    l2_loss = common_vars["l2_loss"]
    num_regressions = common_vars["num_regressions"]

    # Create mock input tensors
    tensor1 = Input(shape=input_shape)
    tensor2 = Input(shape=input_shape)
    tensors = [tensor1, tensor2]

    # Build the multibox head with L2 norm and batch norm enabled
    outputs = build_multibox_head(
        tensors,
        num_classes=num_classes,
        num_priors=num_priors,
        l2_loss=l2_loss,
        num_regressions=num_regressions,
        l2_norm=True,
        batch_norm=True,
    )

    # Create a model to inspect the output shape
    model = Model(inputs=tensors, outputs=outputs)

    # Generate random input data
    x1 = np.random.rand(1, *input_shape).astype(np.float32)
    x2 = np.random.rand(1, *input_shape).astype(np.float32)

    # Get the output
    output = model.predict([x1, x2])

    # Calculate expected number of boxes
    total_boxes = sum(prior * np.prod(input_shape[:2]) for prior in num_priors)
    expected_shape = (1, total_boxes, num_regressions + num_classes)
    assert output.shape == expected_shape


def test_l2_normalization(common_vars):
    """Test if L2 normalization is applied correctly."""
    input_shape = common_vars["input_shape"]
    num_classes = common_vars["num_classes"]
    num_priors = common_vars["num_priors"]
    l2_loss = common_vars["l2_loss"]
    num_regressions = common_vars["num_regressions"]

    # Create mock input tensors
    tensor1 = Input(shape=input_shape)
    tensor2 = Input(shape=input_shape)
    tensors = [tensor1, tensor2]

    # Build the multibox head with L2 normalization enabled
    outputs_with_l2 = build_multibox_head(
        tensors,
        num_classes=num_classes,
        num_priors=num_priors,
        l2_loss=l2_loss,
        num_regressions=num_regressions,
        l2_norm=True,
        batch_norm=False,
    )

    # Build the multibox head with L2 normalization disabled
    outputs_without_l2 = build_multibox_head(
        tensors,
        num_classes=num_classes,
        num_priors=num_priors,
        l2_loss=l2_loss,
        num_regressions=num_regressions,
        l2_norm=False,
        batch_norm=False,
    )

    # Create models for both cases
    model_with_l2 = Model(inputs=tensors, outputs=outputs_with_l2)
    model_without_l2 = Model(inputs=tensors, outputs=outputs_without_l2)

    # Generate random input data
    x1 = np.random.rand(1, *input_shape).astype(np.float32)
    x2 = np.random.rand(1, *input_shape).astype(np.float32)

    # Get the outputs
    output_with_l2 = model_with_l2.predict([x1, x2])
    output_without_l2 = model_without_l2.predict([x1, x2])

    # Check if the outputs are different (indicating the effect of L2 normalization)
    assert not np.allclose(output_with_l2, output_without_l2)


def test_batch_normalization(common_vars):
    """Test if batch normalization is applied correctly."""
    input_shape = common_vars["input_shape"]
    num_classes = common_vars["num_classes"]
    num_priors = common_vars["num_priors"]
    l2_loss = common_vars["l2_loss"]
    num_regressions = common_vars["num_regressions"]

    # Create mock input tensors
    tensor1 = Input(shape=input_shape)
    tensor2 = Input(shape=input_shape)
    tensors = [tensor1, tensor2]

    # Build the multibox head with batch normalization enabled
    outputs_with_bn = build_multibox_head(
        tensors,
        num_classes=num_classes,
        num_priors=num_priors,
        l2_loss=l2_loss,
        num_regressions=num_regressions,
        l2_norm=False,
        batch_norm=True,
    )

    # Build the multibox head with batch normalization disabled
    outputs_without_bn = build_multibox_head(
        tensors,
        num_classes=num_classes,
        num_priors=num_priors,
        l2_loss=l2_loss,
        num_regressions=num_regressions,
        l2_norm=False,
        batch_norm=False,
    )

    # Create models for both cases
    model_with_bn = Model(inputs=tensors, outputs=outputs_with_bn)
    model_without_bn = Model(inputs=tensors, outputs=outputs_without_bn)

    # Generate random input data
    x1 = np.random.rand(1, *input_shape).astype(np.float32)
    x2 = np.random.rand(1, *input_shape).astype(np.float32)

    # Get the outputs
    output_with_bn = model_with_bn.predict([x1, x2])
    output_without_bn = model_without_bn.predict([x1, x2])

    # Check if the outputs are different (indicating the effect of batch normalization)
    assert not np.allclose(output_with_bn, output_without_bn)


def test_concatenation(common_vars):
    """Test if classification and regression outputs are concatenated correctly."""
    input_shape = common_vars["input_shape"]
    num_classes = common_vars["num_classes"]
    num_priors = common_vars["num_priors"]
    l2_loss = common_vars["l2_loss"]
    num_regressions = common_vars["num_regressions"]

    # Create mock input tensors
    tensor1 = Input(shape=input_shape)
    tensor2 = Input(shape=input_shape)
    tensors = [tensor1, tensor2]

    # Build the multibox head without L2 norm and batch normalization
    outputs = build_multibox_head(
        tensors,
        num_classes=num_classes,
        num_priors=num_priors,
        l2_loss=l2_loss,
        num_regressions=num_regressions,
        l2_norm=False,
        batch_norm=False,
    )

    # Create a model to inspect the output shape
    model = Model(inputs=tensors, outputs=outputs)

    # Generate random input data
    x1 = np.random.rand(1, *input_shape).astype(np.float32)
    x2 = np.random.rand(1, *input_shape).astype(np.float32)

    # Get the output
    output = model.predict([x1, x2])

    # Check if the last dimension matches the sum of regressions and classes
    expected_last_dim = num_regressions + num_classes
    assert output.shape[-1] == expected_last_dim


if __name__ == "__main__":
    pytest.main([__file__])
