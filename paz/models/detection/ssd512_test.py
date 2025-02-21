import pytest
from paz.models.detection.ssd512 import SSD512
from keras.models import Model


def test_model_creation():
    """Test if SSD512 model is created successfully."""
    num_classes = 81
    input_shape = (512, 512, 3)
    model = SSD512(num_classes=num_classes, input_shape=input_shape)
    output_shape = model.outputs[0].shape

    # Check if the returned object is a Keras Model
    assert isinstance(model, Model)

    # Check if the model has a single concatenated output
    assert len(model.outputs) == 1  # Single output tensor

    # Check the shape of the output tensor: classification + localization
    assert output_shape[-1] == num_classes + 4
    assert output_shape[1] > 0  # Total number of boxes


def test_invalid_base_weights():
    """Test invalid base weights."""
    with pytest.raises(ValueError):
        SSD512(base_weights="INVALID")


def test_invalid_head_weights():
    """Test invalid head weights."""
    with pytest.raises(ValueError):
        SSD512(head_weights="INVALID")


def test_incompatible_num_classes_with_weights():
    """Test incompatible num_classes with pre-trained weights."""
    with pytest.raises(ValueError):
        SSD512(num_classes=10, head_weights="COCO")


if __name__ == "__main__":
    pytest.main([__file__])
