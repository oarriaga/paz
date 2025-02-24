import pytest
from paz.models.detection.ssd300 import SSD300
from keras.models import Model


def test_model_creation():
    """Test if SSD300 model is created successfully."""
    num_classes = 21
    input_shape = (300, 300, 3)
    model = SSD300(num_classes=num_classes, input_shape=input_shape)
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
        SSD300(base_weights="INVALID")


def test_invalid_head_weights():
    """Test invalid head weights."""
    with pytest.raises(ValueError):
        SSD300(head_weights="INVALID")


def test_incompatible_num_classes_with_weights():
    """Test incompatible num_classes with pre-trained weights."""
    with pytest.raises(ValueError):
        SSD300(num_classes=10, head_weights="VOC")


def test_SSD300_VOC_VOC():
    """Test SSD300 with VOC-VOC weights."""
    try:
        SSD300(num_classes=21, base_weights="VOC", head_weights="VOC")
    except ValueError as valuerror:
        pytest.fail(f"SSD VOC-VOC loading failed: {valuerror}")


def test_SSD300_FAT_FAT():
    """Test SSD300 with FAT-FAT weights."""
    try:
        SSD300(num_classes=22, base_weights="FAT", head_weights="FAT")
    except OSError as e:
        pytest.fail(f"SSD FAT-FAT loading failed: {e}")


def test_SSD300_None_VGG():
    """Test SSD300 with None-VGG weights."""
    try:
        SSD300(num_classes=21, base_weights="VGG", head_weights=None)
    except ValueError as valuerror:
        pytest.fail(f"SSD None-VGG loading failed: {valuerror}")


def test_SSD300_None_VOC():
    """Test SSD300 with None-VOC weights."""
    try:
        SSD300(num_classes=21, base_weights="VOC", head_weights=None)
    except ValueError as valuerror:
        pytest.fail(f"SSD None-VOC loading failed: {valuerror}")


def test_SSD300_VOC_VGG():
    """Test SSD300 with VOC-VGG weights."""
    with pytest.raises(NotImplementedError):
        SSD300(num_classes=21, base_weights="VGG", head_weights="VOC")


def test_SSD300_VOC_None():
    """Test SSD300 with VOC-None weights."""
    with pytest.raises(NotImplementedError):
        SSD300(num_classes=2, base_weights=None, head_weights="VOC")


def test_SSD300_FAT_VGG():
    """Test SSD300 with FAT-VGG weights."""
    with pytest.raises(NotImplementedError):
        SSD300(num_classes=21, base_weights="VGG", head_weights="FAT")


def test_SSD300_FAT_None():
    """Test SSD300 with FAT-None weights."""
    with pytest.raises(NotImplementedError):
        SSD300(num_classes=2, base_weights=None, head_weights="FAT")


if __name__ == "__main__":
    pytest.main([__file__])
