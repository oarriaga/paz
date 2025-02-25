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


def test_SSD512_COCO_COCO():
    """Test SSD512 with COCO-COCO weights."""
    try:
        SSD512(num_classes=81, base_weights="COCO", head_weights="COCO")
    except ValueError as valuerror:
        pytest.fail(f"SSD COCO-COCO loading failed: {valuerror}")


def test_SSD512_OIV6Hand_OIV6Hand():
    """Test SSD512 with OIV6Hand-OIV6Hand weights."""
    try:
        SSD512(num_classes=2, base_weights="OIV6Hand", head_weights="OIV6Hand")
    except ValueError as valuerror:
        pytest.fail(f"SSD OIV6Hand-OIV6Hand loading failed: {valuerror}")


def test_SSD512_YCBVideo_COCO():
    """Test SSD512 with YCBVideo-COCO weights."""
    try:
        SSD512(num_classes=22, base_weights="COCO", head_weights="YCBVideo")
    except ValueError as valuerror:
        pytest.fail(f"SSD YCBVideo-COCO loading failed: {valuerror}")


def test_SSD512_OIV6Hand_COCO():
    """Test SSD512 with OIV6Hand-COCO weights."""
    try:
        SSD512(num_classes=2, base_weights="COCO", head_weights="OIV6Hand")
    except ValueError as valuerror:
        pytest.fail(f"SSD OIV6Hand-COCO loading failed: {valuerror}")


def test_SSD512_YCBVideo_OIV6Hand():
    """Test SSD512 with YCBVideo-OIV6Hand weights."""
    with pytest.raises(NotImplementedError):
        SSD512(num_classes=22, base_weights="OIV6Hand", head_weights="YCBVideo")


def test_SSD512_None_OIV6Hand():
    """Test SSD512 with None-OIV6Hand weights."""
    with pytest.raises(NotImplementedError):
        SSD512(num_classes=2, base_weights="OIV6Hand", head_weights=None)


def test_SSD512_COCO_OIV6Hand():
    """Test SSD512 with None-OIV6Hand weights."""
    with pytest.raises(NotImplementedError):
        SSD512(num_classes=2, base_weights="OIV6Hand", head_weights="COCO")


if __name__ == "__main__":
    pytest.main([__file__])
