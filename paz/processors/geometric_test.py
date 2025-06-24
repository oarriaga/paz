import pytest
import jax.numpy as jp
import jax
from paz.processors.geometric import (
    RandomSampleCrop,
)


@pytest.fixture
def sample_data():
    """Fixture to generate sample image and bounding boxes."""
    key = jax.random.PRNGKey(42)  # Set a seed for reproducibility
    image = jax.random.randint(key, (300, 300, 3), 0, 255, dtype=jp.uint8)

    boxes = jp.array(
        [[50, 50, 200, 200, 1], [100, 100, 250, 250, 2]], dtype=jp.float32
    )  # (x_min, y_min, x_max, y_max, label)

    return image, boxes


def test_random_sample_crop(sample_data):
    image, boxes = sample_data
    processor = RandomSampleCrop(probability=1.0)  # Always apply crop

    cropped_image, new_boxes = processor.call(image, boxes)

    # Check if the function returns the correct types
    assert isinstance(
        cropped_image, jp.ndarray
    ), "Cropped image should be a numpy array"
    assert isinstance(
        new_boxes, jp.ndarray
    ), "New bounding boxes should be a numpy array"

    # Check image shape is valid
    assert cropped_image.shape[-1] == 3, "Cropped image should have 3 channels"

    # Ensure bounding boxes are within the cropped image
    assert jp.all(
        new_boxes[:, :4] >= 0
    ), "Bounding box coordinates should be non-negative"
    assert jp.all(
        new_boxes[:, 2] <= cropped_image.shape[1]
    ), "x_max should be within image width"
    assert jp.all(
        new_boxes[:, 3] <= cropped_image.shape[0]
    ), "y_max should be within image height"


def test_no_crop_when_probability_is_zero(sample_data):
    image, boxes = sample_data
    processor = RandomSampleCrop(probability=0.0)  # No crop should happen

    cropped_image, new_boxes = processor.call(image, boxes)

    # Ensure output is identical to input
    assert jp.array_equal(image, cropped_image), "Image should remain unchanged"
    assert jp.array_equal(
        boxes, new_boxes
    ), "Bounding boxes should remain unchanged"


def test_aspect_ratio_constraint(sample_data):
    image, boxes = sample_data
    processor = RandomSampleCrop(probability=1.0)
    for _ in range(10):  # Test multiple random crops
        cropped_image, _ = processor.call(image.copy(), boxes.copy())
        H, W = cropped_image.shape[:2]
        aspect_ratio = H / W
        assert 0.5 <= aspect_ratio <= 2.0, "Aspect ratio out of bounds"


def test_boxes_clamped_and_shifted(sample_data):
    image, boxes = sample_data

    # Create a fixed random key for reproducibility
    key = jax.random.PRNGKey(0)

    # Instantiate the processor with a fixed probability
    processor = RandomSampleCrop(probability=1.0)

    # Override the processor's internal key (instead of passing it to `call`)
    processor.key = key

    # Call the processor (no need to pass the key explicitly)
    cropped_image, new_boxes = processor.call(image, boxes)

    # Ensure boxes are within the cropped image bounds
    if new_boxes.size > 0:
        x_min, y_min = 0, 0  # Crop starts at (0,0) in shifted coordinates
        x_max, y_max = cropped_image.shape[1], cropped_image.shape[0]

        assert jp.all(new_boxes[:, 0] >= x_min), "x_min out of bounds"
        assert jp.all(new_boxes[:, 1] >= y_min), "y_min out of bounds"
        assert jp.all(new_boxes[:, 2] <= x_max), "x_max out of bounds"
        assert jp.all(new_boxes[:, 3] <= y_max), "y_max out of bounds"


def test_no_valid_crop_after_max_trials(sample_data):
    image, boxes = sample_data
    # Set impossible IoU constraints to trigger max trials
    processor = RandomSampleCrop(probability=1.0)
    processor.jaccard_min_max = [(0.95, 1.0)]  # High IoU unlikely to be met
    processor.max_trials = 1  # Reduce trials for quick test
    cropped_image, new_boxes = processor.call(image, boxes)
    # Should return original image and boxes
    assert jp.array_equal(cropped_image, image), "Image should be unchanged"
    assert jp.array_equal(new_boxes, boxes), "Boxes should be unchanged"


def test_labels_integrity(sample_data):
    image, boxes = sample_data
    processor = RandomSampleCrop(probability=1.0)
    cropped_image, new_boxes = processor.call(image, boxes)
    if new_boxes.size > 0:
        original_labels = boxes[:, -1]
        new_labels = new_boxes[:, -1]
        # Ensure all new labels exist in the original labels
        assert jp.all(jp.isin(new_labels, original_labels)), "Label mismatch"


def test_boxes_center_in_crop(sample_data):
    image, boxes = sample_data
    processor = RandomSampleCrop(probability=1.0)
    cropped_image, new_boxes = processor.call(image, boxes)
    if new_boxes.size > 0:
        # Calculate centers of new boxes
        centers = (new_boxes[:, :2] + new_boxes[:, 2:4]) / 2
        H, W = cropped_image.shape[:2]
        # Check centers are within crop bounds
        assert jp.all(centers >= 0), "Centers should be >= 0"
        assert jp.all(centers[:, 0] <= W), "X centers exceed crop width"
        assert jp.all(centers[:, 1] <= H), "Y centers exceed crop height"
