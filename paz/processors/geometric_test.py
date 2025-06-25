import pytest
import jax.numpy as jp
import jax
from paz.processors.geometric import (
    RandomSampleCrop,
)


@pytest.fixture
def sample_data():
    """Fixture to generate sample image and detections."""
    key = jax.random.PRNGKey(42)
    image = jax.random.randint(key, (300, 300, 3), 0, 255, dtype=jp.uint8)
    detections = jp.array(
        [[50, 50, 200, 200, 1], [100, 100, 250, 250, 2]], dtype=jp.float32
    )
    return image, detections


@pytest.fixture
def image():
    key = jax.random.PRNGKey(42)
    return jax.random.randint(key, (300, 300, 3), 0, 255, dtype=jp.uint8)


@pytest.fixture
def detections():
    return jp.array(
        [[50, 50, 200, 200, 1], [100, 100, 250, 250, 2]], dtype=jp.float32
    )


def test_random_sample_crop(image, detections):
    # image, boxes = sample_data
    processor = RandomSampleCrop(probability=1.0)
    cropped_image, new_detections = processor.call(image, detections)
    assert isinstance(cropped_image, jp.ndarray)
    assert isinstance(new_detections, jp.ndarray)
    assert cropped_image.shape[-1] == 3
    # "Bounding box coordinates should be non-negative"
    assert jp.all(new_detections[:, :4] >= 0)
    # "x_max should be within image width"
    assert jp.all(new_detections[:, 2] <= cropped_image.shape[1])
    # "y_max should be within image height"
    assert jp.all(new_detections[:, 3] <= cropped_image.shape[0])


def test_no_crop_when_probability_is_zero(image, detections):
    processor = RandomSampleCrop(probability=0.0)
    cropped_image, new_detections = processor.call(image, detections)
    assert jp.array_equal(image, cropped_image)
    assert jp.array_equal(detections, new_detections)


def test_aspect_ratio_constraint(image, detections):
    processor = RandomSampleCrop(probability=1.0)
    for _ in range(10):
        cropped_image, _ = processor.call(image.copy(), detections.copy())
        H, W = cropped_image.shape[:2]
        aspect_ratio = H / W
        assert 0.5 <= aspect_ratio <= 2.0, "Aspect ratio out of bounds"


def test_boxes_clamped_and_shifted(image, detections):
    key = jax.random.PRNGKey(0)
    processor = RandomSampleCrop(probability=1.0)
    processor.key = key
    cropped_image, new_detections = processor.call(image, detections)
    # Ensure boxes are within the cropped image bounds
    if new_detections.size > 0:
        x_min, y_min = 0, 0  # Crop starts at (0,0) in shifted coordinates
        x_max, y_max = cropped_image.shape[1], cropped_image.shape[0]
        assert jp.all(new_detections[:, 0] >= x_min), "x_min out of bounds"
        assert jp.all(new_detections[:, 1] >= y_min), "y_min out of bounds"
        assert jp.all(new_detections[:, 2] <= x_max), "x_max out of bounds"
        assert jp.all(new_detections[:, 3] <= y_max), "y_max out of bounds"


def test_no_valid_crop_after_max_trials(image, detections):
    # Set impossible IoU constraints to trigger max trials
    processor = RandomSampleCrop(probability=1.0)
    processor.jaccard_min_max = [(0.95, 1.0)]  # High IoU unlikely to be met
    processor.max_trials = 1  # Reduce trials for quick test
    cropped_image, new_detections = processor.call(image, detections)
    # Should return original image and boxes
    assert jp.array_equal(cropped_image, image), "Image should be unchanged"
    assert jp.array_equal(
        new_detections, detections
    ), "Boxes should be unchanged"


def test_labels_integrity(image, detections):
    processor = RandomSampleCrop(probability=1.0)
    cropped_image, new_detections = processor.call(image, detections)
    if new_detections.size > 0:
        original_labels = detections[:, -1]
        new_labels = new_detections[:, -1]
        assert jp.all(jp.isin(new_labels, original_labels)), "Label mismatch"


def test_boxes_center_in_crop(image, detections):
    processor = RandomSampleCrop(probability=1.0)
    cropped_image, new_detections = processor.call(image, detections)
    if new_detections.size > 0:
        centers = (new_detections[:, :2] + new_detections[:, 2:4]) / 2
        H, W = cropped_image.shape[:2]
        assert jp.all(centers >= 0), "Centers should be >= 0"
        assert jp.all(centers[:, 0] <= W), "X centers exceed crop width"
        assert jp.all(centers[:, 1] <= H), "Y centers exceed crop height"
