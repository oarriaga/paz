import pytest
import jax
import jax.numpy as jp
import paz
import numpy as np


@pytest.fixture
def boxes_A():
    return jp.array(
        [
            [54, 66, 198, 114],
            [42, 78, 186, 126],
            [18, 63, 235, 135],
            [18, 63, 235, 135],
            [54, 72, 198, 120],
            [36, 60, 180, 108],
        ]
    )


@pytest.fixture
def boxes_B():
    return jp.array(
        [
            [39, 63, 203, 112],
            [49, 75, 203, 125],
            [31, 69, 201, 125],
            [50, 72, 197, 121],
            [35, 51, 196, 110],
        ]
    )


@pytest.fixture
def true_IOUs():
    return jp.array([0.48706725, 0.787838, 0.70033113, 0.70739083, 0.39040922])


@pytest.fixture
def boxes_D_corner_form():
    return jp.array([[0.0, 0.0, 2.0, 2.0]])


@pytest.fixture
def boxes_D_center_form():
    return jp.array([[1.0, 1.0, 2.0, 2.0]])


def test_compute_IOUs_self_intersection_A(boxes_A):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_A, boxes_A)
    assert jp.allclose(1.0, jp.diag(pred_IOUs))


def test_compute_IOUs_self_intersection_B(boxes_B):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_B, boxes_B)
    assert jp.allclose(1.0, jp.diag(pred_IOUs))


def test_compute_IOUs_shape(boxes_A, boxes_B):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_A, boxes_B)
    assert len(pred_IOUs.shape) == 2
    num_rows, num_cols = pred_IOUs.shape
    assert num_rows == len(boxes_A)
    assert num_cols == len(boxes_B)


def test_compute_ious(boxes_A, boxes_B, true_IOUs):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_A[1:2, :], boxes_B)
    assert jp.allclose(true_IOUs, pred_IOUs)


def test_compute_generalized_IOUs_self_intersection(boxes_A):
    IOUs = paz.boxes.compute_generalized_IOUs(boxes_A, boxes_A)
    assert jp.allclose(1.0, jp.diag(IOUs))


def test_compute_generalized_IOUs_match_IOUs_when_nested():
    """A box inside another shares its hull, so both measures agree."""
    inner = jp.array([[10.0, 10.0, 20.0, 20.0]])
    outer = jp.array([[0.0, 0.0, 40.0, 40.0]])
    IOUs = paz.boxes.compute_IOUs(inner, outer)
    generalized = paz.boxes.compute_generalized_IOUs(inner, outer)
    assert jp.allclose(IOUs, generalized)


def test_compute_generalized_IOUs_are_negative_when_far_apart():
    near = jp.array([[0.0, 0.0, 10.0, 10.0]])
    far = jp.array([[100.0, 100.0, 110.0, 110.0]])
    assert paz.boxes.compute_generalized_IOUs(near, far)[0, 0] < 0.0
    assert paz.boxes.compute_IOUs(near, far)[0, 0] == 0.0


def test_compute_generalized_IOUs_shape(boxes_A, boxes_B):
    IOUs = paz.boxes.compute_generalized_IOUs(boxes_A, boxes_B)
    assert IOUs.shape == (len(boxes_A), len(boxes_B))


def test_to_center_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_center_form(boxes_D_corner_form)
    assert jp.allclose(boxes_D_center_form, values)


def test_to_corner_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_corner_form(boxes_D_center_form)
    assert jp.allclose(boxes_D_corner_form, values)


def test_square_extends_the_shorter_side_around_the_center():
    boxes = jp.array([[0, 0, 4, 2]])
    expected = jp.array([[0, -1, 4, 3]])
    assert jp.array_equal(paz.boxes.square(boxes), expected)


def test_clip_limits_boxes_to_the_last_pixel():
    boxes = jp.array([[-10, -5, 150, 200]])
    expected = jp.array([[0, 0, 119, 99]])
    assert jp.array_equal(paz.boxes.clip(boxes, 100, 120), expected)


def test_denormalize_scales_x_by_width_and_y_by_height():
    boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    expected = jp.array([[100, 50, 200, 100]])
    assert jp.array_equal(paz.boxes.denormalize(boxes, 100, 200), expected)


def test_flip_left_right():
    boxes = jp.array([[10.0, 20.0, 30.0, 40.0]])
    width = 100
    flipped = paz.boxes.flip_left_right(boxes, width)
    expected = jp.array([[70.0, 20.0, 90.0, 40.0]])
    assert jp.allclose(flipped, expected)


def test_normalize_inverts_denormalize():
    boxes = jp.array([[0.5, 0.25, 1.0, 0.75]])
    image_boxes = paz.boxes.denormalize(boxes, 100, 200)
    assert jp.allclose(paz.boxes.normalize(image_boxes, 100, 200), boxes)


def test_pad_when_boxes_are_smaller():
    """
    Tests the padding case where num_boxes < size.
    """
    # 1. Arrange: Set up the inputs
    input_boxes = jp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    target_size = 5
    pad_value = -1

    # 2. Act: Call the function
    result = paz.boxes.pad(input_boxes, target_size, pad_value)

    # 3. Assert: Check if the output is correct
    expected_output = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ]
    )

    assert result.shape == (5, 4), "Output shape is incorrect"
    np.testing.assert_array_equal(result, expected_output)


def test_pad_when_boxes_are_larger():
    """
    Tests the truncating case where num_boxes > size.
    """
    # 1. Arrange
    input_boxes = jp.arange(20).reshape((5, 4))
    target_size = 3
    pad_value = -1

    # 2. Act
    result = paz.boxes.pad(input_boxes, target_size, pad_value)

    # 3. Assert
    expected_output = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    assert result.shape == (3, 4), "Output shape is incorrect"
    np.testing.assert_array_equal(result, expected_output)


def test_pad_when_boxes_are_equal_size():
    """
    Tests the case where num_boxes == size (no change expected).
    """
    # 1. Arrange
    input_boxes = jp.arange(12).reshape((3, 4))
    target_size = 3

    # 2. Act
    result = paz.boxes.pad(input_boxes, target_size)

    # 3. Assert
    assert result.shape == (3, 4), "Output shape is incorrect"
    # The result should be identical to the input
    np.testing.assert_array_equal(result, input_boxes)


def test_pad_with_empty_input():
    """
    Tests the edge case of an empty input array.
    """
    # 1. Arrange
    input_boxes = jp.empty((0, 4))
    target_size = 3
    pad_value = 0

    # 2. Act
    result = paz.boxes.pad(input_boxes, target_size, value=pad_value)

    # 3. Assert
    expected_output = np.zeros((3, 4))
    assert result.shape == (3, 4), "Output shape is incorrect"
    np.testing.assert_array_equal(result, expected_output)


def test_pad_with_custom_value():
    """
    Tests that a custom padding value is applied correctly.
    """
    # 1. Arrange
    input_boxes = jp.ones((2, 4))
    target_size = 4
    pad_value = 99

    # 2. Act
    result = paz.boxes.pad(input_boxes, target_size, value=pad_value)

    # 3. Assert
    expected_output = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [99, 99, 99, 99], [99, 99, 99, 99]]
    )
    assert result.shape == (4, 4), "Output shape is incorrect"
    np.testing.assert_array_equal(result, expected_output)


def test_sample_negatives_picks_from_all_trials():
    key = jax.random.PRNGKey(0)
    boxes = jp.array([[10.0, 10.0, 50.0, 50.0]])
    negatives = paz.boxes.sample_negatives(key, boxes, 100, 100, 0.3, 20, 50)
    trials = paz.boxes.sample(key, 100, 100, 0.3, 0.3, 50)
    mean_IOUs = jp.mean(paz.boxes.compute_IOUs(trials, boxes), axis=1)
    expected = trials[jp.argsort(mean_IOUs)[:20]]
    assert negatives.shape == (20, 4)
    assert np.array_equal(negatives, expected)
