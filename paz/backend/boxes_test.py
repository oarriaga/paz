import pytest
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


def test_to_center_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_center_form(boxes_D_corner_form)
    assert jp.allclose(boxes_D_center_form, values)


def test_to_corner_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_corner_form(boxes_D_center_form)
    assert jp.allclose(boxes_D_corner_form, values)


@pytest.mark.skip(reason="changed implementation")
def test_to_one_hot():
    class_indices = jp.array([1, 3])
    one_hot = to_one_hot(class_indices, 4)
    expected = jp.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    assert jp.allclose(one_hot, expected)


@pytest.mark.skip(reason="changed implementation")
def test_make_box_square():
    box = (0, 0, 4, 2)
    square_box = paz.boxes.square(jp.array(box))
    # Expected to adjust y coordinates to match width
    expected = (0, -1, 4, 3)
    assert jp.array_equal(square_box, jp.array(expected))


@pytest.mark.skip(reason="changed implementation")
def test_offset():
    coords = (10, 20, 30, 40)
    offset_scales = (0.1, 0.1)
    new_coords = offset(coords, offset_scales)
    # x offset is (30-10)*0.1 = 2, y offset is (40-20)*0.1 = 2
    # x_min -2, x_max +2; y_min -2, y_max +2
    expected = (8, 18, 32, 42)
    assert jp.array_equal(new_coords, jp.array(expected))


@pytest.mark.skip(reason="changed implementation")
def test_clip():
    coords = (-10, -5, 150, 200)
    image_shape = (100, 100)
    clipped = clip(coords, image_shape)
    expected = (0, 0, 100, 100)
    assert jp.array_equal(clipped, jp.array(expected))


@pytest.mark.skip(reason="changed implementation")
def test_denormalize_box():
    box = jp.array([0.5, 0.5, 1.0, 1.0])
    image_shape = (100, 200)
    denorm_box = denormalize_box(box, image_shape)
    expected = (100, 50, 200, 100)
    assert jp.array_equal(denorm_box, jp.array(expected))


@pytest.mark.skip(reason="changed implementation")
def test_flip_left_right():
    boxes = jp.array([[10.0, 20.0, 30.0, 40.0]])
    width = 100
    flipped = flip_left_right(boxes, width)
    expected = jp.array([[70.0, 20.0, 90.0, 40.0]])
    assert jp.allclose(flipped, expected)


@pytest.mark.skip(reason="changed implementation")
def test_coordinate_conversions():
    image = jp.zeros((100, 200, 3))
    boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    image_boxes = to_image_coordinates(boxes, image)
    normalized_boxes = to_normalized_coordinates(image_boxes, image)
    assert jp.allclose(normalized_boxes, boxes, atol=1e-4)


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
