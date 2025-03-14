import pytest
import jax
import jax.numpy as jnp
from build.lib.paz.backend.boxes import (
    to_center_form,
    to_corner_form,
    encode,
    decode,
    compute_ious,
    compute_max_matches,
    get_matches_masks,
    mask_classes,
    match,
    match2,
    compute_iou,
    apply_non_max_suppression,
    nms_per_class,
    to_one_hot,
    make_box_square,
    offset,
    clip,
    denormalize_box,
    flip_left_right,
    to_image_coordinates,
    to_normalized_coordinates,
)


# Test to_center_form and to_corner_form
def test_to_center_form():
    boxes = jnp.array([[0.0, 0.0, 2.0, 2.0]])
    center_boxes = to_center_form(boxes)
    expected = jnp.array([[1.0, 1.0, 2.0, 2.0]])
    assert jnp.allclose(center_boxes, expected)


def test_to_corner_form():
    boxes = jnp.array([[1.0, 1.0, 2.0, 2.0]])
    corner_boxes = to_corner_form(boxes)
    expected = jnp.array([[0.0, 0.0, 2.0, 2.0]])
    assert jnp.allclose(corner_boxes, expected)


# Test encode and decode functions
def test_encode_decode():
    matched = jnp.array([[0.0, 0.0, 2.0, 2.0, 1.0]])
    priors = to_center_form(matched[:, :4])
    encoded = encode(matched, priors)
    decoded = decode(encoded, priors)
    assert jnp.allclose(decoded[:, :4], matched[:, :4], atol=1e-4)


# Test compute_ious
def test_compute_ious():
    boxes_A = jnp.array([[0.0, 0.0, 2.0, 2.0]])
    boxes_B = jnp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    ious = compute_ious(boxes_A, boxes_B)
    expected = jnp.array([[1.0, 1 / 7]])
    assert jnp.allclose(ious, expected, atol=1e-4)


# Test compute_max_matches
def test_compute_max_matches():
    boxes = jnp.array([[0.0, 0.0, 2.0, 2.0]])
    prior_boxes = jnp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    iou, arg = compute_max_matches(boxes, prior_boxes)
    assert iou.shape == (2,)
    assert arg.shape == (2,)
    assert jnp.allclose(iou, jnp.array([1.0, 1 / 7]), atol=1e-4)


# Test get_matches_masks
def test_get_matches_masks():
    boxes = jnp.array(
        [[0.0, 0.0, 2.0, 2.0, 1.0]]
    )  # Ground truth in corner form
    # Prior boxes in CENTER FORM (x_center, y_center, width, height)
    prior_boxes = jnp.array(
        [
            [1.0, 1.0, 2.0, 2.0],  # Converts to [0, 0, 2, 2] in corner form
            [2.0, 2.0, 2.0, 2.0],
        ]
    )  # Converts to [1, 1, 3, 3] in corner form
    matched_arg, pos_mask, ignore_mask = get_matches_masks(
        boxes, prior_boxes, 0.5, 0.4
    )
    assert pos_mask[0] == True  # IoU is 1.0 (perfect overlap)
    assert pos_mask[1] == False  # IoU is 1/7 ≈ 0.14 (no overlap)
    assert ignore_mask[1] == False  # Negative mask (IoU < 0.4)


# Test mask_classes
def test_mask_classes():
    matched_boxes = jnp.array(
        [[0.0, 0.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 3.0, 2.0]]
    )
    positive_mask = jnp.array([True, False])
    ignoring_mask = jnp.array([False, False])
    masked = mask_classes(matched_boxes, positive_mask, ignoring_mask)
    assert masked[0, 4] == 1.0
    assert masked[1, 4] == 0.0


# Test match and match2 functions
def test_match():
    boxes = jnp.array([[0.0, 0.0, 2.0, 2.0, 1.0]])
    prior_boxes = jnp.array([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    matched = match(boxes, prior_boxes, 0.5, 0.4)
    assert matched.shape == (2, 5)
    assert (
        matched[0, 4] == 1.0
    )  # IoU is (1/7) which is ~0.142 <0.5, so negative


def test_match2():
    boxes = jnp.array([[0.0, 0.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 3.0, 2.0]])
    prior_boxes = jnp.array([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    matched = match2(boxes, prior_boxes, 0.5)
    assert matched.shape == (2, 5)
    # Check if class is set to 0 where IoU < threshold


# Test compute_iou
def test_compute_iou():
    box = jnp.array([0.0, 0.0, 2.0, 2.0])
    boxes = jnp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    ious = compute_iou(box, boxes)
    assert jnp.allclose(ious, jnp.array([1.0, 1 / 7]), atol=1e-4)


# Test apply_non_max_suppression
def test_apply_non_max_suppression():
    boxes = jnp.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]
    )
    scores = jnp.array([0.8, 0.9, 0.7])
    selected_indices, count = apply_non_max_suppression(
        boxes, scores, 0.5, 200
    )
    assert count == 2
    assert selected_indices[0] == 1 and selected_indices[1] == 2


def test_apply_non_max_suppression_empty():
    boxes = jnp.array([]).reshape(0, 4)
    scores = jnp.array([])
    indices, count = apply_non_max_suppression(boxes, scores)
    assert count == 0


# Test nms_per_class
def test_nms_per_class():
    box_data = jnp.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.1, 0.9],
            [0.5, 0.5, 1.5, 1.5, 0.8, 0.2],
            [2.0, 2.0, 3.0, 3.0, 0.7, 0.3],
        ]
    )
    output = nms_per_class(box_data, 0.5, 0.01, 200)
    assert output.shape == (2, 200, 5)
    # Check if class 1 (index 1) has the high-scoring box


# Test to_one_hot
def test_to_one_hot():
    class_indices = jnp.array([1, 3])
    one_hot = to_one_hot(class_indices, 4)
    expected = jnp.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    assert jnp.allclose(one_hot, expected)


# Test make_box_square
def test_make_box_square():
    box = (0, 0, 4, 2)
    square_box = make_box_square(jnp.array(box))
    # Expected to adjust y coordinates to match width
    expected = (0, -1, 4, 3)
    assert jnp.array_equal(square_box, jnp.array(expected))


# Test offset
def test_offset():
    coords = (10, 20, 30, 40)
    offset_scales = (0.1, 0.1)
    new_coords = offset(coords, offset_scales)
    # x offset is (30-10)*0.1 = 2, y offset is (40-20)*0.1 = 2
    # x_min -2, x_max +2; y_min -2, y_max +2
    expected = (8, 18, 32, 42)
    assert jnp.array_equal(new_coords, jnp.array(expected))


# Test clip
def test_clip():
    coords = (-10, -5, 150, 200)
    image_shape = (100, 100)
    clipped = clip(coords, image_shape)
    expected = (0, 0, 100, 100)
    assert jnp.array_equal(clipped, jnp.array(expected))


# Test denormalize_box
def test_denormalize_box():
    box = jnp.array([0.5, 0.5, 1.0, 1.0])
    image_shape = (100, 200)
    denorm_box = denormalize_box(box, image_shape)
    expected = (100, 50, 200, 100)
    assert jnp.array_equal(denorm_box, jnp.array(expected))


# Test flip_left_right
def test_flip_left_right():
    boxes = jnp.array([[10.0, 20.0, 30.0, 40.0]])
    width = 100
    flipped = flip_left_right(boxes, width)
    expected = jnp.array([[70.0, 20.0, 90.0, 40.0]])
    assert jnp.allclose(flipped, expected)


# Test to_image_coordinates and to_normalized_coordinates
def test_coordinate_conversions():
    image = jnp.zeros((100, 200, 3))
    boxes = jnp.array([[0.5, 0.5, 1.0, 1.0]])
    image_boxes = to_image_coordinates(boxes, image)
    normalized_boxes = to_normalized_coordinates(image_boxes, image)
    assert jnp.allclose(normalized_boxes, boxes, atol=1e-4)
