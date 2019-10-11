import numpy as np

from paz.core.ops import compute_iou
from paz.core.ops import compute_ious
from paz.core.ops import denormalize_box
from paz.core.ops import to_point_form
from paz.core.ops import to_center_form
from paz.core.ops import rotation_matrix_to_quaternion
from paz.core.ops import quaternion_to_rotation_matrix


boxes_B = np.array([[39, 63, 203, 112],
                    [49, 75, 203, 125],
                    [31, 69, 201, 125],
                    [50, 72, 197, 121],
                    [35, 51, 196, 110]])

boxes_A = np.array([[54, 66, 198, 114],
                    [42, 78, 186, 126],
                    [18, 63, 235, 135],
                    [18, 63, 235, 135],
                    [54, 72, 198, 120],
                    [36, 60, 180, 108]])

target = [0.48706725, 0.787838, 0.70033113, 0.70739083, 0.39040922]


affine_matrix = np.array(
    [[-0., -0.994522, 0.104528, 3.135854],
     [0., 0.104528, 0.994522, 29.835657],
     [-1., 0., 0., 0.],
     [0., 0., -0., 1.]])


quaternion_target = np.array([0.525483, -0.473147, 0.525483, 0.473147])


def test_compute_iou():
    result = compute_iou(boxes_A[1, :], boxes_B)
    assert np.allclose(result, target)


def test_compute_ious_shape():
    ious = compute_ious(boxes_A, boxes_B)
    target_shape = (boxes_A.shape[0], boxes_B.shape[0])
    assert ious.shape == target_shape


def test_compute_ious():
    result = compute_ious(boxes_A[1:2, :], boxes_B)
    assert np.allclose(result, target)


def test_denormalize_box():
    box = [.1, .2, .3, .4]
    box = denormalize_box(box, (200, 300))
    assert(box == (30, 40, 90, 80))


def test_to_center_form_inverse():
    assert np.all(to_point_form(to_center_form(boxes_A)) == boxes_A)


def test_to_point_form_inverse():
    assert np.all(to_point_form(to_center_form(boxes_A)) == boxes_A)


def test_to_center_form():
    boxes = to_center_form(boxes_A)
    boxes_A_result = to_point_form(boxes)
    print(boxes_A_result == boxes_A)
    print(boxes)


def test_rotation_matrix_to_quaternion():
    result = rotation_matrix_to_quaternion(affine_matrix[:3, :3])
    assert np.allclose(result, quaternion_target)


def test_rotation_matrix_to_quaternion_inverse():
    quaternion = rotation_matrix_to_quaternion(affine_matrix[:3, :3])
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix, affine_matrix[:3, :3])


test_compute_iou()
test_compute_ious()
test_compute_ious_shape()
test_denormalize_box()
test_to_center_form_inverse()
test_to_point_form_inverse()
test_rotation_matrix_to_quaternion()
test_rotation_matrix_to_quaternion_inverse()
