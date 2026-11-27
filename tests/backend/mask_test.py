import jax.numpy as jp
import paz


def test_to_box_single_pixel():
    mask = jp.array([[0.0, 0.0], [0.0, 1.0]])
    expected = jp.array([1.0, 1.0, 1.0, 1.0])
    result = paz.mask.to_box(mask, 1.0)
    assert jp.allclose(result, expected)


def test_to_box_rectangle_region():
    mask = jp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    expected = jp.array([1.0, 1.0, 3.0, 2.0])
    result = paz.mask.to_box(mask, 1.0)
    assert jp.allclose(result, expected)


def test_to_box_full_mask():
    mask = jp.ones((2, 3))
    expected = jp.array([0.0, 0.0, 2.0, 1.0])
    result = paz.mask.to_box(mask, 1.0)
    assert jp.allclose(result, expected)


def test_to_box_zero_value_mask():
    mask = jp.array([[1.0, 1.0], [0.0, 0.0]])
    expected = jp.array([0.0, 1.0, 1.0, 1.0])
    result = paz.mask.to_box(mask, 0.0)
    assert jp.allclose(result, expected)


def test_to_box_float_mask_value():
    mask = jp.array([[0.5, 0.0], [0.5, 0.0]])
    expected = jp.array([0.0, 0.0, 0.0, 1.0])
    result = paz.mask.to_box(mask, 0.5)
    assert jp.allclose(result, expected)


def test_to_rgb_from_normalized_image():
    image = jp.array(
        [
            [[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = jp.array([[1.0, 0.0], [0.0, 0.0]])
    result = paz.mask.to_rgb(image, mask)
    expected = jp.array([0.0, 0.5, 1.0])
    assert jp.allclose(result, expected)


def test_to_rgb_accepts_channel_mask():
    image = jp.array(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = jp.array([[[1.0], [0.0]], [[0.0], [0.0]]])
    result = paz.mask.to_rgb(image, mask)
    expected = jp.array([0.0, 0.0, 1.0])
    assert jp.allclose(result, expected)


def test_to_RGB_returns_uint8():
    image = jp.array(
        [
            [[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = jp.array([[1.0, 0.0], [0.0, 0.0]])
    result = paz.mask.to_RGB(image, mask)
    expected = jp.array([0, 127, 255], dtype=jp.uint8)
    assert jp.array_equal(result, expected)


def test_erode_shrinks_square_by_one_pixel_per_side():
    mask = jp.zeros((9, 9), bool).at[2:7, 2:7].set(True)
    eroded = paz.mask.erode(mask, 1)
    expected = jp.zeros((9, 9), bool).at[3:6, 3:6].set(True)
    assert jp.array_equal(eroded, expected)


def test_erode_by_zero_pixels_keeps_mask():
    mask = jp.zeros((5, 5), bool).at[1:4, 2].set(True)
    assert jp.array_equal(paz.mask.erode(mask, 0), mask)


def test_erode_removes_single_pixel():
    mask = jp.zeros((5, 5), bool).at[2, 2].set(True)
    assert not bool(paz.mask.erode(mask, 1).any())


def test_erode_keeps_mask_touching_image_border():
    mask = jp.ones((4, 4), bool)
    assert bool(paz.mask.erode(mask, 1).all())


def test_compute_IOU_identical_masks_is_one():
    mask = jp.array([[True, False], [True, True]])
    assert jp.isclose(paz.mask.compute_IOU(mask, mask), 1.0)


def test_compute_IOU_disjoint_masks_is_zero():
    mask_A = jp.array([[True, False]])
    mask_B = jp.array([[False, True]])
    assert jp.isclose(paz.mask.compute_IOU(mask_A, mask_B), 0.0)


def test_compute_IOU_partial_overlap():
    mask_A = jp.array([[True, True, False]])
    mask_B = jp.array([[False, True, True]])
    assert jp.isclose(paz.mask.compute_IOU(mask_A, mask_B), 1.0 / 3.0)


def test_compute_IOU_accepts_float_masks():
    mask_A = jp.array([[1.0, 1.0, 0.0]])
    mask_B = jp.array([[0.0, 1.0, 1.0]])
    assert jp.isclose(paz.mask.compute_IOU(mask_A, mask_B), 1.0 / 3.0)
