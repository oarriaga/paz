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
