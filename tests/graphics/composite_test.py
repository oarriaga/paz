import jax.numpy as jp

from paz.graphics import composite


def test_compute_scene_hit_mask():
    hit_masks = jp.array([[False, True, False], [False, False, True]])
    expected = jp.array([False, True, True])
    result = composite.compute_scene_hit_mask(hit_masks)
    assert jp.array_equal(result, expected)


def test_select_colors():
    depths = jp.array([[10.0, 2.0], [5.0, 8.0]])
    colors = jp.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]])
    expected = jp.array([[0, 0, 1], [0, 1, 0]])
    result = composite.select_colors(depths, colors)
    assert jp.array_equal(result, expected)


def test_find_closest_intersection_args():
    hit_masks = jp.array([[True, False], [False, True]])
    depths = jp.array([[1.0, 10.0], [10.0, 2.0]])
    indices = composite.find_closest_intersection_args(hit_masks, depths)
    assert jp.array_equal(indices, jp.array([0, 1]))


def test_take_closest():
    array = jp.array(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]
    )
    indices = jp.array([0, 1, 0])
    expected = jp.array([[1, 1, 1], [5, 5, 5], [3, 3, 3]])
    result = composite.take_closest(array, indices)
    assert jp.array_equal(result, expected)
