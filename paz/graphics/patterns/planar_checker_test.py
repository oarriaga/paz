import jax.numpy as jp

from paz.graphics.patterns import planar_checker


def test_compute_colors_alternates_across_unit_cell():
    color_A = jp.array([1.0, 0.0, 0.0])
    color_B = jp.array([0.0, 0.0, 1.0])
    points = jp.array(
        [[0.2, 0, 0.2], [0.8, 0, 0.2], [0.2, 0, 0.8], [0.8, 0, 0.8]]
    )
    expected_colors = jp.vstack([color_A, color_B, color_B, color_A])
    actual_colors = planar_checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)
