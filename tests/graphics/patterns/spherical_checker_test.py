import jax.numpy as jp

from paz.graphics.patterns import spherical_checker


def test_compute_colors_on_axis_points():
    color_A = jp.array([1.0, 0.0, 0.0])
    color_B = jp.array([0.0, 0.0, 1.0])
    points = jp.array(
        [
            [1.0, 0.0, 0.0],  # u=0.25, v=0.5 -> floor(4+8)=12 (even) -> A
            [0.0, 0.0, 1.0],  # u=0.5,  v=0.5 -> floor(8+8)=16 (even) -> A
            [0.0, 1.0, 0.0],  # u=0.5,  v=1.0 -> floor(8+16)=24 (even) -> A
        ]
    )
    expected_colors = jp.vstack([color_A, color_A, color_A])
    actual_colors = spherical_checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)
