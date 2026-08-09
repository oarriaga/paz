import jax.numpy as jp

from paz.graphics.patterns import checker


def test_compute_colors_alternates_by_cell_parity():
    color_A = jp.array([1.0, 0.0, 0.0])
    color_B = jp.array([0.0, 0.0, 1.0])
    points = jp.array(
        [
            [0.1, 0.2, 0.3],  # sum(floor) = 0 (even) -> A
            [1.1, 0.2, 0.3],  # sum(floor) = 1 (odd)  -> B
            [-0.1, -0.8, 0.1],  # sum(floor) = -2 (even) -> A
            [2.0, 2.0, 2.0],  # sum(floor) = 6 (even) -> A
        ]
    )
    expected_colors = jp.vstack([color_A, color_B, color_A, color_A])
    actual_colors = checker.compute_colors(points, color_A, color_B)
    assert jp.allclose(actual_colors, expected_colors)
