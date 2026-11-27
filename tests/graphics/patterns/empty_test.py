import jax.numpy as jp

from paz.graphics.patterns import empty


def test_compute_colors_returns_black():
    points = jp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = empty.compute_colors(points, None)
    assert jp.allclose(colors, jp.zeros_like(points))
