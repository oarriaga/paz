import jax.numpy as jp
from pytest import approx

from paz.graphics.patterns import spherical


def build_four_color_image():
    top_row = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    bottom_row = [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    return jp.array([top_row, bottom_row])


def test_spherical_map_on_poles_and_equator():
    poles = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    equator = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    equator += [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    u, v = spherical.spherical_map(jp.array(poles + equator))
    assert v[0] == approx(1.0)
    assert v[1] == approx(0.0)
    assert v[2] == approx(0.5)
    assert u[3] == approx(0.5)
    assert u[2] == approx(0.25)
    assert u[5] == approx(0.0)


def test_compute_colors_samples_image_at_x_axis():
    four_colors = build_four_color_image()
    points = jp.array([[1.0, 0.0, 0.0]])
    actual_color = spherical.compute_colors(points, four_colors)
    assert jp.allclose(actual_color, four_colors[0, 0])
